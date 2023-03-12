from flax import linen as nn  # Linen API
from flax.training.train_state import TrainState
from functools import partial
import jax
from jax import Array, jit, random
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple

from src.structs import TaskCallables
from src.training.initialization import initialize_train_state
from src.training.optim import create_learning_rate_fn


@partial(
    jit,
    static_argnums=(2, 3),
    static_argnames=("learning_rate_fn", "task_callables"),
)
def train_step(
    state: TrainState,
    batch: Dict[str, Array],
    task_callables: TaskCallables,
    learning_rate_fn: Callable,
) -> Tuple[TrainState, Dict[str, Array]]:
    """
    Trains the neural network for one step.
    Args:
        state: training state of the neural network
        batch: dictionary of batch data
        task_callables: struct containing the functions for the learning task
        learning_rate_fn: A function that takes the current step and returns the current learning rate.
            It has the signature learning_rate_fn(step: int) -> lr.
    Returns:
        state: updated training state of the neural network
        metrics: Dictionary of training metrics
    """
    loss_fn = partial(task_callables.loss_fn, batch)
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, preds), grad_nn_params = grad_fn(state.params)

    # optimize the neural network parameters with gradient descent
    state = state.apply_gradients(grads=grad_nn_params)

    # compute metrics
    metrics = task_callables.compute_metrics_fn(batch, preds)
    metrics["loss"] = loss
    # save the currently active learning rates to the `metrics` dictionary
    metrics["lr"] = learning_rate_fn(state.step)

    return state, metrics


@partial(
    jit,
    static_argnums=(2, ),
    static_argnames=("task_callables", ),
)
def eval_step(
    state: TrainState, batch: Dict[str, Array], task_callables: TaskCallables,
) -> Dict[str, jnp.ndarray]:
    """
    One validation step of the neural networks.
    Args:
        state: training state of the neural network
        batch: dictionary of batch data
        task_callables: struct containing the functions for the learning task
    Returns:
        metrics: Dictionary of validation metrics
    """
    loss, preds = task_callables.loss_fn(batch, state.params)

    # compute metrics
    metrics = task_callables.compute_metrics_fn(batch, preds)
    metrics["loss"] = loss

    return metrics


def train_epoch(
    epoch: int,
    state: TrainState,
    train_ds: tf.data.Dataset,
    task_callables: TaskCallables,
    learning_rate_fn: Callable,
) -> Tuple[TrainState, float, Dict[str, float]]:
    """
    Train for a single epoch.
    Args:
        epoch: Index of current epoch.
        state: training state of the neural network
        train_ds: Training dataset as tf.data.Dataset object.
        task_callables: struct containing the functions for the learning task
        learning_rate_fn: A function that takes the current step and returns the current learning rate.
            It has the signature learning_rate_fn(step: int) -> lr.
    Returns:
        state: updated training state of the neural network
        train_loss: Training loss of the current epoch.
        train_metrics: Dictionary of training metrics.
    """
    steps_per_epoch = train_ds.cardinality().numpy()

    step_metrics_list = []
    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        print("step: ", step, "/", steps_per_epoch)
        states, step_metrics = train_step(state, batch, task_callables, learning_rate_fn)
        step_metrics_list.append(step_metrics)

    # compute mean of metrics across each batch in epoch.
    step_metrics_np = jax.device_get(step_metrics_list)
    epoch_metrics_np = {
        k: onp.mean(jnp.array([metrics[k] for metrics in step_metrics_np])).item()
        for k in step_metrics_np[0]
    }  # jnp.mean does not work on lists

    return state, epoch_metrics_np["loss"], epoch_metrics_np


def eval_model(
    state: TrainState,
    eval_ds: tf.data.Dataset,
    task_callables: TaskCallables,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """
    Validate the model on the validation dataset.
    Args:
        state: training state of the neural network
        eval_ds: Evaluation dataset as tf.data.Dataset object.
        task_callables: struct containing the functions for the learning task
    Returns:
        val_loss: Validation loss.
        val_metrics: Dictionary of metrics.
    """
    step_metrics_list = []
    for step, batch in enumerate(eval_ds.as_numpy_iterator()):
        step_metrics = eval_step(state, batch, task_callables)
        step_metrics_list.append(step_metrics)

    # compute mean of metrics across each batch in epoch.
    step_metrics_np = jax.device_get(step_metrics_list)
    eval_metrics = {
        k: onp.mean(jnp.array([metrics[k] for metrics in step_metrics_np])).item()
        for k in step_metrics_np[0]
    }  # jnp.mean does not work on lists

    return eval_metrics["loss"], eval_metrics


def run_training(
    rng: random.PRNGKey,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    nn_model: nn.Module,
    task_callables: TaskCallables,
    num_epochs: int,
    batch_size: int,
    base_lr: float,
    warmup_epochs: int = 0,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> Tuple[
    Array,
    List[Dict[str, jnp.ndarray]],
    List[Dict[str, jnp.ndarray]],
    List[TrainState],
]:
    """
    Run the training loop.
    Args:
        rng: PRNG key for pseudo-random number generation.
        train_ds: Dictionary of jax arrays containing the training dataset.
        val_ds: Dictionary of jax arrays containing the validation dataset.
        nn_model: Neural network model.
        task_callables: struct containing the functions for the learning task
        num_epochs: Number of epochs to train for.
        batch_size: The size of a minibatch (i.e. number of samples in a batch).
        base_lr: Base learning rate (after warmup and before decay).
        warmup_epochs: Number of epochs for warmup.
        weight_decay: Weight decay.
        verbose: If True, print the training progress.
    Returns:
        val_loss_history: Array of validation losses for each epoch.
        train_metrics_history: List of dictionaries containing the training metrics for each epoch.
        val_metrics_history: List of dictionaries containing the validation metrics for each epoch.
        train_states_history: List of dictionaries containing the training states for each epoch.
    """
    # initialize the learning rate scheduler
    lr_fn = create_learning_rate_fn(
        num_epochs,
        steps_per_epoch=len(train_ds) // batch_size,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs
    )

    # extract dummy batch from dataset
    nn_dummy_batch = next(train_ds.as_numpy_iterator())
    # assemble input for dummy batch
    nn_dummy_input = task_callables.assemble_input_fn(nn_dummy_batch)

    # initialize the train state
    state = initialize_train_state(
        rng,
        nn_model,
        nn_dummy_input=nn_dummy_input,
        learning_rate_fn=lr_fn,
        weight_decay=weight_decay,
    )

    # initialize the lists for the training history
    val_loss_history = []  # list with validation losses
    train_metrics_history = []  # list with train metric dictionaries
    val_metrics_history = []  # list with validation metric dictionaries
    state_history = []  # list with dictionaries of model states

    if verbose:
        print(f"Training the Lagrangian neural network for {num_epochs} epochs...")

    for epoch in (pbar := tqdm(range(1, num_epochs + 1))):
        # Run the training for the current epoch
        state, train_loss, train_metrics = train_epoch(
            epoch, state, train_ds, task_callables, lr_fn
        )

        # Evaluate the current set of neural network parameters on the validation set
        val_loss, val_metrics = eval_model(state, val_ds, task_callables)

        # Save the model parameters and the validation loss for the current epoch
        val_loss_history.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        state_history.append(state)

        if verbose:
            pbar.set_description(
                "Epoch: %d, lr: %.6f, train loss: %.7f, val loss: %.7f"
                % (epoch, train_metrics["lr"], train_loss, val_loss)
            )

    # array of shape (num_epochs, ) with the validation losses of each epoch
    val_loss_history = jnp.array(val_loss_history)

    return val_loss_history, train_metrics_history, val_metrics_history, state_history
