from flax.training.train_state import TrainState
from functools import partial
import jax
from jax import Array, jit, random
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
from typing import Callable, Dict, List, Tuple

from src.structs import TaskCallables


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


@jit
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
    states: Dict[str, TrainState],
    train_ds: tf.data.Dataset,
    batch_size: int,
    epoch: int,
    learning_rate_fn: Callable,
    rng: random.KeyArray,
) -> Tuple[Dict[str, TrainState], float, Dict[str, float]]:
    """
    Train for a single epoch.
    Args:
        states: Dictionary containing the current states of the training of the two neural networks.
            Entries of dictionary:
                - MassMatrixNN: TrainState of the mass matrix neural network
                - PotentialEnergyNN: TrainState of the potential energy neural network
        train_ds: Dictionary containing the training dataset.
        batch_size: Batch size of training loop.
        epoch: Index of current epoch.
        learning_rate_fn: A function that takes the current step and returns the current learning rate.
            It has the signature learning_rate_fn(step: int) -> lr.
        rng: PRNG key for pseudo-random number generation.
    Returns:
        states: Dictionary of updated training states.
        train_loss: Training loss of the current epoch.
        train_metrics: Dictionary of training metrics.
    """
    train_ds_size = int(train_ds["th_curr_ss"].shape[0])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)  # get a randomized index array
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape(
        (steps_per_epoch, batch_size)
    )  # index array, where each row is a batch
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        states, metrics = train_step(states, batch, learning_rate_fn)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: onp.mean(jnp.array([metrics[k] for metrics in batch_metrics_np])).item()
        for k in batch_metrics_np[0]
    }  # jnp.mean does not work on lists

    return states, epoch_metrics_np["loss"], epoch_metrics_np


def eval_model(
    states: Dict[str, TrainState],
    val_ds: Dict[str, jnp.ndarray],
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """
    Validate the model on the validation dataset.
    Args:
        states: Dictionary containing the current states of the training of the two neural networks.
            Entries of dictionary:
                - MassMatrixNN: TrainState of the mass matrix neural network
                - PotentialEnergyNN: TrainState of the potential energy neural network
        val_ds: Dictionary containing the validation dataset.
    Returns:
        val_loss: Validation loss.
        val_metrics: Dictionary of metrics.
    """
    val_metrics = eval_step(states, val_ds)
    val_metrics = jax.device_get(val_metrics)
    val_metrics = jax.tree_util.tree_map(
        lambda x: x.item(), val_metrics
    )  # map the function over all leaves in metrics

    return val_metrics["loss"], val_metrics


def run_lnn_training(
    rng: random.PRNGKey,
    train_ds: Dict[str, jnp.ndarray],
    val_ds: Dict[str, jnp.ndarray],
    num_epochs: int,
    batch_size: int,
    base_lr: float,
    warmup_epochs: int = 0,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> Tuple[
    jnp.ndarray,
    List[Dict[str, jnp.ndarray]],
    List[Dict[str, jnp.ndarray]],
    List[Dict[str, TrainState]],
]:
    """
    Run the training loop.
    Args:
        rng: PRNG key for pseudo-random number generation.
        train_ds: Dictionary of jax arrays containing the training dataset.
        val_ds: Dictionary of jax arrays containing the validation dataset.
        num_epochs: Number of epochs to train for.
        batch_size: The size of a minibatch (i.e. number of samples in a batch).
        base_lr: Base learning rate (after warmup and before decay).
        warmup_epochs: Number of epochs for warmup.
        weight_decay: Weight decay.
        verbose: If True, print the training progress.
    Returns:
        val_loss_history: Array of validation losses for each epoch.
        val_metrics_history: List of dictionaries containing the validation metrics for each epoch.
        train_states_history: List of dictionaries containing the training states for each epoch.
    """

    # number of training samples
    num_train_samples = len(train_ds["th_curr_ss"])

    # initialize the learning rate scheduler
    learning_rate_fn = None
    ### BEGIN SOLUTION
    learning_rate_fn = create_learning_rate_fn(
        num_epochs, num_train_samples // batch_size, base_lr, warmup_epochs
    )
    ### END SOLUTION

    # split of PRNG keys
    # the 1st is used for training,
    # the 2nd to initialize the neural network weights.
    rng, init_train_states_rng = random.split(rng, 2)

    # initialize the train states
    states = initialize_train_states(
        init_train_states_rng, learning_rate_fn, weight_decay
    )

    # initialize the lists for the training history
    val_loss_history = []  # list with validation losses
    train_metrics_history = []  # list with train metric dictionaries
    val_metrics_history = []  # list with validation metric dictionaries
    states_history = []  # list with dictionaries of model states

    if verbose:
        print(f"Training the Lagrangian neural network for {num_epochs} epochs...")

    for epoch in (pbar := tqdm(range(1, num_epochs + 1))):
        # Split the `rng` PRNG key into two new keys
        # use the 1st PRNG to update the `rng` variable
        # store the 2nd PRNG key in the variable `epoch_rng`
        epoch_rng = None
        ### BEGIN SOLUTION
        rng, epoch_rng = random.split(rng)
        ### END SOLUTION

        # Run the training for the current epoch
        # Use the `epoch_rng` to randomly shuffle the batches
        train_loss, train_metrics = jnp.array(0.0), {}
        ### BEGIN SOLUTION
        states, train_loss, train_metrics = train_epoch(
            states, train_ds, batch_size, epoch, learning_rate_fn, epoch_rng
        )
        ### END SOLUTION

        # Evaluate the current set of neural network parameters on the validation set
        val_loss, val_metrics = jnp.array(0.0), {}
        ### BEGIN SOLUTION
        val_loss, val_metrics = eval_model(states, val_ds)
        ### END SOLUTION

        # Save the model parameters and the validation loss for the current epoch
        val_loss_history.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        states_history.append(states)

        if verbose:
            pbar.set_description(
                "Epoch: %d, lr: %.6f, train loss: %.7f, val loss: %.7f"
                % (epoch, train_metrics["lr_mass_matrix_nn"], train_loss, val_loss)
            )

    # array of shape (num_epochs, ) with the validation losses of each epoch
    val_loss_history = jnp.array(val_loss_history)

    return val_loss_history, train_metrics_history, val_metrics_history, states_history
