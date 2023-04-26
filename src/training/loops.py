import ciclo
from ciclo import History
from flax import linen as nn  # Linen API
from functools import partial
import jax
from jax import Array, debug, jit, random
import jax.numpy as jnp
import jax_metrics as jm
from pathlib import Path
import tensorflow as tf
from typing import Callable, Dict, List, Tuple

from src.structs import TaskCallables, TrainState
from src.training.checkpoint import OrbaxCheckpoint
from src.training.train_state_utils import initialize_train_state
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
) -> Tuple[ciclo.Logs, TrainState]:
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

    # extract the learning rate for the current step
    lr = learning_rate_fn(state.step)

    # compute metrics
    metrics_dict = task_callables.compute_metrics_fn(batch, preds)
    metrics = state.metrics.update(loss=loss, lr=lr, **metrics_dict)

    # save metrics to logs
    logs = ciclo.logs()
    logs.add_stateful_metrics(**metrics.compute())

    return logs, state.replace(metrics=metrics)


@partial(
    jit,
    static_argnums=(2,),
    static_argnames=("task_callables",),
)
def eval_step(
    state: TrainState,
    batch: Dict[str, Array],
    task_callables: TaskCallables,
) -> Tuple[ciclo.Logs, TrainState]:
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
    metrics_dict = task_callables.compute_metrics_fn(batch, preds)
    metrics = state.metrics.update(loss=loss, lr=jnp.zeros(()), **metrics_dict)

    # save metrics to logs
    logs = ciclo.logs()
    computed_metrics = metrics.compute()
    # delete the learning rate from the metrics
    del computed_metrics["lr"]
    logs.add_stateful_metrics(**computed_metrics)

    return logs, state


@jit
def reset_step(state: TrainState):
    """
    Reset the metrics of the training state.
    """
    return state.replace(metrics=state.metrics.reset())


def run_training(
    rng: random.PRNGKey,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    nn_model: nn.Module,
    task_callables: TaskCallables,
    metrics: jm.Metrics,
    num_epochs: int,
    base_lr: float,
    warmup_epochs: int = 0,
    weight_decay: float = 0.0,
    logdir: Path = None,
) -> Tuple[TrainState, History]:
    """
    Run the training loop.
    Args:
        rng: PRNG key for pseudo-random number generation.
        train_ds: Training dataset as tf.data.Dataset object.
        val_ds: Validation dataset as tf.data.Dataset object.
        nn_model: Neural network model.
        task_callables: struct containing the functions for the learning task
        metrics: Struct containing the metrics to be computed during training.
        num_epochs: Number of epochs to train for.
        base_lr: Base learning rate (after warmup and before decay).
        warmup_epochs: Number of epochs for warmup.
        weight_decay: Weight decay.
        logdir: Path to the directory where the training logs should be saved.
    Returns:
        val_loss_history: Array of validation losses for each epoch.
        train_metrics_history: List of dictionaries containing the training metrics for each epoch.
        val_metrics_history: List of dictionaries containing the validation metrics for each epoch.
        best_state: TrainState object of the model with the lowest validation loss.
    """
    steps_per_epoch = len(train_ds)
    num_total_train_steps = num_epochs * steps_per_epoch
    num_val_steps = len(val_ds)

    # initialize the learning rate scheduler
    lr_fn = create_learning_rate_fn(
        num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
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
        metrics=metrics,
        learning_rate_fn=lr_fn,
        weight_decay=weight_decay,
    )

    callbacks = []
    if logdir is not None:
        callbacks.append(
            OrbaxCheckpoint(
                logdir,
                monitor="loss_val",
                mode="min",
            ),
        )
    callbacks.append(ciclo.keras_bar(total=num_total_train_steps))

    state, history, _ = ciclo.train_loop(
        state,
        train_ds.repeat(
            num_epochs
        ).as_numpy_iterator(),  # repeat the training dataset for num_epochs
        {
            ciclo.on_train_step: [
                partial(
                    train_step,
                    task_callables=task_callables,
                    learning_rate_fn=lr_fn,
                )
            ],
            ciclo.on_test_step: [
                partial(
                    eval_step,
                    task_callables=task_callables,
                )
            ],
            ciclo.on_reset_step: [reset_step],
        },
        callbacks=callbacks,
        test_dataset=lambda: val_ds.as_numpy_iterator(),
        epoch_duration=steps_per_epoch,
        test_duration=num_val_steps,
        test_name="val",
        stop=num_total_train_steps,
    )

    return state, history


def run_eval(
    eval_ds: tf.data.Dataset,
    state: TrainState,
    task_callables: TaskCallables,
) -> ciclo.History:
    """
    Run the test loop.
    Args:
        eval_ds: Evaluation dataset as tf.data.Dataset object.
        state: training state of the neural network
        task_callables: struct containing the functions for the learning task
    Returns:
        history: History object containing the test metrics.
    """
    kbar = ciclo.keras_bar(total=len(eval_ds))
    setattr(
        kbar,
        ciclo.on_test_step,
        lambda state, batch, elapsed, loop_state: kbar.__loop_callback__(loop_state),
    )
    _, history, _ = ciclo.test_loop(
        state,
        eval_ds.as_numpy_iterator(),
        tasks={
            ciclo.on_test_step: [
                partial(
                    eval_step,
                    task_callables=task_callables,
                )
            ],
        },
        callbacks=[kbar],
        stop=len(eval_ds),
    )

    return history
