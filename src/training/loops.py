import ciclo
from ciclo import Elapsed, History
from clu import metrics as clu_metrics
from flax import linen as nn  # Linen API
from functools import partial
import jax
from jax import Array, debug, random
import jax.numpy as jnp
import optax
from pathlib import Path
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from src.structs import TaskCallables, TrainState
from src.training.checkpointing import OrbaxCheckpointerCallback
from src.training.train_state_utils import (
    initialize_train_state,
    print_number_of_trainable_params,
)
from src.training.optim import create_learning_rate_fn


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
        logs: ciclo logs
        state: updated training state of the neural network
    """
    # split the PRNG key
    rng, rng_loss_fn = random.split(state.rng)

    loss_fn = partial(task_callables.loss_fn, batch, rng=rng_loss_fn, training=True)
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, preds), grad_nn_params = grad_fn(state.params)

    # optimize the neural network parameters with gradient descent
    state = state.apply_gradients(grads=grad_nn_params)

    # extract the learning rate for the current step
    lr = learning_rate_fn(state.step)

    # update the PRNG key in the training state
    state = state.replace(rng=rng)

    # compute metrics
    batch_metrics_dict = task_callables.compute_metrics_fn(batch, preds)
    batch_metrics = state.metrics.single_from_model_output(
        loss=loss, lr=lr, **batch_metrics_dict
    )
    merged_metrics = state.metrics.merge(batch_metrics)
    # update metrics in the training state
    state = state.replace(metrics=merged_metrics)

    # save metrics to logs
    logs = ciclo.logs()
    logs.add_stateful_metrics(**merged_metrics.compute())

    return logs, state


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
        logs: ciclo logs
        state: updated training state of the neural network
    """
    # split the PRNG key
    rng, rng_loss_fn = random.split(state.rng)

    loss, preds = task_callables.loss_fn(
        batch, state.params, rng=rng_loss_fn, training=False
    )

    # update the PRNG key in the training state
    state = state.replace(rng=rng)

    # compute metrics
    batch_metrics_dict = task_callables.compute_metrics_fn(batch, preds)
    batch_metrics = state.metrics.single_from_model_output(
        loss=loss, lr=jnp.zeros(1), **batch_metrics_dict
    )
    merged_metrics = state.metrics.merge(batch_metrics)
    # update metrics in the training state
    state = state.replace(metrics=merged_metrics)

    # save metrics to logs
    logs = ciclo.logs()
    computed_metrics = merged_metrics.compute()
    # delete the learning rate from the metrics
    del computed_metrics["lr"]
    logs.add_stateful_metrics(**computed_metrics)

    return logs, state


def reset_step(state: TrainState):
    """
    Reset the metrics of the training state.
    """
    return state.replace(metrics=state.metrics.empty())


def run_training(
    rng: random.KeyArray,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    task_callables: TaskCallables,
    metrics_collection_cls: Type[clu_metrics.Collection],
    num_epochs: int,
    jit: bool = True,
    state: Optional[TrainState] = None,
    nn_model: Optional[nn.Module] = None,
    init_fn: Optional[Callable] = None,
    init_kwargs: Dict[str, Any] = None,
    tx: optax.GradientTransformation = None,
    learning_rate_fn: Optional[Callable] = None,
    base_lr: Optional[float] = None,
    warmup_epochs: int = 0,
    cosine_decay_epochs: int = None,
    b1: float = 0.9,
    b2: float = 0.999,
    weight_decay: float = 0.0,
    callbacks: Optional[List[Any]] = None,
    logdir: Path = None,
    show_pbar: bool = True,
) -> Tuple[TrainState, History, Elapsed]:
    """
    Run the training loop.
    Args:
        rng: PRNG key for pseudo-random number generation.
        train_ds: Training dataset as tf.data.Dataset object.
        val_ds: Validation dataset as tf.data.Dataset object.
        task_callables: struct containing the functions for the learning task
        metrics_collection_cls: Metrics collection class for respective task.
        num_epochs: Number of epochs to train for.
        jit: Whether to jit-compile the training and evaluation step.
        state: TrainState object. If provided, the training will continue from this state.
        nn_model: Neural network model. Only needed if no TrainState is provided.
        init_fn: Method of the neural network to call for initializing neural network parameters.
        init_kwargs: Keyword arguments for the `init_fn` of the neural network.
        tx: optimizer. Either an optimizer needs to be provided or a learning rate function.
        learning_rate_fn: A function that takes the current step and returns the current learning rate.
        base_lr: Base learning rate (after warmup and before decay).
        warmup_epochs: Number of epochs for warmup.
        b1: Exponential decay rate for the first moment estimates of the Adam optimizer.
        b2: Exponential decay rate for the second moment estimates of the Adam optimizer.
        weight_decay: Weight decay.
        cosine_decay_epochs: Number of epochs for cosine decay. If None, will use num_epochs - warmup_epochs.
        callbacks: List of callbacks at each iteration of the loop.
        logdir: Path to the directory where the training logs should be saved.
        show_pbar: Whether to use a progress bar.
    Returns:
        state: final TrainState object
        history: History object containing the training metrics.
        elapsed: Elapsed number of steps.
    """
    steps_per_epoch = len(train_ds)
    num_total_train_steps = num_epochs * steps_per_epoch
    num_val_steps = len(val_ds)

    # initialize the learning rate scheduler
    if learning_rate_fn is None:
        learning_rate_fn = create_learning_rate_fn(
            num_epochs,
            steps_per_epoch=steps_per_epoch,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            cosine_decay_epochs=cosine_decay_epochs,
        )

    # if no state is provided, initialize the train state
    if state is None:
        assert (
            nn_model is not None
        ), "If no state is provided, a neural network model must be provided."

        # extract dummy batch from dataset
        nn_dummy_batch = next(train_ds.as_numpy_iterator())
        # assemble input for dummy batch
        nn_dummy_input = task_callables.assemble_input_fn(nn_dummy_batch)

        # initialize the train state
        state = initialize_train_state(
            rng,
            nn_model,
            nn_dummy_input=nn_dummy_input,
            metrics_collection_cls=metrics_collection_cls,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
            tx=tx,
            learning_rate_fn=learning_rate_fn,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
        )
    else:
        state = state.replace(metrics=metrics_collection_cls.empty())

        if tx is None:
            # initialize the Adam with weight decay optimizer for both neural networks
            tx = optax.adamw(learning_rate_fn, b1=b1, b2=b2, weight_decay=weight_decay)
        state = state.replace(tx=tx, opt_state=tx.init(state.params))

    # print number of trainable parameters
    print_number_of_trainable_params(state)

    if callbacks is None:
        callbacks = []

    orbax_checkpointer_callback = None
    if logdir is not None:
        orbax_checkpointer_callback = OrbaxCheckpointerCallback(
            logdir.resolve(),
            max_to_keep=1,
            monitor="loss_val",
            mode="min",
        )
        callbacks.append(orbax_checkpointer_callback)
    if show_pbar:
        callbacks.append(ciclo.keras_bar(total=num_total_train_steps))

    train_step_fn = partial(
        train_step,
        task_callables=task_callables,
        learning_rate_fn=learning_rate_fn,
    )
    eval_step_fn = partial(
        eval_step,
        task_callables=task_callables,
    )
    reset_step_fn = reset_step
    if jit is True:
        train_step_fn = jax.jit(train_step_fn)
        eval_step_fn = jax.jit(eval_step_fn)
        reset_step_fn = jax.jit(reset_step_fn)

    state, history, elapsed = ciclo.train_loop(
        state,
        train_ds.repeat(
            num_epochs
        ).as_numpy_iterator(),  # repeat the training dataset for num_epochs
        {
            ciclo.on_train_step: [train_step_fn],
            ciclo.on_test_step: [eval_step_fn],
            ciclo.on_reset_step: [reset_step_fn],
        },
        callbacks=callbacks,
        test_dataset=lambda: val_ds.as_numpy_iterator(),
        epoch_duration=steps_per_epoch,
        test_duration=num_val_steps,
        test_name="val",
        stop=num_total_train_steps,
    )

    if orbax_checkpointer_callback is not None:
        orbax_checkpointer_callback.wait_until_finished()

    return state, history, elapsed


def run_eval(
    eval_ds: tf.data.Dataset,
    state: TrainState,
    task_callables: TaskCallables,
    jit: bool = True,
    show_pbar: bool = True,
) -> Tuple[TrainState, ciclo.History]:
    """
    Run the test loop.
    Args:
        eval_ds: Evaluation dataset as tf.data.Dataset object.
        state: training state of the neural network
        task_callables: struct containing the functions for the learning task
        jit: Whether to jit-compile the training and evaluation step.
        show_pbar: Whether to use a progress bar.
    Returns:
        state: final TrainState object
        history: History object containing the test metrics.
    """
    callbacks = []
    if show_pbar:
        kbar = ciclo.keras_bar(total=len(eval_ds))
        setattr(
            kbar,
            ciclo.on_test_step,
            lambda state, batch, elapsed, loop_state: kbar.__loop_callback__(
                loop_state
            ),
        )
        callbacks.append(kbar)

    test_step_fn = partial(
        eval_step,
        task_callables=task_callables,
    )
    if jit:
        test_step_fn = jax.jit(test_step_fn)

    state, history, _ = ciclo.test_loop(
        state,
        eval_ds.as_numpy_iterator(),
        tasks={
            ciclo.on_test_step: [test_step_fn],
        },
        callbacks=callbacks,
        stop=len(eval_ds),
    )

    return state, history
