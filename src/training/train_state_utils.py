from flax import linen as nn  # Linen API
from flax.training import orbax_utils
from jax import Array, random
import jax.numpy as jnp
import jax_metrics as jm
from orbax.checkpoint import Checkpointer, CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
import optax
import os
from typing import Callable, Type, Union

from src.structs import TrainState


def initialize_train_state(
    rng: random.KeyArray,
    nn_model: nn.Module,
    nn_dummy_input: Array,
    metrics: jm.Metrics,
    learning_rate_fn: Union[float, Callable],
    weight_decay: float = 0.0,
) -> TrainState:
    """
    Initialize the train state for the neural network.
    Args:
        rng: PRNG key for pseudo-random number generation.
        nn_model: Neural network object.
        nn_dummy_input: Dummy input to initialize the neural network parameters.
        metrics: Metrics object for respective task.
        learning_rate_fn: A function that takes the current step and returns the current learning rate.
            It has the signature learning_rate_fn(step: int) -> lr.
        weight_decay: Weight decay of the Adam optimizer for training the neural networks.
    Returns:
        state: TrainState object for the neural network.
    """
    # initialize parameters of the neural networks by passing a dummy input through the network
    # Hint: pass the `rng` and a dummy input to the `init` method of the neural network object
    nn_params = nn_model.init(rng, nn_dummy_input)["params"]

    # initialize the Adam with weight decay optimizer for both neural networks
    tx = optax.adamw(learning_rate_fn, weight_decay=weight_decay)

    # create the TrainState object for both neural networks
    state = TrainState.create(
        apply_fn=nn_model.apply, params=nn_params, tx=tx, metrics=metrics
    )

    return state


def restore_train_state(
    ckpt_dir: os.PathLike,
    nn_model: nn.Module,
    metrics: jm.Metrics,
    step: int = None,
    learning_rate_fn: Union[float, Callable] = 0.0,
    weight_decay: float = 0.0,
) -> TrainState:
    ckptr = Checkpointer(PyTreeCheckpointer())
    ckpt_mgr = CheckpointManager(
        ckpt_dir,
        ckptr,
    )
    if step is None:
        step = ckpt_mgr.latest_step()

    # restore_args = orbax_utils.restore_args_from_target(nn_model, mesh=None)
    # nn_model = ckpt_mgr.restore(step, items=nn_model, restore_kwargs={'restore_args': restore_args})
    nn_params = ckpt_mgr.restore(step)["params"]

    # initialize the Adam with weight decay optimizer for both neural networks
    tx = optax.adamw(learning_rate_fn, weight_decay=weight_decay)

    # create the TrainState object for both neural networks
    state = TrainState.create(
        apply_fn=nn_model.apply, params=nn_params, tx=tx, metrics=metrics
    )

    return state