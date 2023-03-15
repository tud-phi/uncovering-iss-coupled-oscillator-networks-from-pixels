from flax import linen as nn  # Linen API
from jax import Array, random
import jax.numpy as jnp
import optax
from typing import Callable, Type

from src.structs import TrainState


def initialize_train_state(
    rng: random.KeyArray,
    nn_model: nn.Module,
    nn_dummy_input: Array,
    learning_rate_fn: Callable,
    weight_decay: float = 0.0,
) -> TrainState:
    """
    Initialize the train state for the neural network.
    Args:
        rng: PRNG key for pseudo-random number generation.
        nn_model: Neural network object.
        nn_dummy_input: Dummy input to initialize the neural network parameters.
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
        apply_fn=nn_model.apply,
        params=nn_params,
        tx=tx,
    )

    return state
