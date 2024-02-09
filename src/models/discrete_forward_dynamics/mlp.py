from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .discrete_forward_dynamics_base import DiscreteForwardDynamicsBase


class DiscreteMlpDynamics(DiscreteForwardDynamicsBase):
    """A simple MLP ODE."""

    state_dim: int
    input_dim: int
    output_dim: int
    dt: float

    num_layers: int = 5
    hidden_dim: int = 20
    nonlinearity: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        """
        Args:
            x: state of shape (state_dim, )
            tau: control input of shape (input_dim, )
        Returns:
            x_next: state of shape (output_dim, )
        """
        # concatenate the state and the control input
        tmp = jnp.concatenate([x, tau], axis=-1)

        # pass through MLP
        for _ in range(self.num_layers - 1):
            tmp = nn.Dense(features=self.hidden_dim)(tmp)
            tmp = self.nonlinearity(tmp)

        # return the next latent state
        x_next = x[-1] + self.dt * nn.Dense(features=self.output_dim)(tmp)

        return x_next
