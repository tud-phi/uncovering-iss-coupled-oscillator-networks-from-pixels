from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .discrete_forward_dynamics_base import DiscreteForwardDynamicsBase


class DiscreteMlpDynamics(DiscreteForwardDynamicsBase):
    """A simple MLP ODE."""

    latent_dim: int
    input_dim: int
    output_dim: int

    dt: float
    num_past_timesteps: int = 2

    num_layers: int = 5
    hidden_dim: int = 20
    nonlinearity: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, z_ts: Array, tau_ts: Array) -> Array:
        """
        Args:
            z_ts: latent state of shape (num_past_timesteps, latent_dim)
            tau_ts: control input of shape (num_past_timesteps, input_dim)
        Returns:
            z_next: latent state of shape (latent_dim, )
        """
        # concatenate the state and the control input
        tmp = jnp.concatenate([z_ts, tau_ts], axis=-1).reshape((-1,))

        # pass through MLP
        for _ in range(self.num_layers - 1):
            tmp = nn.Dense(features=self.hidden_dim)(tmp)
            tmp = self.nonlinearity(tmp)

        # return the next latent state
        z_next = z_ts[-1] + self.dt * nn.Dense(features=self.latent_dim)(tmp)

        return z_next
