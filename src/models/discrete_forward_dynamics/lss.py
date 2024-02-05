from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .discrete_forward_dynamics_base import DiscreteForwardDynamicsBase


class DiscreteLssDynamics(DiscreteForwardDynamicsBase):
    """
    A simple MLP ODE.
    """

    latent_dim: int
    input_dim: int
    output_dim: int

    dt: float
    num_past_timesteps: int = 2
    # if input_displacements is True, input relative differences between latents instead of absolute values
    # J. Martinez, M. J. Black, and J. Romero, “On human motion prediction using recurrent neural networks,”
    # in Proc. IEEE Conf. on Comput. Vis. Pattern Recognit., 2017.
    input_displacements: bool = False

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
        if self.input_displacements:
            tmp = jnp.concatenate([z_ts[0:1], jnp.diff(z_ts, axis=0)], axis=0)
        else:
            tmp = z_ts
        tmp = jnp.concatenate([tmp, tau_ts], axis=-1).reshape((-1,))

        # pass through MLP
        for _ in range(self.num_layers - 1):
            tmp = nn.Dense(features=self.hidden_dim)(tmp)
            tmp = self.nonlinearity(tmp)

        # return the next latent state
        z_next = z_ts[-1] + self.dt * nn.Dense(features=self.latent_dim)(tmp)

        return z_next
