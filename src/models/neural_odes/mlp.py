from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .neural_ode_base import NeuralOdeBase


class MlpOde(NeuralOdeBase):
    """A simple MLP ODE."""

    latent_dim: int
    input_dim: int
    num_layers: int = 5
    hidden_dim: int = 20
    nonlinearity: Callable = nn.sigmoid
    mechanical_system: bool = True

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        # concatenate the state and the control input
        tmp = jnp.concatenate([x, tau], axis=-1)

        # pass through MLP
        for _ in range(self.num_layers - 1):
            tmp = nn.Dense(features=self.hidden_dim)(tmp)
            tmp = self.nonlinearity(tmp)

        if self.mechanical_system:
            # the velocity of the latent variables is given in the input
            z_d = x[..., self.latent_dim :]
            # the output of the MLP is interpreted as the acceleration of the latent variables
            z_dd = nn.Dense(features=self.latent_dim)(tmp)
            # concatenate the velocity and acceleration of the latent variables
            x_d = jnp.concatenate([z_d, z_dd], axis=-1)
        else:
            # state dimension is 2 * latent_dim
            x_d = nn.Dense(features=2 * self.latent_dim)(tmp)

        return x_d
