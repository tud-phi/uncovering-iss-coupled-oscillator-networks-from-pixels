from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .neural_ode_base import NeuralOdeBase


class LinearStateSpaceOde(NeuralOdeBase):
    """An ODE based on a linear state space model."""

    latent_dim: int
    input_dim: int
    mechanical_system: bool = True

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        if self.mechanical_system:
            # the velocity of the latent variables is given in the input
            z_d = x[..., self.latent_dim:]
            # compute z_dd = A @ x + B @ tau where A and B are learned matrices
            z_dd = (
                    nn.Dense(features=self.latent_dim, use_bias=False)(x)
                    + nn.Dense(features=self.latent_dim, use_bias=False)(tau)
            )
            # concatenate the velocity and acceleration of the latent variables
            x_d = jnp.concatenate([z_d, z_dd], axis=-1)
        else:
            # compute x_d = A @ x + B @ tau where A and B are learned matrices
            x_d = (
                    nn.Dense(features=2*self.latent_dim, use_bias=False)(x)
                    + nn.Dense(features=2*self.latent_dim, use_bias=False)(tau)
            )

        return x_d
