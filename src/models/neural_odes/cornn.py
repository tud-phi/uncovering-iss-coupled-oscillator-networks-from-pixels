from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .neural_ode_base import NeuralOdeBase


class CornnOde(NeuralOdeBase):
    """
    Inspired by the coupled oscillator ODE from the CORNN paper.
    @inproceedings{rusch2021coupled,
      title={Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies},
      author={Rusch, T. Konstantin and Mishra, Siddhartha},
      booktitle={International Conference on Learning Representations},
      year={2021}
    }
    https://github.com/tk-rusch/coRNN
    """

    latent_dim: int
    input_dim: int
    gamma: float = 1.0
    epsilon: float = 1.0
    nonlinearity: Callable = nn.tanh

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        # the latent variables are given in the input
        z = x[..., : self.latent_dim]
        # the velocity of the latent variables is given in the input
        z_d = x[..., self.latent_dim :]

        # concatenate the state and the control input
        input = jnp.concatenate([x, tau], axis=-1)

        # compute the acceleration of the latent variables
        z_dd = (
            self.nonlinearity(nn.Dense(features=self.latent_dim)(input))
            - self.gamma * z
            - self.epsilon * z_d
        )

        # concatenate the velocity and acceleration of the latent variables
        x_d = jnp.concatenate([z_d, z_dd], axis=-1)

        return x_d
