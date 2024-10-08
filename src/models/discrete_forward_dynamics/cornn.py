__all__ = ["DiscreteCornn"]
from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .discrete_forward_dynamics_base import DiscreteForwardDynamicsBase


class DiscreteCornn(DiscreteForwardDynamicsBase):
    """
    Implements the Coupled Oscillatory Recurrent Neural Network (coRNN) as a discrete-time ODE.
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
    dt: float
    dynamics_order: int = 2

    gamma: float = 1.0
    epsilon: float = 1.0
    nonlinearity: Callable = nn.tanh

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        """
        Args:
            x: state of shape (2 * latent_dim, ). In RNN literature, this is often called the "hidden state" or "carry" h.
            tau: control input of shape (input_dim, ). In RNN literature, this is often called the "input x".
        Returns:
            x_next: state of shape (2 * latent_dim, ). In RNN literature, this is often called the next "hidden state" or "carry" h.
        """
        # concatenate the state and the control input
        input = jnp.concatenate([x, tau], axis=-1)

        match self.dynamics_order:
            case 1:
                x_d = (
                    self.nonlinearity(nn.Dense(features=self.latent_dim)(input))
                    - self.gamma * x
                )
                x_next = x + self.dt * x_d
            case 2:
                # the latent variables are given in the input
                z = x[..., : self.latent_dim]
                # the velocity of the latent variables is given in the input
                z_d = x[..., self.latent_dim :]

                # compute the acceleration of the latent variables
                z_dd = (
                    self.nonlinearity(nn.Dense(features=self.latent_dim)(input))
                    - self.gamma * z
                    - self.epsilon * z_d
                )

                # integrate using the Euler method
                x_next = x + self.dt * jnp.concatenate([z_d, z_dd], axis=-1)
            case _:
                raise ValueError(f"Invalid dynamics order {self.dynamics_order}")

        return x_next
