from flax import linen as nn  # Linen API
from hippox.main import Hippo
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .neural_ode_base import NeuralOdeBase


class LinearStateSpaceOde(NeuralOdeBase):
    """
    An ODE based on a linear state space model.
        x_d = A @ x + B @ tau
    In modern literature, this is also known as SSM or S4.
        https://arxiv.org/abs/2111.00396
        https://github.com/state-spaces/s4
    """

    latent_dim: int
    input_dim: int

    transition_matrix_init: str = "general"  # in ["general", "mechanical", "hippo"]
    # HiPPO parameters
    # in ["legs", "legt", "lmu", "lagt", "glagt", "fout", "foud", "fourier_decay", "fourier_double", "linear", "inverse"]
    hippo_measure: str = "legs"

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        if self.transition_matrix_init == "mechanical":
            # the velocity of the latent variables is given in the input
            z_d = x[..., self.latent_dim :]
            # compute z_dd = A @ x + B @ tau where A and B are learned matrices
            z_dd = nn.Dense(features=self.latent_dim, use_bias=False)(x) + nn.Dense(
                features=self.latent_dim, use_bias=False
            )(tau)
            # concatenate the velocity and acceleration of the latent variables
            x_d = jnp.concatenate([z_d, z_dd], axis=-1)
        elif self.transition_matrix_init == "hippo":
            hippo_params = Hippo(
                state_size=2 * self.latent_dim,
                basis_measure=self.hippo_measure,
                diagonalize=False,
            )()
            A = hippo_params.state_matrix
            # Structure State Space Models usually assume a 1D input
            # but we have self.input_dim inputs. We can repeat the input matrix B self.input_dim times
            # to make it compatible with the input
            B = jnp.repeat(hippo_params.input_matrix[:, None], self.input_dim, axis=1)
            # compute x_d = A @ x + B @ tau
            x_d = A @ x + B @ tau
        else:
            # compute x_d = A @ x + B @ tau where A and B are learned matrices
            x_d = nn.Dense(features=2 * self.latent_dim, use_bias=False)(x) + nn.Dense(
                features=2 * self.latent_dim, use_bias=False
            )(tau)

        return x_d
