from flax import linen as nn  # Linen API
from hippox.main import Hippo
from jax import Array
import jax.numpy as jnp
from typing import Callable
from warnings import warn

from .neural_ode_base import NeuralOdeBase


class MambaOde(NeuralOdeBase):
    """
    An ODE based on a mamba state space model.
        x_d = Ad @ x + Bd(tau) @ tau
    where Ad is initialized as a Hippo matrix and Bd is a learned matrix.
    """

    latent_dim: int
    input_dim: int

    # HiPPO parameters
    # in ["legs", "legt", "lmu", "lagt", "glagt", "fout", "foud", "fourier_decay", "fourier_double", "linear", "inverse"]
    hippo_measure: str = "legs"

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        hippo = Hippo(
            state_size=2 * self.latent_dim,
            basis_measure=self.hippo_measure,
            dplr=True,
            diagonalize=False,
        )
        hippo()

        A = self.param(
            "lambda", hippo.lambda_initializer("full"), (2 * self.latent_dim,)
        )
        B_flat = nn.Dense(features=2 * self.latent_dim * self.input_dim)(tau)
        B = jnp.reshape(B_flat, (2 * self.latent_dim, self.input_dim))

        # compute x_d = A @ x + B(tau) @ tau
        x_d = A @ x + B @ tau

        return x_d
