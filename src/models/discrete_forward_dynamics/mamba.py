__all__ = ["DiscreteMambaDynamics"]
from flax import linen as nn  # Linen API
from hippox.main import Hippo
from jax import Array, debug
import jax.numpy as jnp
from typing import Any, Callable
from warnings import warn

from .discrete_forward_dynamics_base import DiscreteForwardDynamicsBase
from .utils import discretize_state_space_model


class DiscreteMambaDynamics(DiscreteForwardDynamicsBase):
    """
    A mamba state space layer.
    Important note: We keep a constant time step dt for the discretization instead of learning it input-dependently.
    Also, we don't do anything with the C matrix. We consider the following discrete-time state space model:
        x_next = Ad @ x + Bd(tau) @ tau
    where Ad is initialized as a Hippo matrix and Bd is a learned matrix.
    """

    state_dim: int
    input_dim: int
    output_dim: int
    dt: float

    discretization_method: str = "zoh"  # in ["zoh", "bilinear"]
    # HiPPO parameters
    # in ["legs", "legt", "lmu", "lagt", "glagt", "fout", "foud", "fourier_decay", "fourier_double", "linear", "inverse"]
    hippo_measure: str = "legs"

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        """
        Args:
            x: state of shape (state_dim, )
            tau: control input of shape (input_dim, )
        Returns:
            x_next: state of shape (output_dim, )
        """
        assert (
            self.output_dim <= self.state_dim
        ), "Output dim must be less than or equal to state dim"

        hippo = Hippo(
            state_size=self.state_dim,
            basis_measure=self.hippo_measure,
            dplr=True,
            diagonalize=False,
        )
        hippo()

        A = self.param("lambda", hippo.lambda_initializer("full"), (self.state_dim,))
        B_flat = nn.Dense(features=self.state_dim * self.input_dim)(tau)
        B = jnp.reshape(B_flat, (self.state_dim, self.input_dim))

        # compute x_d = Ad @ x + Bd @ tau where Ad and Bd are time-discrete matrices
        Ad, Bd = discretize_state_space_model(
            A, B, self.dt, method=self.discretization_method
        )
        x_next = Ad @ x + Bd @ tau

        # the state dim might be larger than the output dim
        # in which case we need to slice the output
        x_next = x_next[..., -self.output_dim :]

        return x_next
