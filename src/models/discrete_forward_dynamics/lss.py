from flax import linen as nn  # Linen API
from hippox.main import Hippo
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .discrete_forward_dynamics_base import DiscreteForwardDynamicsBase
from .utils import discretize_state_space_model


class DiscreteLssDynamics(DiscreteForwardDynamicsBase):
    """
    A simple linear state space model.
    """

    state_dim: int
    input_dim: int
    output_dim: int
    dt: float

    transition_matrix_init: str = "general"  # in ["general", "hippo"]
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

        if self.transition_matrix_init == "hippo":
            hippo_params = Hippo(
                state_size=self.state_dim,
                basis_measure=self.hippo_measure,
                diagonalize=False,
            )()
            # TODO: make A and B learnable
            A = hippo_params.state_matrix
            # Structure State Space Models usually assume a 1D input
            # but we have self.input_dim inputs. We can repeat the input matrix B self.input_dim times
            # to make it compatible with the input
            B = jnp.repeat(hippo_params.input_matrix[:, None], self.input_dim, axis=1)
            # compute x_d = Ad @ x + Bd @ tau where Ad and Bd are time-discrete matrices
            Ad, Bd = discretize_state_space_model(
                A, B, self.dt, method=self.discretization_method
            )
            x_next = Ad @ x + Bd @ tau
        else:
            # compute x_d = Ad @ x + Bd @ tau where Ad and Bd are learned, time-discrete matrices
            x_next = nn.Dense(features=self.state_dim, use_bias=False)(x) + nn.Dense(
                features=self.state_dim, use_bias=False
            )(tau)

        # the state dim might be larger than the output dim
        # in which case we need to slice the output
        x_next = x_next[..., -self.output_dim :]

        return x_next
