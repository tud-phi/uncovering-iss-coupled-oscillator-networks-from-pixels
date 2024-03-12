from flax import linen as nn  # Linen API
from jax import Array, debug
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .neural_ode_base import NeuralOdeBase
from .utils import generate_positive_definite_matrix_from_params


default_kernel_init = nn.initializers.lecun_normal()


class ConIaeOde(NeuralOdeBase):
    """
    Coupled oscillator ODE with trainable parameters with autoencoding on the input.
    """

    latent_dim: int
    input_dim: int

    num_layers: int = 5
    hidden_dim: int = 32
    potential_nonlinearity: Callable = nn.tanh
    input_nonlinearity: Optional[Callable] = nn.tanh

    param_dtype: Any = jnp.float32
    bias_init: Callable = nn.initializers.zeros

    diag_shift: float = 1e-6
    diag_eps: float = 2e-6

    # control settings
    apply_feedforward_term: bool = True
    apply_feedback_term: bool = True

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        """
        Args:
            x: latent state of shape (2* latent_dim, )
            tau: control input of shape (input_dim, )
        Returns:
            x_d: latent state derivative of shape (2* latent_dim, )
        """
        # initializer for triangular matrix parameters
        # this is "fan-out" mode for lecun_normal
        # TODO: make this standard deviation tunable for each matrix separately?
        tri_params_init = nn.initializers.normal(stddev=jnp.sqrt(1.0 / self.latent_dim))

        # constructing Lambda_w as a positive definite matrix
        num_Lambda_w_params = int((self.latent_dim**2 + self.latent_dim) / 2)
        # vector of parameters for triangular matrix
        lambda_w = self.param(
            "lambda_w", tri_params_init, (num_Lambda_w_params,), self.param_dtype
        )
        Lambda_w = generate_positive_definite_matrix_from_params(
            self.latent_dim,
            lambda_w,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )

        # constructing E_w as a positive definite matrix
        num_E_w_params = int((self.latent_dim**2 + self.latent_dim) / 2)
        # vector of parameters for triangular matrix
        e_w = self.param("e_w", tri_params_init, (num_E_w_params,), self.param_dtype)
        E_w = generate_positive_definite_matrix_from_params(
            self.latent_dim, e_w, diag_shift=self.diag_shift, diag_eps=self.diag_eps
        )

        # bias term
        bias = self.param("bias", self.bias_init, (self.latent_dim,), self.param_dtype)

        # number of params in B_w / B_w_inv matrix
        num_B_w_params = int((self.latent_dim**2 + self.latent_dim) / 2)

        # input-state coupling matrix
        tmp = tau
        for _ in range(self.num_layers - 1):
            tmp = nn.Dense(features=self.hidden_dim)(tmp)
            tmp = self.input_nonlinearity(tmp)
        V = nn.Dense(features=(self.latent_dim * self.input_dim))(tmp).reshape(self.latent_dim, self.input_dim)
        # compute the latent-space input
        u = V @ tau

        # constructing Bw_inv as a positive definite matrix
        b_w_inv = self.param(
            "b_w_inv", tri_params_init, (num_B_w_params,), self.param_dtype
        )
        B_w_inv = generate_positive_definite_matrix_from_params(
            self.latent_dim,
            b_w_inv,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )

        # the latent variables are given in the input
        zw = x[..., : self.latent_dim]
        # the velocity of the latent variables is given in the input
        zw_d = x[..., self.latent_dim :]

        zw_dd = B_w_inv @ (
            u
            - Lambda_w @ zw
            - E_w @ zw_d
            - self.potential_nonlinearity(zw + bias)
        )

        # concatenate the velocity and acceleration of the latent variables
        x_d = jnp.concatenate([zw_d, zw_dd], axis=-1)

        return x_d
