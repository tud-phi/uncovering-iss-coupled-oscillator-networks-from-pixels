from flax import linen as nn  # Linen API
from jax import Array, debug
import jax.numpy as jnp
from typing import Any, Callable

from .neural_ode_base import NeuralOdeBase
from .utils import generate_positive_definite_matrix_from_params


default_kernel_init = nn.initializers.lecun_normal()


class ConOde(NeuralOdeBase):
    """
    Coupled oscillator ODE with trainable parameters.
    Formulated in such way that we can prove input-to-state stability.
    """

    latent_dim: int
    input_dim: int

    nonlinearity: Callable = nn.tanh

    param_dtype: Any = jnp.float32
    bias_init: Callable = nn.initializers.zeros

    use_w_coordinates: bool = False
    diag_shift: float = 1e-6
    diag_eps: float = 2e-6

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
            "l_Lambda_w", tri_params_init, (num_Lambda_w_params,), self.param_dtype
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
        e_w = self.param("l_E_w", tri_params_init, (num_E_w_params,), self.param_dtype)
        E_w = generate_positive_definite_matrix_from_params(
            self.latent_dim, e_w, diag_shift=self.diag_shift, diag_eps=self.diag_eps
        )

        # bias term
        bias = self.param("bias", self.bias_init, (self.latent_dim,), self.param_dtype)

        # number of params in B_w / B_w_inv matrix
        num_B_w_params = int((self.latent_dim**2 + self.latent_dim) / 2)

        if self.use_w_coordinates:
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
            z_w = x[..., : self.latent_dim]
            # the velocity of the latent variables is given in the input
            z_d_w = x[..., self.latent_dim :]

            z_dd_w = B_w_inv @ (
                self.nonlinearity(
                    nn.Dense(features=self.latent_dim, use_bias=False)(tau)
                )
                - Lambda_w @ z_w
                - E_w @ z_d_w
                - self.nonlinearity(z_w + bias)
            )

            # concatenate the velocity and acceleration of the latent variables
            x_d = jnp.concatenate([z_d_w, z_dd_w], axis=-1)
        else:
            # constructing Bw as a positive definite matrix
            # vector of parameters for triangular matrix
            b_w = self.param(
                "b_w", tri_params_init, (num_B_w_params,), self.param_dtype
            )
            B_w = generate_positive_definite_matrix_from_params(
                self.latent_dim, b_w, diag_shift=self.diag_shift, diag_eps=self.diag_eps
            )
            """
            # print minimum eigenvalue of B_w
            debug.print(
                "min Eigenvalue of B_w: {min_eig}", min_eig=jnp.min(jnp.linalg.eigh(B_w)[0])
            )
            """

            # compute everything in the orginal coordinates
            W = jnp.linalg.inv(B_w)
            Lambda = Lambda_w @ B_w
            E = E_w @ B_w

            # the latent variables are given in the input
            z = x[..., : self.latent_dim]
            # the velocity of the latent variables is given in the input
            z_d = x[..., self.latent_dim :]

            z_dd = (
                self.nonlinearity(
                    nn.Dense(features=self.latent_dim, use_bias=False)(tau)
                )
                - Lambda @ z
                - E @ z_d
                - self.nonlinearity(W @ z + bias)
            )

            # concatenate the velocity and acceleration of the latent variables
            x_d = jnp.concatenate([z_d, z_dd], axis=-1)

        return x_d
