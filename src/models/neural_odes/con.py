from flax import linen as nn  # Linen API
from jax import Array, debug
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple, Union

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

    potential_nonlinearity: Callable = nn.tanh
    input_nonlinearity: Optional[Callable] = None

    param_dtype: Any = jnp.float32
    bias_init: Callable = nn.initializers.zeros

    use_w_coordinates: bool = False
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
        V = self.param(
            "V",
            default_kernel_init,
            (self.latent_dim, self.input_dim),
            self.param_dtype,
        )
        # compute the input torque in the latent space
        if self.input_nonlinearity is None:
            tau_input = V @ tau
        else:
            tau_input = self.input_nonlinearity(V @ tau)

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
            zw = x[..., : self.latent_dim]
            # the velocity of the latent variables is given in the input
            zw_d = x[..., self.latent_dim :]

            zw_dd = B_w_inv @ (
                tau_input
                - Lambda_w @ zw
                - E_w @ zw_d
                - self.potential_nonlinearity(zw + bias)
            )

            # concatenate the velocity and acceleration of the latent variables
            x_d = jnp.concatenate([zw_d, zw_dd], axis=-1)
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
                tau_input
                - Lambda @ z
                - E @ z_d
                - self.potential_nonlinearity(W @ z + bias)
            )

            # concatenate the velocity and acceleration of the latent variables
            x_d = jnp.concatenate([z_d, z_dd], axis=-1)

        return x_d
    
    def energy_fn(self, x: Array) -> Array:
        """
        Compute the energy of the system.
        Args:
            x: latent state of shape (2* latent_dim, )
        Returns:
            V: energy of the system
        """
        # extract the matrices from the neural network
        bias = self.get_variable("params", "bias")
        lambda_w = self.get_variable("params", "lambda_w")
        Lambda_w = generate_positive_definite_matrix_from_params(
            self.latent_dim,
            lambda_w,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )

        if self.use_w_coordinates:
            zw = x[..., : self.latent_dim]
            zw_d = x[..., self.latent_dim :]

            # constructing Bw_inv as a positive definite matrix
            b_w_inv = self.get_variable("params", "b_w_inv")
            B_w_inv = generate_positive_definite_matrix_from_params(
                self.latent_dim,
                b_w_inv,
                diag_shift=self.diag_shift,
                diag_eps=self.diag_eps,
            )

            # computing B_w from B_w_inv
            B_w = jnp.linalg.inv(B_w_inv)
        else:
            z = x[..., : self.latent_dim]
            z_d = x[..., self.latent_dim :]

            # constructing Bw as a positive definite matrix
            # vector of parameters for triangular matrix
            b_w = self.get_variable("params", "b_w")
            B_w = generate_positive_definite_matrix_from_params(
                self.latent_dim, b_w, diag_shift=self.diag_shift, diag_eps=self.diag_eps
            )

            # compute W
            W = jnp.linalg.inv(B_w)

            # map the latent variables to the w-coordinates
            zw = W @ z
            zw_d = W @ z_d
        

        # compute the potential energy
        U = (
            0.5 * zw[None, :] @ Lambda_w @ zw[:, None] 
            + jnp.log(jnp.cosh(zw + bias))
        )
        # compute the kinetic energy
        T = 0.5 * zw_d[None, :] @ B_w @ zw_d[:, None]

        # compute the total energy
        V = T + U

        return V

    def setpoint_regulation_control_fn(
        self,
        x: Array,
        z_des: Array,
        kp: Union[float, Array] = 0.0,
        kd: Union[float, Array] = 0.0,
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Control function for setpoint regulation.
        Args:
            x: latent state of shape (2* latent_dim, )
            z_des: desired latent state of shape (latent_dim, )
            kp: proportional gain
            kd: derivative gain
        Returns:
            tau: control input
            control_info: dictionary with control information
        """
        # extract the matrices from the neural network
        bias = self.get_variable("params", "bias")
        lambda_w = self.get_variable("params", "lambda_w")
        Lambda_w = generate_positive_definite_matrix_from_params(
            self.latent_dim,
            lambda_w,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )

        if self.use_w_coordinates:
            zw = x[..., : self.latent_dim]
            zw_d = x[..., self.latent_dim :]

            # compute error in the latent space
            error_zw = z_des - zw
            # compute the feedback term
            tau_z_fb = kp * error_zw - kd * zw_d
            # compute the feedforward term
            tau_z_ff = Lambda_w @ z_des + jnp.tanh(z_des + bias)
        else:
            z = x[..., : self.latent_dim]
            z_d = x[..., self.latent_dim :]

            # extract the matrices from the neural network
            b_w = self.get_variable("params", "b_w")
            B_w = generate_positive_definite_matrix_from_params(
                self.latent_dim,
                b_w,
                diag_shift=self.diag_shift,
                diag_eps=self.diag_eps,
            )

            W = jnp.linalg.inv(B_w)
            Lambda = Lambda_w @ B_w

            # compute error in the latent space
            error_z = z_des - z

            # compute the feedback term
            tau_z_fb = kp * error_z - kd * z_d
            # compute the feedforward term
            tau_z_ff = Lambda @ z_des + jnp.tanh(W @ z_des + bias)

        # compute the torque in latent space
        tau_z = jnp.zeros_like(z_des)
        if self.apply_feedforward_term:
            tau_z = tau_z + tau_z_ff
        if self.apply_feedback_term:
            tau_z = tau_z + tau_z_fb

        # extract the V matrix from the neural network
        V = self.get_variable("params", "V")

        if self.input_nonlinearity is None:
            # compute the control input
            tau = jnp.linalg.pinv(V) @ tau_z
        elif type(self.input_nonlinearity) is nn.tanh:
            # clip the control input in the range [-1, 1]
            tau_z_clipped = jnp.clip(tau_z, -1.0, 1.0)
            # compute the control input
            tau = jnp.linalg.pinv(V) @ jnp.arctanh(tau_z_clipped)
        else:
            raise NotImplementedError(
                "Only the inverse of the hyperbolic tangent is implemented"
            )

        control_info = dict(
            tau=tau,
            tau_z=tau_z,
            tau_z_ff=tau_z_ff,
            tau_z_fb=tau_z_fb,
        )

        return tau, control_info
