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

    def setup(self):
        # initializer for triangular matrix parameters
        # this is "fan-out" mode for lecun_normal
        # TODO: make this standard deviation tunable for each matrix separately?
        tri_params_init = nn.initializers.normal(stddev=jnp.sqrt(1.0 / self.latent_dim))

        # constructing Lambda_w as a positive definite matrix
        num_Lambda_w_params = int((self.latent_dim ** 2 + self.latent_dim) / 2)
        # vector of parameters for triangular matrix
        lambda_w = self.param(
            "lambda_w", tri_params_init, (num_Lambda_w_params,), self.param_dtype
        )
        self.Lambda_w = generate_positive_definite_matrix_from_params(
            self.latent_dim,
            lambda_w,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )

        # constructing E_w as a positive definite matrix
        num_E_w_params = int((self.latent_dim ** 2 + self.latent_dim) / 2)
        # vector of parameters for triangular matrix
        e_w = self.param("e_w", tri_params_init, (num_E_w_params,), self.param_dtype)
        self.E_w = generate_positive_definite_matrix_from_params(
            self.latent_dim, e_w, diag_shift=self.diag_shift, diag_eps=self.diag_eps
        )

        # bias term
        self.bias = self.param("bias", self.bias_init, (self.latent_dim,), self.param_dtype)

        # number of params in B_w / B_w_inv matrix
        num_B_w_params = int((self.latent_dim ** 2 + self.latent_dim) / 2)

        # constructing Bw_inv as a positive definite matrix
        b_w_inv = self.param(
            "b_w_inv", tri_params_init, (num_B_w_params,), self.param_dtype
        )
        self.B_w_inv = generate_positive_definite_matrix_from_params(
            self.latent_dim,
            b_w_inv,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )

        V_layers = []
        for _ in range(self.num_layers - 1):
            V_layers.append(nn.Dense(features=self.hidden_dim))
            V_layers.append(self.input_nonlinearity)
        V_layers.append(nn.Dense(features=(self.latent_dim * self.input_dim)))
        self.V_nn = nn.Sequential(V_layers)

        Y_layers = []
        for _ in range(self.num_layers - 1):
            Y_layers.append(nn.Dense(features=self.hidden_dim))
            Y_layers.append(self.input_nonlinearity)
        Y_layers.append(nn.Dense(features=(self.input_dim * self.latent_dim)))
        self.Y_nn = nn.Sequential(Y_layers)

    def __call__(self, x: Array, tau: Array) -> Array:
        """
        Args:
            x: latent state of shape (2* latent_dim, )
            tau: control input of shape (input_dim, )
        Returns:
            x_d: latent state derivative of shape (2* latent_dim, )
        """
        # the latent variables are given in the input
        zw = x[..., : self.latent_dim]
        # the velocity of the latent variables is given in the input
        zw_d = x[..., self.latent_dim :]

        # compute the latent-space input
        u = self.encode_input(tau)

        zw_dd = self.B_w_inv @ (
            u
            - self.Lambda_w @ zw
            - self.E_w @ zw_d
            - self.potential_nonlinearity(zw + self.bias)
        )

        # concatenate the velocity and acceleration of the latent variables
        x_d = jnp.concatenate([zw_d, zw_dd], axis=-1)

        return x_d

    def forward_all_layers(self, x: Array, tau: Array):
        x_d = self.__call__(x, tau)
        tau_hat = self.autoencode_input(tau)
        return x_d

    def input_state_coupling(self, tau: Array) -> Array:
        V = self.V_nn(tau).reshape(self.latent_dim, self.input_dim)
        return V

    def encode_input(self, tau: Array):
        V = self.input_state_coupling(tau)
        u = V @ tau
        return u

    def decode_input(self, u: Array):
        Y = self.Y_nn(u).reshape(self.input_dim, self.latent_dim)
        tau = Y @ u
        return tau

    def autoencode_input(self, tau: Array):
        u = self.encode_input(tau)
        tau_hat = self.decode_input(u)
        return tau_hat
    
    def energy_fn(self, x: Array) -> Array:
        """
        Compute the energy of the system.
        Args:
            x: state of shape (2* latent_dim, )
        Returns:
            V: energy of the system of shape ()
        """
        zw = x[..., : self.latent_dim]
        zw_d = x[..., self.latent_dim :]

        # mass matrix
        B = jnp.linalg.inv(self.B_w_inv)

        # compute the kinetic energy
        T = (0.5 * zw_d[None, :] @ B @ zw_d[:, None]).squeeze()

        # compute the potential energy
        U = (
            0.5 * zw[None, :] @ self.Lambda_w @ zw[:, None]
            + jnp.sum(jnp.log(jnp.cosh(zw + self.bias)))
        ).squeeze()

        # compute the total energy
        V = T + U

        return V
   
    def setpoint_regulation_fn(
        self,
        x: Array,
        control_state: Dict[str, Array],
        dt: Union[float, Array],
        z_des: Array,
        kp: Union[float, Array] = 0.0,
        ki: Union[float, Array] = 0.0,
        kd: Union[float, Array] = 0.0,
        gamma: Union[float, Array] = 1.0,
    ) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
        """
        P-satI-D feedback together with potential force compensation in the latent (w) variables.
        Args:
            x: latent state of shape (2* latent_dim, )
            control_state: dictionary with the controller's stateful information. Contains entry with key "e_int" for the integral error.
            dt: time step
            z_des: desired latent state of shape (latent_dim, )
            kp: proportional gain. Scalar or array of shape (latent_dim, )
            ki: integral gain. Scalar or array of shape (latent_dim, )
            kd: derivative gain. Scalar or array of shape (latent_dim, )
            gamma: horizontal compression factor of the hyperbolic tangent. Array of shape () or (latent_dim, )
        Returns:
            tau: control input
            control_state: dictionary with the controller's stateful information. Contains entry with key "e_int" for the integral error.
            control_info: dictionary with control information
        """
        zw = x[..., : self.latent_dim]
        zw_d = x[..., self.latent_dim :]

        # compute error in the latent space
        error_z = z_des - zw

        # compute the feedback term
        u_fb = kp * error_z + ki * control_state["e_int"] - kd * zw_d

        # compute the feedforward term
        u_ff = self.Lambda_w @ z_des + jnp.tanh(
            z_des + self.bias
        )

        # compute the torque in latent space
        u = jnp.zeros_like(z_des)
        if self.apply_feedforward_term:
            u = u + u_ff
        if self.apply_feedback_term:
            u = u + u_fb

        # decode the control input into the input space
        tau = self.decode_input(u)

        # update the integral error
        control_state["e_int"] += jnp.tanh(gamma * error_z) * dt

        control_info = dict(
            tau=tau,
            tau_z=u,
            tau_z_ff=u_ff,
            tau_z_fb=u_fb,
            e_int=control_state["e_int"],
        )

        return tau, control_state, control_info
