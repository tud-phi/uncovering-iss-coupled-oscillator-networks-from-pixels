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
            Lambda = Lambda_w @ W
            E = E_w @ W

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

    def get_terms(self, coordinate: str = "z") -> Dict[str, Array]:
        """
        Get the terms of the Equations of Motion.
        Args:
            coordinate: coordinates in which to express the terms. Can be ["z", "zw", "zeta"]
        Returns:
            terms: dictionary with the terms of the EoM
        """
        # extract the matrices from the neural network
        bias = self.get_variable("params", "bias")

        # Lambda_w
        lambda_w = self.get_variable("params", "lambda_w")
        Lambda_w = generate_positive_definite_matrix_from_params(
            self.latent_dim,
            lambda_w,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )

        # E_w
        e_w = self.get_variable("params", "e_w")
        E_w = generate_positive_definite_matrix_from_params(
            self.latent_dim, e_w, diag_shift=self.diag_shift, diag_eps=self.diag_eps
        )

        # V
        V = self.get_variable("params", "V")

        if self.use_w_coordinates:
            # constructing Bw_inv as a positive definite matrix
            b_w_inv = self.get_variable("params", "b_w_inv")
            B_w_inv = generate_positive_definite_matrix_from_params(
                self.latent_dim,
                b_w_inv,
                diag_shift=self.diag_shift,
                diag_eps=self.diag_eps,
            )

            # compute jacobian mapping from z to zw coordinates
            J_w = B_w_inv

            # computing B_w from B_w_inv
            B_w = jnp.linalg.inv(B_w_inv)
        else:
            # constructing Bw as a positive definite matrix
            # vector of parameters for triangular matrix
            b_w = self.get_variable("params", "b_w")
            B_w = generate_positive_definite_matrix_from_params(
                self.latent_dim, b_w, diag_shift=self.diag_shift, diag_eps=self.diag_eps
            )

            # compute jacobian mapping from z to zw coordinates
            J_w = jnp.linalg.inv(B_w)

        match coordinate:
            case "z":
                terms = dict(
                    B=jnp.eye(self.latent_dim),
                    W=J_w,
                    bias=bias,
                    Lambda=Lambda_w @ J_w,
                    E=E_w @ J_w,
                    V=V,
                    J_w=J_w,
                )
            case "zw":
                terms = dict(
                    B=B_w,
                    W=jnp.eye(self.latent_dim),
                    bias=bias,
                    Lambda=Lambda_w,
                    E=E_w,
                    V=V,
                    J_w=J_w,
                )
            case "zeta":
                assert (
                    self.input_nonlinearity is None
                ), "Mapping into collocated coordinates is only implemented for dynamics affine in control."
                assert (
                    self.latent_dim >= self.input_dim
                ), "Mapping into collocated coordinates is only implemented for systems with latent_dim >= input_dim."

                # compute the Jacobian of the map into collocated coordinates
                J_h = jnp.concatenate(
                    [
                        V.T,
                        jnp.concatenate(
                            [
                                jnp.zeros(
                                    (self.latent_dim - self.input_dim, self.input_dim)
                                ),
                                jnp.eye(self.latent_dim - self.input_dim),
                            ],
                            axis=1,
                        ),
                    ],
                    axis=0,
                )
                # compute the inverse of the Jacobian
                J_h_inv = jnp.linalg.inv(J_h)

                # map terms from w into collocated coordinates
                B_zeta = J_h_inv.T @ B_w @ J_h_inv
                Lambda_zeta = J_h_inv.T @ Lambda_w @ J_h_inv
                E_zeta = J_h_inv.T @ E_w @ J_h_inv

                # actuation matrix in collocated coordinates
                V_zeta = jnp.concatenate(
                    [
                        jnp.eye(self.input_dim),
                        jnp.zeros((self.latent_dim - self.input_dim, self.input_dim)),
                    ],
                    axis=0,
                )

                terms = dict(
                    B=B_zeta,
                    W=J_h_inv,
                    bias=bias,
                    Lambda=Lambda_zeta,
                    E=E_zeta,
                    V=V_zeta,
                    J_w=J_w,
                    J_h=J_h,
                    J_h_inv=J_h_inv,
                )
            case _:
                raise ValueError(f"Coordinate {coordinate} not supported.")

        return terms

    def energy_fn(self, x: Array, coordinate: str = "z") -> Array:
        """
        Compute the energy of the system.
        Args:
            x: state of shape (2* latent_dim, )
            coordinate: coordinates in the state x is expressed. Can be ["z", "zw", "zeta"]
        Returns:
            V: energy of the system of shape ()
        """
        terms = self.get_terms(coordinate=coordinate)

        z = x[..., : self.latent_dim]
        z_d = x[..., self.latent_dim :]

        # compute the kinetic energy
        T = (0.5 * z_d[None, :] @ terms["B"] @ z_d[:, None]).squeeze()

        # compute the potential energy
        U = (
            0.5 * z[None, :] @ terms["Lambda"] @ z[:, None]
            + jnp.sum(jnp.log(jnp.cosh(terms["W"] @ z + terms["bias"])))
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
        P-satI-D feedback together with potential force compensation in the latent variables.
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
        z = x[..., : self.latent_dim]
        z_d = x[..., self.latent_dim :]

        if self.use_w_coordinates:
            terms = self.get_terms(coordinate="zw")
        else:
            terms = self.get_terms(coordinate="z")

        # compute error in the latent space
        error_z = z_des - z

        # compute the feedback term
        tau_z_fb = kp * error_z + ki * control_state["e_int"] - kd * z_d

        # compute the feedforward term
        tau_z_ff = terms["Lambda"] @ z_des + jnp.tanh(
            terms["W"] @ z_des + terms["bias"]
        )

        # compute the torque in latent space
        tau_z = jnp.zeros_like(z_des)
        if self.apply_feedforward_term:
            tau_z = tau_z + tau_z_ff
        if self.apply_feedback_term:
            tau_z = tau_z + tau_z_fb

        # extract the V matrix from the neural network
        V = terms["V"]

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

        # update the integral error
        control_state["e_int"] += jnp.tanh(gamma * error_z) * dt

        control_info = dict(
            tau=tau,
            tau_z=tau_z,
            tau_z_ff=tau_z_ff,
            tau_z_fb=tau_z_fb,
            e_int=control_state["e_int"],
        )

        return tau, control_state, control_info

    def setpoint_regulation_collocated_form_fn(
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
        P-satI-D feedback together with potential force compensation in the collocated variables.
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
        assert (
            self.input_nonlinearity is None
        ), "Mapping into collocated coordinates is only implemented for dynamics affine in control."
        assert (
            self.latent_dim >= self.input_dim
        ), "Mapping into collocated coordinates is only implemented for systems with latent_dim >= input_dim."

        terms = self.get_terms(coordinate="zeta")

        if self.use_w_coordinates:
            zw = x[..., : self.latent_dim]
            zw_d = x[..., self.latent_dim :]
            zw_des = z_des
        else:
            z = x[..., : self.latent_dim]
            z_d = x[..., self.latent_dim :]

            # map into w-coordinates
            zw, zw_d = terms["J_w"] @ z, terms["J_w"] @ z_d
            zw_des = terms["J_w"] @ z_des

        J_h, J_h_inv = terms["J_h"], terms["J_h_inv"]

        # map into collocated coordinates
        zeta, zeta_d = J_h @ zw, J_h @ zw_d
        zeta_des = J_h @ zw_des

        # compute error in the collocated coordinates
        error_zeta = zeta_des - zeta

        # compute the feedback term
        tau_zeta_fb = kp * error_zeta + ki * control_state["e_int"] - kd * zeta_d

        # compute the feedforward term
        tau_zeta_ff = terms["Lambda"] @ zeta_des + J_h_inv.T @ jnp.tanh(
            terms["W"] @ zeta_des + terms["bias"]
        )

        # compute the torque in latent space
        tau_zeta = jnp.zeros_like(zeta_des)
        if self.apply_feedforward_term:
            tau_zeta = tau_zeta + tau_zeta_ff
        if self.apply_feedback_term:
            tau_zeta = tau_zeta + tau_zeta_fb

        # take the first input_dim rows as the control input
        tau = tau_zeta[: self.input_dim]

        # update the integral error
        control_state["e_int"] += jnp.tanh(gamma * error_zeta) * dt

        control_info = dict(
            tau=tau,
            tau_zeta=tau_zeta,
            tau_zeta_ff=tau_zeta_ff,
            tau_zeta_fb=tau_zeta_fb,
            zeta=zeta,
            zeta_d=zeta_d,
            zeta_des=zeta_des,
            e_int=control_state["e_int"],
        )

        return tau, control_state, control_info
