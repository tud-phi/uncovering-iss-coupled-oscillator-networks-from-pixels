_all_ = ["DconOde"]
from flax import linen as nn  # Linen API
from jax import Array, debug, grad
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .neural_ode_base import NeuralOdeBase
from .utils import generate_positive_definite_matrix_from_params


default_kernel_init = nn.initializers.lecun_normal()


class PotentialEnergyNN(nn.Module):
    """
    A neural network that outputs a potential energy.
    """

    num_layers: int = 5
    hidden_dim: int = 32
    nonlinearity: Callable = nn.softplus

    @nn.compact
    def __call__(self, z: Array) -> Array:
        """
        Args:
            x: latent state of shape (latent_dim, )
        Returns:
            U: potential energy of shape (1, )
        """
        tmp = z
        for _ in range(self.num_layers - 1):
            tmp = nn.Dense(features=self.hidden_dim)(tmp)
            tmp = self.nonlinearity(tmp)

        U = nn.Dense(features=1)(tmp)

        return U


class DconOde(NeuralOdeBase):
    """
    Deeply Coupled Oscillator Network (DCON) ODE with trainable parameters.
    The main difference with respect to CON is that we use a deep neural network to learn the potential.
    """

    latent_dim: int
    input_dim: int

    gamma: float = 1.0
    epsilon: float = 1.0

    num_layers: int = 5
    hidden_dim: int = 32

    param_dtype: Any = jnp.float32

    # control settings
    apply_feedforward_term: bool = True
    apply_feedback_term: bool = True

    def setup(self) -> None:
        self.coupling_potential_energy_nn = PotentialEnergyNN(
            num_layers=self.num_layers, hidden_dim=self.hidden_dim
        )

        # input-state coupling matrix
        self.V = self.param(
            "V",
            default_kernel_init,
            (self.latent_dim, self.input_dim),
            self.param_dtype,
        )

    def __call__(self, x: Array, tau: Array) -> Array:
        """
        Args:
            x: latent state of shape (2* latent_dim, )
            tau: control input of shape (input_dim, )
        Returns:
            x_d: latent state derivative of shape (2* latent_dim, )
        """

        # the latent variables are given in the input
        z = x[..., : self.latent_dim]
        # the velocity of the latent variables is given in the input
        z_d = x[..., self.latent_dim :]

        # the potential force is the gradient of the potential energy
        tau_pot = grad(self.coupling_potential_energy_fn)(z)

        z_dd = self.V @ tau - self.gamma * z - self.epsilon * z_d - tau_pot

        # concatenate the velocity and acceleration of the latent variables
        x_d = jnp.concatenate([z_d, z_dd], axis=-1)

        return x_d

    def coupling_potential_energy_fn(self, z: Array) -> Array:
        return self.coupling_potential_energy_nn(z).squeeze()

    def get_terms(self, coordinate: str = "z") -> Dict:
        """
        Get the terms of the Equations of Motion.
        Args:
            coordinate: coordinates in which to express the terms. Can be ["z", "zeta"]
        Returns:
            terms: dictionary with the terms of the EoM
        """
        # mass matrix
        B = jnp.eye(self.latent_dim)

        # oscillator impedance
        Lambda = self.gamma * jnp.eye(self.latent_dim)
        E = self.epsilon * jnp.eye(self.latent_dim)

        # V
        V: Array = self.get_variable("params", "V")

        match coordinate:
            case "z":
                terms = dict(
                    B=B,
                    Lambda=Lambda,
                    E=E,
                    coupling_potential_energy_fn=self.coupling_potential_energy_fn,
                    V=V,
                )
            case "zeta":
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
                B_zeta = J_h_inv.T @ B @ J_h_inv
                Lambda_zeta = J_h_inv.T @ Lambda @ J_h_inv
                E_zeta = J_h_inv.T @ E @ J_h_inv

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
                    Lambda=Lambda_zeta,
                    E=E_zeta,
                    coupling_potential_energy_fn=lambda zeta: self.coupling_potential_energy_fn(
                        J_h_inv @ zeta
                    ),
                    V=V_zeta,
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
            coordinate: coordinates in the state x is expressed. Can be ["z", "zeta"]
        Returns:
            V: energy of the system of shape ()
        """
        terms = self.get_terms(coordinate=coordinate)

        z = x[..., : self.latent_dim]
        z_d = x[..., self.latent_dim :]

        # compute the kinetic energy
        T = (0.5 * z_d[None, :] @ terms["B"] @ z_d[:, None]).squeeze()

        # compute the potential energy
        U = (0.5 * z[None, :] @ terms["Lambda"] @ z[:, None]).squeeze() + terms[
            "coupling_potential_energy_fn"
        ](z)

        # compute the total energy
        V = T + U

        return V

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

        z = x[..., : self.latent_dim]
        z_d = x[..., self.latent_dim :]

        # get mapping into collocated coordinates
        J_h, J_h_inv = terms["J_h"], terms["J_h_inv"]

        # map into collocated coordinates
        zeta, zeta_d = J_h @ z, J_h @ z_d
        zeta_des = J_h @ z_des

        # compute error in the collocated coordinates
        error_zeta = zeta_des - zeta

        # compute the feedback term
        tau_zeta_fb = kp * error_zeta + ki * control_state["e_int"] - kd * zeta_d

        # compute the feedforward term
        tau_zeta_ff = terms["Lambda"] @ zeta + grad(
            terms["coupling_potential_energy_fn"]
        )(zeta)

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
