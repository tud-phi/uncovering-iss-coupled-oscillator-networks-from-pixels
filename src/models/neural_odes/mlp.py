__all__ = ["MlpOde"]
from flax import linen as nn  # Linen API
from jax import Array, jacfwd
import jax.numpy as jnp
from typing import Callable, Dict, Tuple, Union

from .neural_ode_base import NeuralOdeBase


class MlpOde(NeuralOdeBase):
    """A simple MLP ODE."""

    latent_dim: int
    input_dim: int
    dynamics_order: int = 2

    num_layers: int = 5
    hidden_dim: int = 20
    nonlinearity: Callable = nn.sigmoid
    mechanical_system: bool = True

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        # concatenate the state and the control input
        tmp = jnp.concatenate([x, tau[: self.input_dim]], axis=-1)

        # pass through MLP
        for _ in range(self.num_layers - 1):
            tmp = nn.Dense(features=self.hidden_dim)(tmp)
            tmp = self.nonlinearity(tmp)

        match self.dynamics_order:
            case 1:
                # state dimension is latent_dim
                x_d = nn.Dense(features=self.latent_dim)(tmp)
            case 2:
                if self.mechanical_system:
                    # the velocity of the latent variables is given in the input
                    z_d = x[..., self.latent_dim :]
                    # the output of the MLP is interpreted as the acceleration of the latent variables
                    z_dd = nn.Dense(features=self.latent_dim)(tmp)
                    # concatenate the velocity and acceleration of the latent variables
                    x_d = jnp.concatenate([z_d, z_dd], axis=-1)
                else:
                    # state dimension is 2 * latent_dim
                    x_d = nn.Dense(features=2 * self.latent_dim)(tmp)
            case _:
                raise ValueError("Invalid dynamics order")

        return x_d

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
        z = x[..., : self.latent_dim]
        z_d = x[..., self.latent_dim :]

        # compute error in the latent space
        error_z = z_des - z

        # compute the feedback term
        u_fb = kp * error_z + ki * control_state["e_int"] - kd * z_d
        u = u_fb

        # decode the control input into the input space
        # linearize the system w.r.t. the actuation
        tau_eq = jnp.zeros((self.input_dim,))
        B = jacfwd(lambda x, tau: self.__call__(x, tau)[self.latent_dim :], argnums=1)(
            x, tau_eq
        )
        tau = B.T @ u
        """
        primals, f_vjp = nn.vjp(lambda mdl, tau: self.__call__(x, tau)[self.latent_dim:], self, tau_eq)
        tau = f_vjp(u)
        """

        # update the integral error
        control_state["e_int"] += jnp.tanh(gamma * error_z) * dt

        control_info = dict(
            tau=tau,
            tau_z=u,
            tau_z_ff=jnp.zeros_like(u),
            tau_z_fb=u_fb,
            e_int=control_state["e_int"],
        )

        return tau, control_state, control_info
