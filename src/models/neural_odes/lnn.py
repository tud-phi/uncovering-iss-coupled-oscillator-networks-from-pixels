__all__ = ["LnnOde"]

from flax import linen as nn  # Linen API
from jax import Array, debug, grad, hessian, jvp
import jax.numpy as jnp
from typing import Any, Callable, Tuple

from .neural_ode_base import NeuralOdeBase
from .utils import generate_positive_definite_matrix_from_params


default_kernel_init = nn.initializers.lecun_normal()


class MassMatrixNN(nn.Module):
    """
    A neural network that outputs a positive definite mass matrix.
    """

    num_layers: int = 5
    hidden_dim: int = 32
    nonlinearity: Callable = nn.softplus
    diag_shift: float = 1e-6
    diag_eps: float = 2e-6

    @nn.compact
    def __call__(self, z: Array) -> Array:
        """
        Args:
            x: latent state of shape (latent_dim, )
        Returns:
            A: positive definite matrix of shape (latent_dim, latent_dim)
        """
        latent_dim = z.shape[-1]

        tmp = z
        for _ in range(self.num_layers - 1):
            tmp = nn.Dense(features=self.hidden_dim)(tmp)
            tmp = self.nonlinearity(tmp)

        # vector of parameters for triangular matrix
        l_params = nn.Dense(features=int((latent_dim**2 + latent_dim) / 2))(tmp)
        M = generate_positive_definite_matrix_from_params(
            latent_dim, l_params, diag_shift=self.diag_shift, diag_eps=self.diag_eps
        )
        return M


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


class LnnOde(NeuralOdeBase):
    """
    Coupled oscillator ODE with trainable parameters.
    Formulated in such way that we can prove input-to-state stability.
    """

    latent_dim: int
    input_dim: int

    num_layers: int = 5
    hidden_dim: int = 32
    nonlinearity: Callable = nn.softplus
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
        # the latent variables are given in the input
        z = x[..., : self.latent_dim]
        # the velocity of the latent variables is given in the input
        z_d = x[..., self.latent_dim :]

        def lagrangian_fn(_z: Array, _z_d: Array) -> Tuple[Array, Array]:
            """
            Args:
                _z: latent state of shape (latent_dim, )
                _z_d: latent state derivative of shape (latent_dim, )
            Returns:
                _L: Lagrangian of shape ()
                _M: mass matrix of shape (latent_dim, latent_dim)
            """
            # compute the mass matrix
            _M = MassMatrixNN(
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                diag_shift=self.diag_shift,
                diag_eps=self.diag_eps,
            )(_z)

            # compute the kinetic energy
            _T = 0.5 * _z_d.transpose() @ _M @ _z_d
            # compute the potential energy
            _U = PotentialEnergyNN(
                num_layers=self.num_layers, hidden_dim=self.hidden_dim
            )(_z)
            # compute the Lagrangian
            _L = (_T - _U).squeeze()
            return _L, _M

        # compute the gradient of the Lagrangian with respect to the latent state
        # which is equal to the negative of the potential force
        # grad_fn = grad(lagrangian_fn, argnums=(0,), has_aux=True)
        # tau_pot, M = -grad_fn(z, z_d)
        dL_dz, M = grad(lagrangian_fn, argnums=0, has_aux=True)(z, z_d)
        tau_pot = -dL_dz

        # compute the corioli forces
        _, tau_corioli, _ = jvp(
            lambda _z: grad(lagrangian_fn, argnums=1, has_aux=True)(
                _z, z_d
            ),  # first differentiate with respect to z_d
            # then differentiate with respect to z_d and multiply by z_d with Jacobian-vector product
            # i.e., do C(q, q_d) @ q_d
            primals=(z,),
            tangents=(z_d,),
            has_aux=True,
        )

        z_dd = jnp.linalg.inv(M) @ (tau - tau_corioli - tau_pot)

        # concatenate the velocity and acceleration of the latent variables
        x_d = jnp.concatenate([z_d, z_dd], axis=-1)

        return x_d
