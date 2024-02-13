from jax import Array
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Tuple, Union


def discretize_state_space_model(
    A: Array, B: Array, dt: Union[Array, float], method: str = "zoh"
) -> Tuple[Array, Array]:
    """
    Discretize a continuous-time linear state space model.
    Args:
        A: state matrix of shape (state_dim, state_dim)
        B: input matrix of shape (state_dim, input_dim)
        dt: time step
        method: discretization method, one of ["zoh", "bilinear"]
    Returns:
        Ad: discretized state matrix of shape (state_dim, state_dim)
        Bd: discretized input matrix of shape (state_dim, input_dim)
    """
    if method == "zoh":
        # zero-order hold discretization
        # https://github.com/scipy/scipy/blob/v1.12.0/scipy/signal/_lti_conversion.py#L479C9-L493C32
        # Build an exponential matrix
        em_upper = jnp.hstack((A, B))

        # Need to stack zeros under the A and B matrices
        em_lower = jnp.hstack(
            (jnp.zeros_like(A, shape=(B.shape[1], A.shape[0])), jnp.zeros_like(B, shape=(B.shape[1], B.shape[1])))
        )
        em = jnp.vstack((em_upper, em_lower))
        ms = jsp.linalg.expm(dt * em)

        # Dispose of the lower rows
        ms = ms[: A.shape[0], :]
        Ad = ms[:, 0 : A.shape[1]]
        Bd = ms[:, A.shape[1] :]
    elif method == "bilinear":
        # Bilinear (Tustin) approximation
        # https://srush.github.io/annotated-s4/#part-1-state-space-models
        I = jnp.eye(A.shape[0], dtype=A.dtype)
        BL = jnp.linalg.inv(I - (dt / 2.0) * A)
        Ad = BL @ (I + (dt / 2.0) * A)
        Bd = (BL * dt) @ B
    else:
        raise ValueError(f"Unknown discretization method: {method}")

    return Ad, Bd
