import jax
import jax.numpy as jnp


def apply_eps_to_array(a: jax.Array, eps: float = 1e-6) -> jax.Array:
    """
    Add a small number to avoid singularities if the entries are too close to zero.
    Args:
        a: input array
        eps: small number to add
    """
    # get the sign of the entries
    a_sign = jnp.sign(a)
    # set zero sign to 1 (i.e. positive)
    a_sign = jnp.where(a_sign == 0, 1, a_sign)
    # add eps to the diagonal
    a_epsed = jax.lax.select(
        jnp.abs(a) < eps,
        a_sign * eps,
        a,
    )

    return a_epsed
