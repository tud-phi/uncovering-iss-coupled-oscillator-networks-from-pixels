import flax.linen as nn
from jax import Array
import jax.numpy as jnp
import numpy as onp


def generate_positive_definite_matrix_from_params(
    n: int, a: Array, diag_shift: float = 1e-6, diag_eps: float = 2e-6
) -> Array:
    """
    Generate a positive definite matrix of shape (n, n) from a vector of parameters.
    Args:
        params: A vector of parameters of shape ((n^2 + n) / 2, ).
        diag_shift: A small value that is added to the diagonal entries of the matrix before the softplus.
        diag_eps: A small value that is added to the diagonal entries of the matrix after the softplus.
    Returns:
        A: A positive definite matrix of shape (n, n).
    """
    # construct empty triangular matrix
    tril_mat = jnp.zeros((n, n))
    # (i, j) indices of lower triangular matrix
    tril_indices = onp.tril_indices(n)
    # populate triangular matrix from vector
    tril_mat = tril_mat.at[tril_indices].set(a)

    # make sure that the diagonal entries are positive
    diag_indices = onp.diag_indices(n)
    tril_mat = tril_mat.at[diag_indices].set(
        nn.softplus(tril_mat[diag_indices] + diag_shift) + diag_eps
    )

    # construct mass matrix from triangular matrix
    A = tril_mat @ tril_mat.transpose()

    return A
