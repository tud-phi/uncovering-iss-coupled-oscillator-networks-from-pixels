from functools import partial
from jax import Array, debug, jit
import jax.numpy as jnp
from typing import Callable


@jit
def imq_kernel(
    x: Array,
    y: Array,
    kernel_bandwidth: float = 1.0,
    scales: Array = jnp.array([0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]),
) -> Array:
    """
    Inverse Multiquadratics kernel k(x, y) = C / (C + ||x - y||^2)
    where C = 2 * d_z * sigma_z^2

    Arguments:
        x: samples from the first distribution. Shape: (batch_size, latent_dim)
        y: samples from the second distribution. Shape: (batch_size, latent_dim)
        kernel_bandwidth (float): The kernel bandwidth. Default: 1.0
        scales: 1D array. The scales to apply if using multi-scale imq kernels. If None, use a unique
            imq kernel. Default: [.1, .2, .5, 1., 2., 5, 10.].
    Returns:
        k: distance metric between x and y. Shape: (batch_size, batch_size)
    """
    N = x.shape[0]
    latent_dim = x.shape[-1]

    Cbase = 2 * latent_dim * kernel_bandwidth ** 2
    xy_dist_sq_norm = jnp.sum((
        jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0)
    ) ** 2, axis=-1)

    k = jnp.zeros((N, N))
    for scale in scales:
        C = scale * Cbase
        k = k + C / (C + xy_dist_sq_norm)

    return k


def rbf_kernel(x: Array, y: Array, kernel_bandwidth: float = 1.0) -> Array:
    """
    Radial Basis Function kernel k(x, y) = exp(-||x - y||^2 / C)
    where C = 2 * d_z * sigma_z^2
    Arguments:
        x: samples from the first distribution. Shape: (batch_size, latent_dim)
        y: samples from the second distribution. Shape: (batch_size, latent_dim)
        kernel_bandwidth (float): The kernel bandwidth. Default: 1
    Returns:
        k: distance metric between x and y. Shape: (batch_size, batch_size)
    """
    latent_dim = x.shape[-1]

    C = 2.0 * latent_dim * kernel_bandwidth ** 2
    xy_dist_sq_norm = jnp.sum((
          jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0)
    ) ** 2, axis=-1)

    k = jnp.exp(-xy_dist_sq_norm / C)

    return k


@partial(jit, static_argnames=("kernel_fn",))
def wae_mmd_loss(
    x_rec: Array,
    x_target: Array,
    z: Array,
    z_prior: Array,
    kernel_fn: Callable = imq_kernel,
) -> Array:
    """
    Regularization loss of a Wasserstein Auto-Encoder with a Maximum Mean Discrepancy loss.
    Implementation based on pythae
    https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/wae_mmd/wae_mmd_model.py
    Implementation by original authors: https://github.com/tolstikhin/wae/blob/master/wae.py
    Arguments:
        x_rec: reconstructed samples. Shape: (batch_size, *input_dim)
        x_target: target samples. Shape: (batch_size, *input_dim)
        z: latent samples. Shape: (batch_size, latent_dim)
        z_prior: prior latent samples. Shape: (batch_size, latent_dim)
        kernel_fn: kernel function to use. Default: imq_kernel
            Needs to implement the signature (x, y) -> k
    """
    N = z.shape[0]  # batch size

    # compute the kernel matrices
    k_zz = kernel_fn(z, z)
    k_zprior_zprior = kernel_fn(z_prior, z_prior)
    k_z_zprior = kernel_fn(z, z_prior)

    mmd_zz = (k_zz - jnp.diag(jnp.diag(k_zz))).sum() / (N * (N - 1))
    mmd_zprior_zprior = (
        k_zprior_zprior - jnp.diag(jnp.diag(k_zprior_zprior))
    ).sum() / (N * (N - 1))
    mmd_z_zprior = k_z_zprior.sum() / (N ** 2)

    mmd_loss = mmd_zz + mmd_zprior_zprior - 2 * mmd_z_zprior

    return mmd_loss
