from functools import partial
from jax import Array, debug, jit, random
import jax.numpy as jnp
from typing import Callable, Optional, Sequence


@jit
def imq_kernel(
    x: Array,
    y: Array,
    C: Array,
    scales: Array = jnp.array([0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]),
) -> Array:
    """
    Inverse Multiquadratics kernel k(x, y) = C / (C + ||x - y||^2)
    where C = 2 * d_z * sigma_z^2

    Arguments:
        x: samples from the first distribution. Shape: (batch_size, latent_dim)
        y: samples from the second distribution. Shape: (batch_size, latent_dim)
        C: the kernel bandwidth. Shape: (1,)
        scales: 1D array. The scales to apply if using multi-scale imq kernels. If None, use a unique
            imq kernel. Default: [.1, .2, .5, 1., 2., 5, 10.].
    Returns:
        k: distance metric between x and y. Shape: (batch_size, batch_size)
    """
    N = x.shape[0]
    latent_dim = x.shape[-1]

    xy_dist_sq_norm = jnp.sum(
        (jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0)) ** 2, axis=-1
    )

    k = jnp.zeros((N, N))
    for scale in scales:
        C_scaled = scale * C
        k = k + C_scaled / (C_scaled + xy_dist_sq_norm)

    return k


def rbf_kernel(x: Array, y: Array, C: Array) -> Array:
    """
    Radial Basis Function kernel k(x, y) = exp(-||x - y||^2 / C)
    where C = 2 * d_z * sigma_z^2
    Arguments:
        x: samples from the first distribution. Shape: (batch_size, latent_dim)
        y: samples from the second distribution. Shape: (batch_size, latent_dim)
        C: the kernel bandwidth. Shape: (1,)
    Returns:
        k: distance metric between x and y. Shape: (batch_size, batch_size)
    """
    latent_dim = x.shape[-1]

    xy_dist_sq_norm = jnp.sum(
        (jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0)) ** 2, axis=-1
    )

    k = jnp.exp(-xy_dist_sq_norm / C)

    return k


def make_wae_mdd_loss(
    kernel_fn: Callable = imq_kernel,
    distribution: str = "normal",
    sigma_z: Array = jnp.array(1.0),
    uniform_distr_range: Sequence = (-1.0, 1.0),
) -> Callable:
    """
    Factory function to create a wae_mmd_loss function with a given kernel function.
    Arguments:
        kernel_fn: kernel function to use. Default: imq_kernel
                Needs to implement the signature (x, y, C=C) -> k
        distribution: prior distribution of the latent space. Default: "normal" (Gaussian).
                Needs to be one of ["normal", "uniform"]
        sigma_z: standard deviation of the normal distribution. Default: 1.0
        uniform_distr_range: range of the uniform distribution. Default: (-1.0, 1.0)
    Returns:
        wae_mmd_loss_fn: wae_mmd_loss function with the given kernel function. Signature:
            (x_rec, x_target, z, rng) -> mmd_loss
    """

    @jit
    def wae_mmd_loss_fn(
        x_rec: Array,
        x_target: Array,
        z: Array,
        z_prior: Array = None,
        rng: Optional[KeyArray] = None,
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
            rng: PRNGKey for random number generation
        Returns:
            mmd_loss: the MMD loss. Shape: ()
        """
        N = z.shape[0]  # batch size

        if distribution == "normal":
            C = 2 * z.shape[-1] * sigma_z**2
        elif distribution == "uniform":
            C = z.shape[-1]
        else:
            raise ValueError(
                f"Unknown prior distribution {distribution}. Needs to be one of ['normal', 'uniform']"
            )

        if z_prior is None:
            # we need to sample from the prior distribution
            if distribution == "normal":
                z_prior = sigma_z * random.normal(rng, shape=z.shape)
            elif distribution == "uniform":
                z_prior = random.uniform(
                    rng,
                    shape=z.shape,
                    minval=uniform_distr_range[0],
                    maxval=uniform_distr_range[1],
                )
            else:
                raise ValueError(
                    f"Unknown prior distribution {distribution}. Needs to be one of ['normal', 'uniform']"
                )

        # compute the kernel matrices
        k_zz = kernel_fn(z, z, C=C)
        k_zprior_zprior = kernel_fn(z_prior, z_prior, C=C)
        k_z_zprior = kernel_fn(z, z_prior, C=C)

        mmd_zz = (k_zz - jnp.diag(jnp.diag(k_zz))).sum() / (N * (N - 1))
        mmd_zprior_zprior = (
            k_zprior_zprior - jnp.diag(jnp.diag(k_zprior_zprior))
        ).sum() / (N * (N - 1))
        mmd_z_zprior = k_z_zprior.sum() / (N**2)

        mmd_loss = mmd_zz + mmd_zprior_zprior - 2 * mmd_z_zprior

        return mmd_loss

    return wae_mmd_loss_fn
