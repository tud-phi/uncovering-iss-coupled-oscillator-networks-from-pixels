from jax import Array
import jax.numpy as jnp


def kullback_leiber_divergence(mu: Array, logvar: Array):
    """
    Kullback-Leiber divergence loss function.
    Arguments:
        mu: mean of the latent distribution. Shape: (batch_size, latent_dim)
        logvar: log-variance of the latent distribution. Shape: (batch_size, latent_dim)
    Returns:
        kld: the Kullback-Leiber divergence. Shape: ()
    """
    return jnp.mean(-0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1))
