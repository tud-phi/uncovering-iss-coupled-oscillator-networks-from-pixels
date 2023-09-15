from jax import Array
import jax.numpy as jnp


def time_alignment_loss(z_ts: Array, margin: float) -> Array:
    """
    Time alignment loss. This brings all the latent samples in z_ts up to within a certain distance of each other.
    Args:
        z_ts: latent trajectory of shape (horizon, latent_dim)
        margin: margin for the time alignment loss (i.e. the maximum distance between time-consecutive latent samples)
    Returns:
        loss: time alignment loss
    """
    # compute the distance between time-consecutive latent samples
    z_ts_diff = z_ts[1:] - z_ts[:-1]
    z_ts_diff_norm = jnp.linalg.norm(z_ts_diff, axis=-1)

    # compute the time alignment loss
    loss = jnp.mean(jnp.maximum(z_ts_diff_norm - margin, 0.0))

    return loss
