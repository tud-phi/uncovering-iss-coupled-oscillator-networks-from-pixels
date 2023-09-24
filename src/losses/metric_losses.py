from functools import partial
from jax import Array, debug, random, vmap
import jax.numpy as jnp


def sum_squared_distance(x1: Array, x2: Array) -> Array:
    """
    Compute the sum of the squared distance between two vectors along the last dimension.
    Args:
        x1: first vector
        x2: second vector
    Returns:
        distance: Euclidean distance between x1 and x2
    """
    return jnp.sum((x1 - x2) ** 2, axis=-1)


def positive_alignment_loss(x1: Array, x2: Array, margin: float) -> Array:
    """
    Alignment loss between the two positive (latent) vectors x1 and x2.

    Args:
        x1: first vector of shape (N, )
        x2: second vector of shape (N, )
        margin: margin for the contrastive loss as a scalar or of shape (N, )
    Returns:
        loss: contrastive loss
    """
    # compute the Euclidean distance between x1 and x2
    distance = sum_squared_distance(x1, x2)

    # compute the contrastive loss
    loss = jnp.clip(distance - margin, a_min=0.0, a_max=None)

    return loss


def batch_time_alignment_loss(z_bt: Array, margin: float) -> Array:
    """
    Batch time alignment loss.
    This brings all the time-consecutive latent samples in z_bt up to within a certain distance of each other.
    Args:
        z_bt: latent batch of shape (batch_size, horizon, latent_dim)
        margin: margin for the batch time alignment loss (i.e. the minimum distance between time-separate latent samples)
        rng: random number generator
    Returns:
        loss: batch time alignment loss
    """
    horizon = z_bt.shape[1]
    start_time_indices = jnp.arange(horizon - 1)

    loss = jnp.mean(
        # vmap over the batch dimension
        vmap(
            # vmap over the time dimension
            vmap(
                lambda z_ts, start_time_idx: positive_alignment_loss(z_ts[start_time_idx], z_ts[start_time_idx + 1], margin=margin),
                in_axes=(None, 0),
                out_axes=0,
            ),
            in_axes=(0, None),
            out_axes=0,
        )(z_bt, start_time_indices)
    )

    return loss


def contrastive_loss(x1: Array, x2: Array, gamma: float, margin: Array) -> Array:
    """
    Contrastive loss between the two (latent) vectors x1 and x2.

    References:
    https://towardsdatascience.com/the-why-and-the-how-of-deep-metric-learning-e70e16e199c0
    KAYA, Mahmut, and Hasan Şakir BİLGE. 2019. "Deep Metric Learning: A Survey" Symmetry 11, no. 9: 1066.
    https://doi.org/10.3390/sym11091066

    Args:
        x1: first vector of shape (N, )
        x2: second vector of shape (N, )
        gamma: scalar defining if x1 and x2 form a positive (value 1) or a negative pair (value 0).
        margin: margin for the contrastive loss as a scalar or of shape (N, )
    Returns:
        loss: contrastive loss
    """
    # compute the Euclidean distance between x1 and x2
    distance = sum_squared_distance(x1, x2)

    # compute the contrastive loss
    loss = gamma * distance + (1.0 - gamma) * jnp.clip(
        margin - distance, a_min=0.0, a_max=None
    )

    return loss


def batch_time_contrastive_loss(
    z_bt: Array, margin: float, rng: random.KeyArray
) -> Array:
    """
    Batch contrastive loss.
    This brings all the time-consecutive latent samples in z_bt up to within a certain distance of each other.
    Args:
        z_bt: latent batch of shape (batch_size, horizon, latent_dim)
        margin: margin for the batch contrastive loss (i.e. the minimum distance between time-separate latent samples)
        rng: random number generator
    Returns:
        loss: batch contrastive loss
    """
    horizon = z_bt.shape[1]
    rng, subkey1, subkey2, subkey3, subkey4 = random.split(rng, 5)

    # generate the contrastive loss for positive (i.e., time-consecutive) pairs
    # randomly select a time index for the positive (time-consecutive) latent sample
    pos_batch_permutation = jnp.arange(z_bt.shape[0])
    pos_time_idx = random.randint(
        subkey1, (z_bt.shape[0],), minval=0, maxval=horizon - 1
    )
    pos_loss = jnp.mean(
        vmap(
            partial(contrastive_loss, gamma=1.0, margin=margin),
            in_axes=(0, 0),
            out_axes=0,
        )(z_bt[pos_batch_permutation, pos_time_idx], z_bt[pos_batch_permutation, pos_time_idx + 1])
    )

    # generate the contrastive loss for negative (i.e., time-separate) pairs
    neg_batch_permutation = random.permutation(subkey2, z_bt.shape[0])
    neg_first_time_idx = random.randint(
        subkey3,
        (z_bt.shape[0],),
        minval=0,
        maxval=jnp.clip(horizon // 2 - 2, a_min=0, a_max=None),
    )
    neg_second_time_idx = random.randint(
        subkey4,
        (z_bt.shape[0],),
        minval=jnp.clip(horizon // 2 + 1, a_min=None, a_max=horizon),
        maxval=horizon,
    )
    neg_loss = jnp.mean(
        vmap(
            partial(contrastive_loss, gamma=0.0, margin=margin),
            in_axes=(0, 0),
            out_axes=0,
        )(z_bt[pos_batch_permutation, neg_first_time_idx], z_bt[neg_batch_permutation, neg_second_time_idx])
    )

    return pos_loss + neg_loss


def triplet_loss(x_a: Array, x_p: Array, x_n: Array, margin: float) -> Array:
    """
    Triplet loss between the anchor, positive, and negative vectors.
    Args:
        x_a: anchor vector of shape (N, )
        x_p: positive vector of shape (N, )
        x_n: negative vector of shape (N, )
        margin: margin for the triplet loss
    Returns:
        loss: triplet loss
    """
    # compute the Euclidean distance between the anchor and positive vectors
    distance_ap = sum_squared_distance(x_a, x_p)

    # compute the Euclidean distance between the anchor and negative vectors
    distance_an = sum_squared_distance(x_a, x_n)

    # compute the triplet loss
    loss = jnp.clip(distance_ap - distance_an + margin, a_min=0.0, a_max=None)

    return loss


def batch_time_triplet_loss(z_bt: Array, margin: float, rng: random.KeyArray) -> Array:
    """
    Batch triplet loss.
    This brings all the time-consecutive latent samples in z_bt up to within a certain distance of each other.
    Args:
        z_bt: latent batch of shape (batch_size, horizon, latent_dim)
        margin: margin for the batch triplet loss
        rng: random number generator
    Returns:
        loss: batch triplet loss
    """
    horizon = z_bt.shape[1]
    rng, subkey1, subkey2, subkey3 = random.split(rng, 4)

    # identify anchor, positive, and negative latent samples
    time_idx_anchor = random.randint(
        subkey1,
        (z_bt.shape[0],),
        minval=0,
        maxval=jnp.clip(horizon // 2 - 2, a_min=0, a_max=None),
    )
    time_idx_pos = time_idx_anchor + 1
    time_idx_neg = random.randint(
        subkey2,
        (z_bt.shape[0],),
        minval=jnp.clip(horizon // 2 + 1, a_min=None, a_max=horizon),
        maxval=horizon,
    )
    batch_no_permutation = jnp.arange(z_bt.shape[0])
    batch_permutation = random.permutation(subkey3, z_bt.shape[0])

    loss = jnp.mean(
        vmap(
            partial(triplet_loss, margin=margin),
            in_axes=(0, 0, 0),
            out_axes=0,
        )(
            z_bt[batch_no_permutation, time_idx_anchor],
            z_bt[batch_no_permutation, time_idx_pos],
            z_bt[batch_permutation, time_idx_neg],
        )
    )

    return loss
