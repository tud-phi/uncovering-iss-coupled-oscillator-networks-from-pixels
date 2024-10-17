import jax
from jax import Array, debug, jit, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
from typing import Tuple


def extract_states_from_dataset(ds: tf.data.Dataset) -> Tuple[Array, Array]:
    num_batches = len(ds)  # number of dataset samples
    x_ss: Array = None
    tau_ss: Array = None

    for batch_idx, batch in (pbar := tqdm(enumerate(ds.as_numpy_iterator()))):
        pbar.set_description(
            f"Extracting states from dataset: processing batch {batch_idx + 1} / {num_batches}"
        )
        x_bt = batch["x_ts"]
        tau_bt = jnp.repeat(
            jnp.expand_dims(batch["tau"], axis=1), x_bt.shape[1], axis=1
        )

        if batch_idx == 0:
            x_ss = jnp.zeros((num_batches,) + x_bt.shape)
            tau_ss = jnp.zeros((num_batches,) + tau_bt.shape)

        x_ss = x_ss.at[batch_idx].set(x_bt)
        tau_ss = tau_ss.at[batch_idx].set(tau_bt)

    x_ss = x_ss.reshape((-1, x_ss.shape[-1]))
    tau_ss = tau_ss.reshape((-1, tau_ss.shape[-1]))

    return x_ss, tau_ss


def denormalize_img(
    img: Array, threshold: float = 127.5, apply_threshold: bool = False
) -> Array:
    """
    Denormalize an image.
    Args:
        img: The normalized image as float in the range [-1, 1].
        apply_threshold: Whether to apply a threshold to the image.
    Returns:
        img: The denormalized image as integer in the range [0, 255].
    """
    img = (img + 1) * 127.5
    img = jnp.clip(img, 0, 255)

    if apply_threshold:
        img = jnp.where(img > threshold, 255, 0)

    return img.astype(jnp.uint8)
