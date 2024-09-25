from jax import Array
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
from typing import Union


def preprocess_rendering(
    rendering: Union[onp.ndarray, Array],
    grayscale: bool = False,
    normalize: bool = True,
    img_min_val: float = 0.0,
    img_max_val: float = 255.0,
) -> Union[onp.ndarray, Array]:
    """
    Preprocesses the rendering image.
    Args:
        rendering: A numpy array of shape (width, height, 3) containing the rendered image.
        grayscale: A boolean flag indicating whether the rendering image is in grayscale.
        normalize: A boolean flag indicating whether to normalize the rendering image to [-1, 1].
        img_min_val: The minimum pixel value of the rendering image.
        img_max_val: The maximum pixel value of the rendering image.
    Returns:
        normalized_rendering: A numpy array containing the preprocessed rendered image.
    """
    input = rendering

    if grayscale:
        # convert rendering image to grayscale
        rendering = tf.image.rgb_to_grayscale(rendering)

    if normalize:
        # normalize rendering image to [-1, 1]
        rendering = (tf.cast(rendering, tf.float32) - img_min_val) / (img_max_val - img_min_val) * 2.0 - 1.0

    if isinstance(input, jnp.ndarray):
        # cast image to jax array
        normalized_rendering = jnp.array(rendering)
    elif isinstance(input, onp.ndarray):
        # cast image to numpy array
        normalized_rendering = onp.array(rendering)
    else:
        normalized_rendering = rendering

    return normalized_rendering
