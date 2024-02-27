import numpy as onp
import tensorflow as tf


def preprocess_rendering(rendering: onp.ndarray, grayscale: bool = False, normalize: bool = True) -> onp.ndarray:
    """
    Preprocesses the rendering image.
    Args:
        rendering: A numpy array of shape (width, height, 3) containing the rendered image.
        grayscale: A boolean flag indicating whether the rendering image is in grayscale.
        normalize: A boolean flag indicating whether to normalize the rendering image to [0, 1].
    Returns:
        normalized_rendering: A numpy array containing the preprocessed rendered image.
    """
    if grayscale:
        # convert rendering image to grayscale
        rendering = tf.image.rgb_to_grayscale(rendering)

    if normalize:
        # normalize rendering image to [0, 1]
        rendering = tf.cast(rendering, tf.float32) / 128.0 - 1.0

    # cast image to numpy array
    normalized_rendering = onp.array(rendering)

    return normalized_rendering
