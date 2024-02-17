from jax import Array
import jax.numpy as jnp
from typing import Union


def peak_signal_to_noise_ratio(
    input: Array,
    target: Array,
    data_min: Union[float, Array] = 0.0,
    data_max: Union[float, Array] = 1.0,
    eps: float = 1e-8
) -> Array:
    # normalize to the minimum being at 0
    delta_input = input - data_min
    delta_target = target - data_min

    # compute the mean-squared error
    mse = jnp.mean(jnp.square(delta_input - delta_target))

    # compute the dynamic range
    dynamic_range = data_max - data_min

    # compute the peak signal-to-noise ratio
    psnr = 20 * jnp.log10(dynamic_range / (jnp.sqrt(mse) + eps))

    return psnr
