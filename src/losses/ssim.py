from jax import Array
import jax.numpy as jnp
import dm_pix
from typing import Union


def structural_similarity_index(
    input: Array,
    target: Array,
    data_min: Union[float, Array] = 0.0,
    data_max: Union[float, Array] = 1.0,
) -> Array:
    # normalize to the minimum being at 0
    delta_input = input - data_min
    delta_target = target - data_min

    # compute the structural similarity index
    ssim = dm_pix.ssim(delta_input, delta_target, max_val=data_max - data_min)
    return ssim
