import cv2
from jax import Array, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict


def render_pendulum(
    forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int,
    height: int,
    line_thickness: int = 2,
) -> onp.ndarray:
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.5 * jnp.sum(params["l"]))  # pixel per meter
    robot_color = (0, 0, 0)  # black robot_color in BGR

    batched_forward_kinematics_fn = vmap(
        forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
    )

    # poses along the robot of shape (3, N)
    link_indices = jnp.arange(params["l"].shape[0], dtype=jnp.int32)
    chi_ls = jnp.zeros((3, link_indices.shape[0] + 1))
    chi_ls = chi_ls.at[:, 1:].set(batched_forward_kinematics_fn(params, q, link_indices))

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, h // 2], dtype=onp.int32
    )  # in x-y pixel coordinates
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((curve_origin + chi_ls[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    cv2.polylines(img, [curve], isClosed=False, color=robot_color, thickness=line_thickness)

    return img
