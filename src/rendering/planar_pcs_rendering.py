import cv2
from jax import Array, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict


def render_planar_pcs(
    forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int,
    height: int,
    line_thickness: int = 2,
    num_points: int = 20,
) -> onp.ndarray:
    """
    Renders a planar pcs soft robot in OpenCV.
    Args:
        forward_kinematics_fn: A function that computes the forward kinematics of the pendulum.
            Namely, the function should compute the SE(2) poses of the tip of a link
        params: A dictionary of robot parameters.
        q: An array of joint angles.
        width: The width (i.e. number of horizontal pixels) of the rendered image.
        height: The height (i.e. number of vertical pixels) of the rendered image.
        line_thickness: The thickness of the rendered lines in pixels.
        num_points: The number of points used for discretizing the backbone curve.
    Returns:
        img: A numpy array of shape (width, height, 3) containing the rendered image.
    """
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.5 * jnp.sum(params["l"]))  # pixel per meter
    robot_color = (0, 0, 0)  # black robot_color in BGR

    # vmap the forward kinematics function
    batched_forward_kinematics_fn = vmap(
        forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
    )

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), num_points)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(params, q, s_ps)

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, 0.6 * h], dtype=onp.int32
    )  # in uv pixel coordinates
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    cv2.polylines(
        img, [curve], isClosed=False, color=robot_color, thickness=line_thickness
    )

    return img
