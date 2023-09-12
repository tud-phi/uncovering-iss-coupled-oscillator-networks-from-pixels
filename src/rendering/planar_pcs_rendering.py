import cv2
from jax import Array, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Optional, Tuple


def render_planar_pcs(
    forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int,
    height: int,
    origin_uv: Optional[Tuple] = None,
    line_thickness: int = 2,
    num_points: int = 25,
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
    ppm = h / (1.6 * jnp.sum(params["l"]))  # pixel per meter
    robot_color = (0, 0, 0)  # black robot_color in BGR

    # in uv pixel coordinates
    if origin_uv is None:
        origin_uv = (w // 2, h // 2)  # center of the image
    origin_uv = onp.array(origin_uv, dtype=onp.int32)

    # vmap the forward kinematics function
    batched_forward_kinematics_fn = vmap(
        forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
    )

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), num_points)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(params, q, s_ps)

    img = 255 * onp.ones((h, w, 3), dtype=jnp.uint8)  # initialize background to white
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((chi_ps[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = - curve[:, 1]

    cv2.polylines(
        img, [origin_uv + curve], isClosed=False, color=robot_color, thickness=line_thickness
    )

    return img
