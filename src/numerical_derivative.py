from jax import Array
import jax.numpy as jnp
from scipy.signal import savgol_filter
from typing import Tuple


def savotzky_golay_scipy(dt: float, img_ts: Array, window_length: int = 3, polyorder: int = 2) -> Tuple[Array, Array]:
    """
    Savotzky-Golay filter implemented using scipy.
    Args:
        dt: time step of the simulation
        img_ts: image time series
        window_length: length of the filter window
        polyorder: order of the polynomial used to fit the samples. Must be less than window_length.
    Returns:
        img_d_ts: numerical 1st derivative of image time series
        img_dd_ts: numerical 2nd derivative of image time series
    """
    img_d_ts = savgol_filter(img_ts, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt, axis=0)
    img_dd_ts = savgol_filter(img_ts, window_length=window_length, polyorder=polyorder, deriv=2, delta=dt, axis=0)

    return img_d_ts, img_dd_ts
