from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from functools import partial
from jax import Array, jit, lax, random
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Union


def rollout_ode(
    ode_fn: Callable,
    ts: Array,
    sim_dt: Union[float, Array],
    x0: Array,
    tau: Array,
    rendering_fn: Optional[Callable] = None,
    solver: AbstractSolver = Dopri5(),
    grayscale_rendering: bool = True,
    normalize_rendering: bool = True,
    show_progress: bool = False,
) -> Dict[str, Array]:
    """
    Rollout system dynamics and rollout the system configurations along the way.
    Args:
        ode_fn: ODE function. It should have the following signature:
            ode_fn(t, x, tau) -> x_dot
        ts: Time steps to rollout the system dynamics.
        sim_dt: Time step used for simulation [s].
        x0: Initial state of the system.
        tau: Control input to the system.
        rendering_fn: Function to render the state of the system. It should have the following signature:
            rendering_fn(q) -> img. If None, no rendering is performed.
        solver: Diffrax solver to use for the simulation.
        grayscale_rendering: Whether to convert the rendering image to grayscale.
        normalize_rendering: Whether to normalize the rendering image to [0, 1].
        show_progress: Whether to show the progress bar.
    Returns:
        data_ts: Dictionary with the following keys:
            - t_ts: Time steps of the rollout.
            - x_ts: System configurations along the rollout.
            - tau: Control input to the system.
            - rendering_ts: Rendered images of the system configurations along the rollout. If rendering_fn is None, this
                key is not present.
    """
    # dimension of the state space
    n_x = x0.shape[0]
    # dimension of configuration space
    n_q = n_x // 2

    # jit the ode fn
    ode_fn = jit(ode_fn)
    # initiate ODE term from `ode_fn`
    ode_term = ODETerm(ode_fn)

    # simulate
    sol = diffeqsolve(
        ode_term,
        solver=solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=sim_dt,
        y0=x0,
        args=tau,
        max_steps=None,
        saveat=SaveAt(ts=ts),
    )

    # states along the simulation
    x_ts = sol.ys

    # define labels dict
    data_ts = dict(t_ts=ts, x_ts=x_ts, tau=tau)

    if rendering_fn is not None:
        rendering_ts = []
        iterator = range(ts.shape[0])
        if show_progress:
            iterator = tqdm(iterator)
        for time_idx in iterator:
            if show_progress:
                iterator.set_description(
                    f"Rendering configuration at time step {time_idx + 1} / {ts.shape[0]}"
                )
            # configuration for current time step
            q = x_ts[time_idx, :n_q]

            # render the image
            img = rendering_fn(q)

            if grayscale_rendering:
                # convert rendering image to grayscale
                img = tf.image.rgb_to_grayscale(img)

            if normalize_rendering:
                # normalize rendering image to [0, 1]
                img = tf.cast(img, tf.float32) / 128.0 - 1.0

            rendering_ts.append(jnp.array(img))

        data_ts["rendering_ts"] = jnp.stack(rendering_ts, axis=0)

    return data_ts
