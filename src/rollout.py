__all__ = ["rollout_ode", "rollout_ode_with_latent_space_control"]
from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from functools import partial
from jax import Array, jit, jvp, lax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple, Union

from src.rendering.normalization import preprocess_rendering


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
            img = preprocess_rendering(img, grayscale=grayscale_rendering, normalize=normalize_rendering)

            rendering_ts.append(jnp.array(img))

        data_ts["rendering_ts"] = jnp.stack(rendering_ts, axis=0)

    return data_ts


def rollout_ode_with_latent_space_control(
    ode_fn: Callable,
    rendering_fn: Callable,
    encode_fn: Callable,
    ts: Array,
    sim_dt: Union[float, Array],
    x0: Array,
    input_dim: int,
    latent_dim: int,
    solver: AbstractSolver = Dopri5(),
    control_fn: Optional[Callable] = None,
    control_state_init: Optional[Dict[str, Array]] = None,
    latent_velocity_source: str = "image-space-finite-differences",
    grayscale_rendering: bool = True,
    normalize_rendering: bool = True,
) -> Dict[str, Array]:
    """
    Rollout system dynamics and rollout the system configurations along the way.
    Args:
        ode_fn: ODE function. It should have the following signature:
            ode_fn(t, x, tau) -> x_dot
        rendering_fn: Function to render the state of the system. It should have the following signature:
            rendering_fn(q) -> img.
        encode_fn: Function to encode the rendered image. It should have the following signature:
            encode_fn(img) -> z.
        ts: Time steps for simulating the system dynamics. At each time step, the control input is computed and applied to the
            system.
        sim_dt: Time step used for simulation [s].
        x0: Initial state of the system.
        input_dim: Dimension of the control input.
        latent_dim: Dimension of the latent space.
        solver: Diffrax solver to use for the simulation.
        control_fn (Optional[Callable]): Function to compute the control input to the system. It should have the
            following signature:
                control_fn(t, x, control_state) -> tau, control_state, control_info.
            If None, no control is applied.
        control_state_init (Optional[Dict[str, Array]]): Initial control state.
        latent_velocity_source: Source of the latent velocity. It can be one of the following:
            - "image-space-finite-differences": Finite differences in the image space.
            - "latent-space-finite-differences": Finite differences in the latent space.
            - "latent-space-integration": Integrating the latent dynamics.
        grayscale_rendering: Whether to convert the rendering image to grayscale.
        normalize_rendering: Whether to normalize the rendering image to [0, 1].
    Returns:
        sim_ts: Dictionary with the following keys:
            - t_ts: Time steps of the rollout.
            - x_ts: System configurations along the rollout.
            - rendering_ts: Rendered images of the system configurations along the rollout.
    """
    # dimension of the state space
    n_x = x0.shape[0]
    # dimension of configuration space
    n_q = n_x // 2
    # dimension of latent space
    n_z = latent_dim
    # dimension of control input
    n_tau = input_dim
    # time step
    dt = ts[1] - ts[0]
    
    # initial control state
    if control_state_init is None:
        control_state_init = dict()

    # initial velocity
    q_d = x0[n_q:]
    assert (
        jnp.allclose(q_d, jnp.zeros((n_q,))),
        "Initial velocity must be zero as we currently have not estimation of initial latent velocity implemented.",
    )

    # jit the ode fn
    ode_fn = jit(ode_fn)
    # initiate ODE term from `ode_fn`
    ode_term = ODETerm(ode_fn)

    def discrete_forward_dynamics_fn(
        t0: Array, t1: Array, x0: Array, tau: Array
    ) -> Array:
        """
        Discrete forward dynamics function.
        Args:
            t0: Current time.
            t1: Next time.
            x0: Initial state of the system.
            tau: Control input to the system.
        Returns:
            x1: State of the system at the next time step.
        """
        sol = diffeqsolve(
            ode_term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=sim_dt,
            y0=x0,
            args=tau,
            max_steps=None,
        )

        x1 = sol.ys[-1]

        return x1

    def sim_step_fn(
        carry: Dict,
        input: Dict[str, Array],
    ) -> Tuple[Dict, Dict[str, Array]]:
        # extract the current state from the carry dictionary
        t_prior = carry["t_prior"]
        t_curr = carry["t"]
        x_curr = carry["x"]
        q_curr = x_curr[:n_q]
        q_d_curr = x_curr[n_q:]

        # render the image
        img_curr = rendering_fn(onp.array(q_curr))
        img_curr = preprocess_rendering(img_curr, grayscale=grayscale_rendering, normalize=normalize_rendering)
        # convert image to jax array
        img_curr = jnp.array(img_curr)

        # encode the image
        z_curr = encode_fn(img_curr[None, ...])[0]

        match latent_velocity_source:
            case "latent-space-finite-differences":
                # apply finite differences to the latent space to get the latent velocity
                z_d_curr = jnp.gradient(
                    jnp.stack([carry["xi_prior"][:n_z], z_curr], axis=0),
                    t_curr - t_prior,
                    axis=0
                )[0]
            case "image-space-finite-differences":
                # apply finite differences to the image space to get the image velocity
                img_d_curr = jnp.gradient(
                    jnp.stack([carry["img_prior"], img_curr], axis=0),
                    t_curr - t_prior,
                    axis=0
                )[0].astype(img_curr.dtype)

                # computing the jacobian-vector product is more efficient
                # than first computing the jacobian and then performing a matrix multiplication
                _, z_d_bt = jvp(
                    encode_fn,
                    (img_curr[None, ...],),
                    (img_d_curr[None, ...],),
                )
                z_d_curr = z_d_bt[0]
            case _:
                raise ValueError(
                    f"latent_velocity_source must be either 'direct-finite-differences' "
                    f"or 'image-space-finite-differences', but is {latent_velocity_source}"
                )

        # current latent state
        xi_curr = jnp.concatenate((z_curr, z_d_curr), axis=0)

        # save the current state and the state transition data
        step_data = dict(
            ts=t_curr,
            x_ts=x_curr,
            rendering_ts=img_curr,
            xi_ts=xi_curr,
        )

        if control_fn is not None:
            # compute the control input
            tau, control_state, control_info = control_fn(t_curr, xi_curr, carry["control_state"])
            control_info_ts = {f"{k}_ts": v for k, v in control_info.items()}
            step_data = step_data | control_info_ts
        else:
            tau = jnp.zeros((n_tau,))
            control_state = carry["control_state"]
            step_data["tau_ts"] = tau

        # perform integration
        t_next = input["ts"]
        x_next = discrete_forward_dynamics_fn(t_curr, t_next, x_curr, tau)

        # update the carry array for the next time step
        carry = dict(
            t=t_next,
            x=x_next,
            t_prior=t_curr,
            img_prior=img_curr,
            xi_prior=xi_curr,
            control_state=control_state
        )

        return carry, step_data

    # initialize carry
    # render the image
    img_prior_init = rendering_fn(onp.array(x0[:n_q]))
    if grayscale_rendering:
        # convert rendering image to grayscale
        img_prior_init = tf.image.rgb_to_grayscale(img_prior_init)
    if normalize_rendering:
        # normalize rendering image to [0, 1]
        img_prior_init = tf.cast(img_prior_init, tf.float32) / 128.0 - 1.0
    # convert image to jax array
    img_prior_init = jnp.array(img_prior_init)
    z_prior_init = encode_fn(img_prior_init[None, ...])[0]
    xi_prior_init = jnp.concatenate((z_prior_init, jnp.zeros((n_z,))), axis=0)
    carry = dict(
        t=ts[0],
        x=x0,
        t_prior=ts[0] - ts[1],
        img_prior=img_prior_init,
        xi_prior=xi_prior_init,
        control_state=control_state_init,
    )

    input_ts = dict(ts=ts[1:])

    _sim_ts = []
    for time_idx in (pbar := tqdm(range(ts.shape[0]))):
        pbar.set_description(f"Simulating time step {time_idx + 1} / {ts.shape[0]}")
        input = {k: v[time_idx] for k, v in input_ts.items()}

        carry, step_data = sim_step_fn(carry, input)
        _sim_ts.append(step_data)

    # define labels dict
    sim_ts = {k: jnp.stack([step_data[k] for step_data in _sim_ts]) for k in _sim_ts[0]}

    return sim_ts
