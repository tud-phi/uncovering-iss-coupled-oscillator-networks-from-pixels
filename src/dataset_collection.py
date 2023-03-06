from alive_progress import alive_bar
import cv2
from diffrax import AbstractERK, diffeqsolve, Dopri5, ODETerm, SaveAt
from jax import Array, lax, random
import jax.numpy as jnp
from pathlib import Path
from typing import Callable, Dict, Generic, Type, TypeVar


def collect_dataset(
    ode_fn: Callable,
    rendering_fn: Callable,
    rng: random.KeyArray,
    num_simulations: int,
    horizon: Array,
    dt: Array,
    state_init_min: Array,
    state_init_max: Array,
    dataset_dir: str,
    solver: Generic = Dopri5(),
):
    # initiate ODE term from `ode_fn`
    ode_term = ODETerm(ode_fn)

    # initiate time steps array
    ts = jnp.arange(0, horizon, dt)

    # number of total samples
    num_samples = num_simulations * (ts.shape[0] - 1)
    # state dimension
    state_dim = state_init_min.shape[0]

    # dataset directory
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Generating dataset...")

    sample_idx = 0
    with alive_bar(num_simulations) as bar:
        for sim_idx in range(num_simulations):
            rng, rng_x0_sampling = random.split(rng)

            # generate initial state of the simulation
            x0 = random.uniform(
                rng_x0_sampling,
                state_init_min.shape,
                minval=state_init_min,
                maxval=state_init_max,
            )

            # simulate
            sol = diffeqsolve(
                ode_term,
                solver=solver,
                t0=ts[0],
                t1=ts[-1],
                dt0=dt,
                y0=x0,
                max_steps=100000,
                saveat=SaveAt(ts=ts),
            )

            # states along the simulation
            x_ts = sol.ys

            # dimension of the state space
            n_x = x_ts.shape[1]
            # dimension of configuration space
            n_q = n_x // 2

            # index for updating the dataset
            start_idx = sim_idx * (n_x - 1)

            labels = dict(x_ss=x_ts)
            # save the labels in dataset_dir
            jnp.savez(file=str(dataset_dir / f"sim-{sim_idx}_labels.npz"), **labels)

            for time_idx in range(x_ts.shape[0]):
                # configuration for current time step
                q = x_ts[time_idx, n_q:]

                # render the image
                img = rendering_fn(q)

                # save the image
                cv2.imwrite(
                    str(dataset_dir / f"sim-{sim_idx}_t-{time_idx}_rendering.jpeg"), img
                )

                # update sample index
                sample_idx += 1

            # update progress bar
            bar()
