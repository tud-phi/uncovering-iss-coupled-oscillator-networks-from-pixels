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
        sim_duration: Array,
        sim_dt: Array,
        state_init_min: Array,
        state_init_max: Array,
        dataset_dir: str,
    solver: Generic = Dopri5(),
):
    # initiate ODE term from `ode_fn`
    ode_term = ODETerm(ode_fn)

    # initiate time steps array
    ts = jnp.arange(0, sim_duration, sim_dt)

    # number of total samples
    num_samples = num_simulations * (ts.shape[0] - 1)
    # state dimension
    state_dim = state_init_min.shape[0]

    # dataset directory
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # reserve memory for dataset
    dataset = dict(
        x_curr_ss=jnp.zeros((num_samples, state_dim)),
        x_next_ss=jnp.zeros((num_samples, state_dim)),
    )

    print("Generating dataset...")

    sample_idx = 0
    with alive_bar(num_simulations) as bar:
        for sim_idx in range(num_simulations):
            rng, rng_x0_sampling = random.split(rng)

            # generate initial state of the simulation
            x0 = random.uniform(rng_x0_sampling, state_init_min.shape, minval=state_init_min, maxval=state_init_max)

            # simulate
            sol = diffeqsolve(
                ode_term,
                solver=solver,
                t0=ts[0],
                t1=ts[-1],
                dt0=sim_dt,
                y0=x0,
                max_steps=100000,
                saveat=SaveAt(ts=ts)
            )

            # states along the simulation
            x_ts = sol.ys

            # dimension of the state space
            n_x = x_ts.shape[1]
            # dimension of configuration space
            n_q = (n_x // 2)

            # index for updating the dataset
            start_idx = sim_idx * (n_x - 1)

            # write samples to dataset
            dataset["x_curr_ss"] = lax.dynamic_update_slice(
                dataset["x_curr_ss"], x_ts[:-1], (start_idx, 0)
            )
            dataset["x_next_ss"] = lax.dynamic_update_slice(
                dataset["x_next_ss"], x_ts[1:], (start_idx, 0)
            )

            for time_idx in range(x_ts.shape[0]):
                # configuration for current time step
                q = x_ts[time_idx, n_q:]

                # render the image
                img = rendering_fn(q)

                # save the image
                cv2.imwrite(str(dataset_dir / f"sample_{sample_idx}.jpeg"), img)

                # update sample index
                sample_idx += 1

            # update progress bar
            bar()

    # save the dataset in problem_2/datasets/dataset_double_pendulum_dynamics.npz
    jnp.savez(file=str(dataset_dir / "dataset.npz"), **dataset)
