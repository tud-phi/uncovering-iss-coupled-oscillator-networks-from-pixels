from alive_progress import alive_bar
import cv2
from diffrax import AbstractERK, diffeqsolve, Dopri5, ODETerm, SaveAt
from jax import Array, lax, random
import jax.numpy as jnp
from pathlib import Path
import shutil
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
    """
    Collect a simulated dataset for a given ODE system. The initial state is uniformly sampled from the given bounds.
    Args:
        ode_fn: ODE function. It should have the following signature:
            ode_fn(t, x) -> x_dot
        rendering_fn: Function to render the state of the system. It should have the following signature:
            rendering_fn(q) -> img
        rng: PRNG key for random number generation.
        num_simulations: Number of simulations to run.
        horizon: Duration of each simulation [s].
        dt: Time step used for simulation [s].
        state_init_min: Array with minimal values for the initial state of the simulation.
        state_init_max: Array with maximal values for the initial state of the simulation.
        dataset_dir: Directory to save the dataset.
        solver: Diffrax solver to use for the simulation.
    """
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
    # empty the directory
    shutil.rmtree(dataset_dir)

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
                max_steps=None,
                saveat=SaveAt(ts=ts),
            )

            # states along the simulation
            x_ts = sol.ys

            # dimension of the state space
            n_x = x_ts.shape[1]
            # dimension of configuration space
            n_q = n_x // 2

            # folder to save the simulation data
            sim_dir = dataset_dir / f"sim-{sim_idx}"
            sim_dir.mkdir(parents=True, exist_ok=True)

            labels = dict(x_ts=x_ts)
            # save the labels in dataset_dir
            jnp.savez(file=str(sim_dir / "labels.npz"), **labels)

            for time_idx in range(x_ts.shape[0]):
                # configuration for current time step
                q = x_ts[time_idx, :n_q]

                # render the image
                img = rendering_fn(q)

                # save the image
                cv2.imwrite(str(sim_dir / f"rendering_time_idx-{time_idx}.jpeg"), img)

                # update sample index
                sample_idx += 1

            # update progress bar
            bar()
