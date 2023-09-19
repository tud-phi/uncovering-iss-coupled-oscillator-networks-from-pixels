from alive_progress import alive_bar
import cv2
from diffrax import AbstractERK, AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
import dill
from functools import partial
from jax import Array, jit, lax, random
import jax.numpy as jnp
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, Optional, Type, TypeVar


def collect_dataset(
    ode_fn: Callable,
    rendering_fn: Callable,
    rng: random.KeyArray,
    num_simulations: int,
    horizon_dim: int,
    dt: Array,
    x0_min: Array,
    x0_max: Array,
    dataset_dir: str,
    solver: AbstractSolver = Dopri5(),
    sim_dt: Optional[Array] = None,
    system_params: Optional[Dict[str, Array]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    x0_sampling_dist: str = "uniform",
    tau_max: Optional[Array] = None,
    save_raw_data: bool = False,
):
    """
    Collect a simulated dataset for a given ODE system. The initial state is uniformly sampled from the given bounds.
    Args:
        ode_fn: ODE function. It should have the following signature:
            ode_fn(t, x, tau) -> x_dot
        rendering_fn: Function to render the state of the system. It should have the following signature:
            rendering_fn(q) -> img
        rng: PRNG key for random number generation.
        num_simulations: Number of simulations to run.
        horizon_dim: Number of samples in each trajectory.
        dt: Time step used for samples [s].
        x0_min: Array with minimal values for the initial state of the simulation.
        x0_max: Array with maximal values for the initial state of the simulation.
        dataset_dir: Directory to save the dataset.
        solver: Diffrax solver to use for the simulation.
        sim_dt: Time step used for simulation [s].
        system_params: Dictionary with system parameters.
        metadata: Dictionary with metadata to save in the dataset directory.
        x0_sampling_dist: Distribution to sample the initial state of the simulation from. Can be one of:
            ["uniform", "arcsine", "half-normal"].
        tau_max: Maximal torque to apply to the system where actual torque is sampled from a uniform distribution.
        save_raw_data: Whether to save the raw data (as images and labels) to the dataset_dir.
    """
    # initiate time steps array of samples
    ts = jnp.arange(0, horizon_dim * dt, step=dt)

    # if the simulation time-step is not given, initialize it to the same value as the sample time step
    if sim_dt is None:
        sim_dt = dt
    else:
        assert (
            sim_dt <= dt
        ), "The simulation time step needs to be smaller than the sampling time step."

    # jit the ode fn
    ode_fn = jit(ode_fn)
    # initiate ODE term from `ode_fn`
    ode_term = ODETerm(ode_fn)

    # number of total samples
    num_samples = num_simulations * (ts.shape[0] - 1)
    # state dimension
    state_dim = x0_min.shape[0]

    # dataset directory
    dataset_path = Path(dataset_dir)
    if dataset_path.exists() and save_raw_data:
        # empty the directory
        shutil.rmtree(dataset_dir)
    # (re)create the directory
    dataset_path.mkdir(parents=True, exist_ok=True)

    print("Dataset will be saved in:", dataset_path.resolve())

    # save the metadata
    if metadata is None:
        metadata = {}
    metadata.update(
        dict(
            dt=dt,
            ts=ts,
            solver_class=type(solver),
            sim_dt=sim_dt,
            x0_min=x0_min,
            x0_max=x0_max,
        )
    )
    if system_params is not None:
        metadata["system_params"] = system_params
    # save the metadata in the `dataset_dir`
    with open(str(dataset_path / "metadata.pkl"), "wb") as f:
        dill.dump(metadata, f)

    print("Generating dataset...")

    sample_idx = 0
    with alive_bar(num_simulations) as bar:
        for sim_idx in range(num_simulations):
            rng, rng_x0_sampling, rng_tau_sampling = random.split(rng, num=3)

            # dimension of the state space
            n_x = x0_max.shape[0]
            # dimension of configuration space
            n_q = n_x // 2

            # generate initial state of the simulation
            if x0_sampling_dist == "uniform":
                x0 = random.uniform(
                    rng_x0_sampling,
                    x0_min.shape,
                    minval=x0_min,
                    maxval=x0_max,
                )
            elif x0_sampling_dist == "arcsine":
                u = random.uniform(rng_x0_sampling, x0_min.shape)
                x0 = x0_min + (x0_max - x0_min) * jnp.sin(jnp.pi * u / 2) ** 2
            elif x0_sampling_dist == "half-normal":
                u = random.normal(rng_x0_sampling, x0_min.shape)
                stdev = (x0_max - x0_min) / 2
                condlist = [u < 0, u >= 0]
                choicelist = [x0_min, x0_max]
                x0 = jnp.select(condlist, choicelist) - u * stdev
                # just to make sure that a very unlikely sample does not bring us out of bounds
                x0 = jnp.clip(x0, x0_min, x0_max)
            else:
                raise ValueError(f"Unknown sampling distribution: {x0_sampling_dist}")

            # sample the external torques or set them to zero
            if tau_max is None:
                tau = jnp.zeros((n_q,))
            else:
                tau = random.uniform(
                    rng_tau_sampling, (n_q,), minval=-tau_max, maxval=tau_max
                )

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
            labels = dict(t_ts=ts, x_ts=x_ts, tau=tau)

            if save_raw_data:
                # folder to save the simulation data
                sim_dir = dataset_path / f"sim-{sim_idx}"
                sim_dir.mkdir(parents=True, exist_ok=True)

                # save the labels in dataset_dir
                jnp.savez(file=str(sim_dir / "labels.npz"), **labels)

            rendering_ts = []
            for time_idx in range(x_ts.shape[0]):
                # configuration for current time step
                q = x_ts[time_idx, :n_q]

                # render the image
                img = rendering_fn(q)
                rendering_ts.append(img)

                if save_raw_data:
                    # save the image
                    cv2.imwrite(
                        str(sim_dir / f"rendering_time_idx-{time_idx}.jpeg"), img
                    )

                # update sample index
                sample_idx += 1

            # merge labels with image and id
            sample = labels | {
                "id": sim_idx,
                "rendering_ts": rendering_ts,
            }

            yield sim_idx, sample

            # update progress bar
            bar()
