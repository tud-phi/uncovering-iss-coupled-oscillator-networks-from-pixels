from diffrax import Euler
import dill
from dm_hamiltonian_dynamics_suite import datasets
import jax
import jax.numpy as jnp
import numpy as onp
from pathlib import Path

from dm_hamiltonian_dynamics_suite.datasets import (
    MASS_SPRING_FRICTION,
    PENDULUM_FRICTION,
    DOUBLE_PENDULUM_FRICTION,
)

# ["toy_physics/mass_spring", "toy_physics/mass_spring_friction", "toy_physics/mass_spring_colors",
# "toy_physics/mass_spring_colors_friction", "toy_physics/mass_spring_long_trajectory",
# "toy_physics/mass_spring_colors_long_trajectory", "toy_physics/pendulum", "toy_physics/pendulum_friction",
# "toy_physics/pendulum_colors", "toy_physics/pendulum_colors_friction", "toy_physics/pendulum_long_trajectory",
# "toy_physics/pendulum_colors_long_trajectory", "toy_physics/double_pendulum", "toy_physics/double_pendulum_friction",
# "toy_physics/double_pendulum_colors",
# "toy_physics/double_pendulum_colors_friction", "toy_physics/two_body", "toy_physics/two_body_colors",
# "multi_agent/matching_pennies", "multi_agent/matching_pennies_long_trajectory", "multi_agent/rock_paper_scissors",
# "multi_agent/rock_paper_scissors_long_trajectory", "mujoco_room/circle", "mujoco_room/spiral"]
dataset_name = "mass_spring_friction"

num_train = 5000
num_test = 2000

dt = 0.05
num_steps = 60
steps_per_dt = 10
data_dir = Path("data/tensorflow_datasets/toy_physics").resolve()
dataset_dir = data_dir / f"{dataset_name}_dt_{str(dt).replace('.', '_')}"


def main():
    match dataset_name:
        case "mass_spring_friction":
            state_dim = 2
        case "pendulum_friction":
            state_dim = 2
        case "double_pendulum_friction":
            state_dim = 4
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

    system, params_fn = globals()[dataset_name.upper()]
    params = params_fn()

    datasets.generate_full_dataset(
        folder=str(data_dir),
        dataset=dataset_name,
        dt=dt,
        num_steps=num_steps,
        steps_per_dt=steps_per_dt,
        num_train=num_train,
        num_test=num_test,
        overwrite=True,
    )

    # write metadata
    ts = jnp.linspace(0, num_steps * dt, num=num_steps + 1)
    rmax = params["radius_range"].max
    x0_min, x0_max = -rmax * jnp.ones((state_dim,)), rmax * jnp.ones((state_dim,))
    dataset_metadata = dict(
        ts=ts,
        dt=dt,
        sim_dt=dt / steps_per_dt,
        solver_class=Euler.__name__,
        x0_min=x0_min,
        x0_max=x0_max,
        rendering=dict(
            width=32,
            height=32,
        ),
        split_sizes=dict(
            train=num_train,
            test=num_test,
        ),
    )

    with open(str(dataset_dir / "metadata.pkl"), "wb") as f:
        dill.dump(dataset_metadata, f)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
