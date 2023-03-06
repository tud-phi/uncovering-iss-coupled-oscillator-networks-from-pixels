from diffrax import Dopri5
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import random
import jax.numpy as jnp
from jsrm.integration import ode_factory
from jsrm.systems import euler_lagrangian, pendulum
from pathlib import Path

from src.dataset_collection import collect_dataset
from src.rendering import render_pendulum

num_links = 1

num_links_to_sym_exp_filepath_map = {
    1: "single_pendulum.dill",
    2: "double_pendulum.dill",
    3: "triple_pendulum.dill",
}

sym_exp_filepath = (
    Path("symbolic_expressions") / num_links_to_sym_exp_filepath_map[num_links]
)

robot_params = {
    "m": jnp.array([10.0]),
    "I": jnp.array([3.0]),
    "l": jnp.array([2.0]),
    "lc": jnp.array([1.0]),
    "g": jnp.array([0.0, -9.81]),
}

num_simulations = 20000  # number of simulations to run
dt = 1e-2  # time step used for simulation [s]
horizon = 1e-1  # duration of each simulation [s]
# maximum magnitude of the initial joint velocity [rad/s]
max_q_d_0 = 30 * jnp.ones((num_links,))

dataset_dir = Path("data") / "raw_datasets" / "single_pendulum"

# Pseudo random number generator
rng = random.PRNGKey(seed=0)

if __name__ == "__main__":
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    rendering_fn = partial(
        render_pendulum,
        forward_kinematics_fn,
        robot_params,
        width=32,
        height=32,
    )

    # set initial conditions
    state_init_min = jnp.zeros((2 * num_links,))
    state_init_max = jnp.zeros((2 * num_links,))
    state_init_min = state_init_min.at[:num_links].set(-jnp.pi)
    state_init_max = state_init_max.at[:num_links].set(jnp.pi)
    state_init_min = state_init_min.at[num_links:].set(-max_q_d_0)
    state_init_max = state_init_max.at[num_links:].set(max_q_d_0)

    # assume an autonomous system
    tau = jnp.zeros((num_links,))

    collect_dataset(
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau),
        rendering_fn=rendering_fn,
        rng=rng,
        num_simulations=num_simulations,
        horizon=jnp.array(horizon),
        dt=jnp.array(dt),
        state_init_min=state_init_min,
        state_init_max=state_init_max,
        dataset_dir=str(dataset_dir),
        solver=Dopri5(),
    )
