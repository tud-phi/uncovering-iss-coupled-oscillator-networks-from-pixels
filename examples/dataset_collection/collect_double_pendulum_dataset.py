import cv2
from diffrax import Dopri5
from functools import partial
from jax import config as jax_config

jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'
jax_config.update("jax_enable_x64", True)  # double precision
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_factory
from jsrm.systems import euler_lagrangian, pendulum
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset_collection import collect_dataset
from src.rendering import render_pendulum


sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"pendulum_nl-2.dill"
)

robot_params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2]),
    "l": jnp.array([2.0, 1.0]),
    "lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
}

num_links = robot_params["l"].shape[0]

num_simulations = 20000  # number of simulations to run
dt = 1e-2  # time step used for simulation [s]
horizon_dim = 11  # number of samples in each trajectory
# maximum magnitude of the initial joint velocity [rad/s]
max_q_d_0 = 2 * jnp.pi * jnp.ones((num_links,))

dataset_dir = Path("data") / "raw_datasets" / "double_pendulum_64x64px"

# Pseudo random number generator
rng = random.PRNGKey(seed=0)

if __name__ == "__main__":
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    rendering_fn = partial(
        render_pendulum,
        forward_kinematics_fn,
        robot_params,
        width=64,
        height=64,
        line_thickness=2,
    )

    sample_q = jnp.array([36 / 180 * jnp.pi, -45 / 180 * jnp.pi])
    sample_img = rendering_fn(sample_q)
    plt.figure(num="Sample rendering")
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.title("q = " + str(sample_q / jnp.pi * 180) + " [deg]")
    plt.show()

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
        horizon_dim=horizon_dim,
        dt=jnp.array(dt),
        state_init_min=state_init_min,
        state_init_max=state_init_max,
        dataset_dir=str(dataset_dir),
        solver=Dopri5(),
        system_params=robot_params,
        do_yield=False,
        save_raw_data=True,
    )
