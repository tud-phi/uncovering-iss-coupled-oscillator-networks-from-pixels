import flax.linen as nn

from functools import partial
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
from jax import Array, debug, grad, jit, lax, random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
import tensorflow as tf
from typing import Dict, Tuple, Union
import warnings

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

from src.models.autoencoders import Autoencoder, VAE
from src.models.neural_odes import (
    ConOde,
    ConIaeOde,
    MlpOde,
)
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.rendering import preprocess_rendering, render_planar_pcs
from src.rollout import rollout_ode, rollout_ode_with_latent_space_control
from src.training.dataset_utils import load_dataset, load_dummy_neural_network_input
from src.tasks import dynamics_autoencoder
from src.training.train_state_utils import restore_train_state
from src.visualization.img_animation import (
    animate_image_cv2,
    animate_pred_vs_target_image_pyplot,
)
from src.visualization.utils import denormalize_img

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

system_type = "pcc_ns-2"
long_horizon_dataset = True
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
dynamics_model_name = "node-con-iae"  # "node-con-iae", "node-con-iae-s", "node-mechanical-mlp"
# latent space shape
n_z = 2
# number of configuration space dimensions
n_q = 2
# whether to use real or learned dynamics
simulate_with_learned_dynamics = False

# simulation settings
# num setpoints
num_setpoints = 7
sim_duration_per_setpoint = 5.0  # s
sim_duration = num_setpoints * sim_duration_per_setpoint  # s
# initial configuration
q0 = jnp.pi * jnp.array([0.0, 0.0])
# specify desired configuration
# q_des = jnp.array([jnp.pi, 1.25 * jnp.pi])
q_des = jnp.pi * jnp.array([-1.0, 0.5])
# control settings
apply_feedforward_term = True
apply_feedback_term = True
# gains
match dynamics_model_name:
    case "node-con-iae" | "node-con-iae-s":
        if simulate_with_learned_dynamics:
            kp, ki, kd = 1e0, 1e0, 0e0
            psatid_gamma = 1.0
        else:
            kp, ki, kd = 1.0e0, 3e0, 0e0
            # kp, ki, kd = 0.1e0, 3e0, 0e0
            psatid_gamma = 1.0
    case "node-mechanical-mlp":
        kp, ki, kd = 1e-2, 2e-2, 5e-5
        psatid_gamma = 1.0
    case _:
        raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")

batch_size = 10
norm_layer = nn.LayerNorm
diag_shift, diag_eps = 1e-6, 2e-6
match dynamics_model_name:
    case "node-con-iae":
        experiment_id = f"2024-05-20_13-14-46/n_z_{n_z}_seed_{seed}"
        num_mlp_layers, mlp_hidden_dim = 5, 30
    case "node-con-iae-s":
        experiment_id = f"2024-03-17_22-26-44/n_z_{n_z}_seed_{seed}"
        num_mlp_layers, mlp_hidden_dim = 2, 12
    case "node-mechanical-mlp":
        experiment_id = f"2024-05-21_07-45-14/n_z_{n_z}_seed_{seed}"
        num_mlp_layers, mlp_hidden_dim = 5, 30
    case _:
        raise ValueError(
            f"No experiment_id for dynamics_model_name={dynamics_model_name}"
        )

# identify the number of segments
if system_type == "cc":
    num_segments = 1
elif system_type.split("_")[0] == "pcc":
    num_segments = int(system_type.split("-")[-1])
else:
    raise ValueError(f"Unknown system_type: {system_type}")
print(f"Number of segments: {num_segments}")

# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in ["node", "discrete"], f"Unknown dynamics_type: {dynamics_type}"

sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_pcs_ns-{num_segments}.dill"
)
ckpt_dir = (
    Path("logs").resolve() / f"{system_type}_dynamics_autoencoder" / experiment_id
)

# plotting setttings
figsize = (6, 4.5)
plt_colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = []
for k in range(n_z // len(plt_colors_cycle) + 1):
    colors += plt_colors_cycle


if __name__ == "__main__":
    # generate a random setpoint sequence
    rng_setpoint = random.PRNGKey(seed=1)
    q_des_ps = 5.0 * jnp.pi * random.uniform(rng_setpoint, shape=(num_setpoints, n_q), minval=-1.0, maxval=1.0)

    dataset_name = f"planar_pcs/{system_type}_32x32px_h-101"
    datasets, dataset_info, dataset_metadata = load_dataset(
        dataset_name,
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # extract the robot parameters from the dataset
    robot_params = dataset_metadata["system_params"]
    print(f"Robot parameters: {robot_params}")
    # dimension of the configuration space
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    # size of torques
    n_tau = train_ds.element_spec["tau"].shape[-1]  # dimension of the control input
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape
    # limits of the configuration space
    q0_min, q0_max = dataset_metadata["x0_min"][:n_q], dataset_metadata["x0_max"][:n_q]

    # get the dynamics function
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = planar_pcs.factory(
        sym_exp_filepath, strain_selector=dataset_metadata["strain_selector"]
    )
    ode_fn = ode_with_forcing_factory(dynamical_matrices_fn, robot_params)

    # initialize the rendering function
    rendering_fn = partial(
        render_planar_pcs,
        forward_kinematics_fn,
        robot_params,
        width=img_shape[0],
        height=img_shape[0],
        origin_uv=dataset_metadata["rendering"]["origin_uv"],
        line_thickness=dataset_metadata["rendering"]["line_thickness"],
    )

    # initialize the neural networks
    if ae_type == "beta_vae":
        autoencoder_model = VAE(
            latent_dim=n_z, img_shape=img_shape, norm_layer=norm_layer
        )
    else:
        autoencoder_model = Autoencoder(
            latent_dim=n_z, img_shape=img_shape, norm_layer=norm_layer
        )
    if dynamics_model_name in ["node-con-iae", "node-con-iae-s"]:
        dynamics_model = ConIaeOde(
            latent_dim=n_z,
            input_dim=n_tau,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            apply_feedforward_term=apply_feedforward_term,
            apply_feedback_term=apply_feedback_term,
        )
    elif dynamics_model_name in ["node-mechanical-mlp", "node-mechanical-mlp-s"]:
        dynamics_model = MlpOde(
            latent_dim=n_z,
            input_dim=n_tau,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            mechanical_system=True,
        )
    else:
        raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")
    nn_model = DynamicsAutoencoder(
        autoencoder=autoencoder_model,
        dynamics=dynamics_model,
        dynamics_type=dynamics_type,
    )

    # import solver class from diffrax
    # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    solver_class = getattr(
        __import__("diffrax", fromlist=[dataset_metadata["solver_class"]]),
        dataset_metadata["solver_class"],
    )

    # define settings for the closed-loop simulation
    control_dt = 1e-2  # control and time step of 1e-2 s
    sim_dt = 1e-3 * control_dt  # simulation time step of 1e-5 s
    ts = jnp.linspace(0.0, sim_duration, num=int(sim_duration / control_dt))
    ode_rollout_fn = partial(
        rollout_ode,
        ode_fn=ode_fn,
        ts=ts,
        sim_dt=sim_dt,
        rendering_fn=rendering_fn,
        solver=solver_class(),
        show_progress=True,
    )
    # define the task callables for the rollout
    (
        task_callables_rollout_learned,
        metrics_collection_cls,
    ) = dynamics_autoencoder.task_factory(
        system_type,
        nn_model,
        ts=ts,
        sim_dt=sim_dt,
        ae_type=ae_type,
        dynamics_type=dynamics_type,
        solver=solver_class(),
        latent_velocity_source="image-space-finite-differences",
    )
    # load the neural network dummy input
    nn_dummy_input = load_dummy_neural_network_input(
        test_ds, task_callables_rollout_learned
    )
    # load the training state from the checkpoint directory
    state = restore_train_state(
        rng=rng,
        ckpt_dir=ckpt_dir,
        nn_model=nn_model,
        nn_dummy_input=nn_dummy_input,
        metrics_collection_cls=metrics_collection_cls,
        init_fn=nn_model.forward_all_layers,
    )
    nn_model_bound = nn_model.bind({"params": state.params})
    dynamics_model_bound = dynamics_model.bind({"params": state.params["dynamics"]})
    energy_fn = getattr(dynamics_model_bound, "energy_fn", None)
    potential_energy_fn: callable = getattr(dynamics_model_bound, "potential_energy_fn", None)
    kinetic_energy_fn: callable = getattr(dynamics_model_bound, "kinetic_energy_fn", None)

    def encode_fn(img: Array) -> Array:
        return partial(
            nn_model.apply,
            {"params": state.params},
            method=nn_model.encode,
        )(img[None, ...])[0, ...]

    def decode_fn(z: Array) -> Array:
        return partial(
            nn_model.apply,
            {"params": state.params},
            method=nn_model.decode,
        )(z[None, ...])[0, ...]
    
    # get an estimate of the maximum latent
    img_q0_max = rendering_fn(q0_max)
    img_q0_max = jnp.array(
        preprocess_rendering(img_q0_max, grayscale=True, normalize=True)
    )
    z0_max = nn_model_bound.encode(img_q0_max[None, ...])[0, ...]
    # create grid for plotting the potential energy landscape
    z1_range = jnp.linspace(-z0_max[0], z0_max[0], 100)
    z2_range = jnp.linspace(-z0_max[1], z0_max[1], 100)
    z1_grid, z2_grid = jnp.meshgrid(z1_range, z2_range)
    z_grid = jnp.stack([z1_grid, z2_grid], axis=-1)
    
    if n_z == 2:
        match dynamics_model_name:
            case "node-con" | "node-w-con":
                # plot the potential energy landscape in the original latent space
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=figsize,
                    num="Potential energy landscape in z-coordinates",
                )
                xi_grid = jnp.concatenate([z_grid, jnp.zeros_like(z_grid)], axis=-1)
                U_grid = jax.vmap(
                    partial(potential_energy_fn, coordinate="z"),
                )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
                tau_pot_grid = -jax.vmap(
                    grad(partial(potential_energy_fn, coordinate="z")),
                )(xi_grid.reshape(-1, xi_grid.shape[-1]))[..., :n_z].reshape(
                    *xi_grid.shape[:2], -1
                )
                # contour plot of the potential energy
                cs = ax.contourf(z1_grid, z2_grid, U_grid, levels=100)
                # quiver plot of the potential energy gradient
                qv_skip = 10
                ax.quiver(
                    z1_grid[::qv_skip, ::qv_skip],
                    z2_grid[::qv_skip, ::qv_skip],
                    tau_pot_grid[::qv_skip, ::qv_skip, 0],
                    tau_pot_grid[::qv_skip, ::qv_skip, 1],
                    angles="xy",
                    scale=None,
                    scale_units="xy",
                    color="white",
                )
                plt.colorbar(cs)
                ax.set_xlabel(r"$z_1$")
                ax.set_ylabel(r"$z_2$")
                ax.set_title("Potential energy landscape of learned latent dynamics")
                plt.grid(True)
                plt.box(True)
                plt.savefig(ckpt_dir / "potential_energy_landscape_z.pdf")
                plt.show()

                # plot the potential energy in the w-coordinates
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=figsize,
                    num="Potential energy landscape in zw-coordinates",
                )
                zw0_max = dynamics_model_bound.W @ z0_max
                zw1_range = jnp.linspace(-zw0_max[0], zw0_max[0], 100)
                zw2_range = jnp.linspace(-zw0_max[1], zw0_max[1], 100)
                zw1_grid, zw2_grid = jnp.meshgrid(zw1_range, zw2_range)
                zw_grid = jnp.stack([zw1_grid, zw2_grid], axis=-1)
                xi_grid = jnp.concatenate([zw_grid, jnp.zeros_like(zw_grid)], axis=-1)
                U_grid = jax.vmap(
                    partial(potential_energy_fn, coordinate="zw"),
                )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
                tau_pot_grid = -jax.vmap(
                    grad(partial(potential_energy_fn, coordinate="zw")),
                )(xi_grid.reshape(-1, xi_grid.shape[-1]))[..., :n_z].reshape(
                    *xi_grid.shape[:2], -1
                )
                # contour plot of the potential energy
                cs = ax.contourf(zw1_grid, zw2_grid, U_grid, levels=100)
                # quiver plot of the potential energy gradient
                qv_skip = 10
                ax.quiver(
                    zw1_grid[::qv_skip, ::qv_skip],
                    zw2_grid[::qv_skip, ::qv_skip],
                    tau_pot_grid[::qv_skip, ::qv_skip, 0],
                    tau_pot_grid[::qv_skip, ::qv_skip, 1],
                    angles="xy",
                    scale=None,
                    scale_units="xy",
                    color="white",
                )
                plt.colorbar(cs)
                ax.set_xlabel(r"$z_{w,1}$")
                ax.set_ylabel(r"$z_{w,2}$")
                ax.set_title(
                    "Potential energy landscape of learned latent dynamics in w-coordinates"
                )
                plt.grid(True)
                plt.box(True)
                plt.savefig(ckpt_dir / "potential_energy_landscape_zw.pdf")
                plt.show()

                # plot the potential energy in the collocated coordinates
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=figsize,
                    num="Potential energy landscape in collocated coordinates",
                )
                zeta1_range = jnp.linspace(-1.0, 1.0, 100)
                zeta2_range = jnp.linspace(-1.0, 1.0, 100)
                zeta1_grid, zeta2_grid = jnp.meshgrid(zeta1_range, zeta2_range)
                zeta_grid = jnp.stack([zeta1_grid, zeta2_grid], axis=-1)
                xi_grid = jnp.concatenate(
                    [zeta_grid, jnp.zeros_like(zeta_grid)], axis=-1
                )
                U_grid = jax.vmap(
                    partial(potential_energy_fn, coordinate="zeta"),
                )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
                tau_pot_grid = -jax.vmap(
                    grad(partial(potential_energy_fn, coordinate="zeta")),
                )(xi_grid.reshape(-1, xi_grid.shape[-1]))[..., :n_z].reshape(
                    *xi_grid.shape[:2], -1
                )
                # contour plot of the potential energy
                cs = ax.contourf(zeta1_grid, zeta2_grid, U_grid, levels=100)
                # quiver plot of the potential energy gradient
                qv_skip = 10
                ax.quiver(
                    zeta1_grid[::qv_skip, ::qv_skip],
                    zeta2_grid[::qv_skip, ::qv_skip],
                    tau_pot_grid[::qv_skip, ::qv_skip, 0],
                    tau_pot_grid[::qv_skip, ::qv_skip, 1],
                    angles="xy",
                    scale=None,
                    scale_units="xy",
                    color="white",
                )
                plt.colorbar(cs)
                ax.set_xlabel(r"$\zeta_1$")
                ax.set_ylabel(r"$\zeta_2$")
                ax.set_title(
                    "Potential energy landscape of learned latent dynamics in collocated coordinates"
                )
                plt.grid(True)
                plt.box(True)
                plt.savefig(ckpt_dir / "potential_energy_landscape_zeta.pdf")
                plt.show()
            case "node-con-iae" | "node-con-iae-s":
                # plot the potential energy landscape in the original latent space
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=figsize,
                    num="Potential energy landscape in z-coordinates",
                )
                xi_grid = jnp.concatenate([z_grid, jnp.zeros_like(z_grid)], axis=-1)
                U_grid = jax.vmap(potential_energy_fn)(
                    xi_grid.reshape(-1, xi_grid.shape[-1])
                ).reshape(xi_grid.shape[:2])
                tau_pot_grid = -jax.vmap(grad(potential_energy_fn))(
                    xi_grid.reshape(-1, xi_grid.shape[-1])
                )[..., :n_z].reshape(*xi_grid.shape[:2], -1)
                # contour plot of the potential energy
                cs = ax.contourf(z1_grid, z2_grid, U_grid, levels=100)
                # quiver plot of the potential energy gradient
                qv_skip = 10
                ax.quiver(
                    z1_grid[::qv_skip, ::qv_skip],
                    z2_grid[::qv_skip, ::qv_skip],
                    tau_pot_grid[::qv_skip, ::qv_skip, 0],
                    tau_pot_grid[::qv_skip, ::qv_skip, 1],
                    angles="xy",
                    scale=None,
                    scale_units="xy",
                    color="white",
                )
                plt.colorbar(cs, label=r"$\mathcal{U}$")
                ax.set_xlabel(r"$z_1$")
                ax.set_ylabel(r"$z_2$")
                # ax.set_title("Potential energy landscape of learned latent dynamics")
                plt.grid(True)
                plt.box(True)
                plt.savefig(ckpt_dir / "potential_energy_landscape_z.pdf")
                plt.show()

    if callable(potential_energy_fn) and n_q == 2:
        # plot the learned potential energy landscape in the configuration space
        q1_range = jnp.linspace(q0_min[0], q0_max[0], 25)
        q2_range = jnp.linspace(q0_min[1], q0_max[1], 25)
        q1_grid, q2_grid = jnp.meshgrid(q1_range, q2_range)
        q_grid = jnp.stack([q1_grid, q2_grid], axis=-1)
        U_grid = jnp.zeros(q_grid.shape[:2])
        tau_pot_grid = jnp.zeros(q_grid.shape[:2] + (n_tau,))

        match dynamics_model_name:
            case "node-con" | "node-w-con":
                terms = dynamics_model_bound.get_terms(coordinate="zeta")
                for i in range(q_grid.shape[0]):
                    for j in range(q_grid.shape[1]):
                        q = q_grid[i, j]
                        img = rendering_fn(q)
                        img = jnp.array(
                            preprocess_rendering(img, grayscale=True, normalize=True)
                        )
                        z = nn_model_bound.encode(img[None, ...])[0, ...]
                        zeta = terms["J_h"] @ terms["J_w"] @ z
                        xi = jnp.concatenate([zeta, jnp.zeros((n_z,))])
                        U = potential_energy_fn(xi, coordinate="zeta")
                        tau_pot = -grad(
                            partial(potential_energy_fn, coordinate="zeta")
                        )(xi)[..., :n_tau]
                        U_grid = U_grid.at[i, j].set(U)
                        tau_pot_grid = tau_pot_grid.at[i, j, :].set(tau_pot)
            case "node-con-iae" | "node-con-iae-s":
                for i in range(q_grid.shape[0]):
                    for j in range(q_grid.shape[1]):
                        q = q_grid[i, j]
                        img = rendering_fn(q)
                        img = jnp.array(
                            preprocess_rendering(img, grayscale=True, normalize=True)
                        )
                        z = nn_model_bound.encode(img[None, ...])[0, ...]
                        xi = jnp.concatenate([z, jnp.zeros((n_z,))])
                        U = potential_energy_fn(xi)
                        tau_pot = -grad(potential_energy_fn)(xi)[..., :n_tau]
                        U_grid = U_grid.at[i, j].set(U)
                        tau_pot_grid = tau_pot_grid.at[i, j, :].set(tau_pot)
            case _:
                raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")

        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="Learned potential energy landscape in configuration space",
        )
        # contour plot of the potential energy
        cs = ax.contourf(q1_grid, q2_grid, U_grid, levels=100)
        plt.colorbar(cs, ax=ax, label=r"$\mathcal{U}$")
        ax.set_xlabel(r"$q_1$ [rad/m]")
        ax.set_ylabel(r"$q_2$ [rad/m]")
        # axes[0].set_title("Learned potential energy in $q$-space")
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "potential_energy_landscape_q.pdf")
        plt.show()

        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="Learned potential energy gradient in configuration space",
        )
        # quiver plot of the potential energy gradient
        qv_skip = 3
        qs = ax.quiver(
            q1_grid[::qv_skip, ::qv_skip],
            q2_grid[::qv_skip, ::qv_skip],
            tau_pot_grid[::qv_skip, ::qv_skip, 0],
            tau_pot_grid[::qv_skip, ::qv_skip, 1],
            jnp.hypot(
                tau_pot_grid[::qv_skip, ::qv_skip, 0],
                tau_pot_grid[::qv_skip, ::qv_skip, 1],
            ),
            angles="xy",
            scale=None,
            scale_units="xy",
        )
        qk = ax.quiverkey(
            qs, 0.9, 0.9, 1, r"$\tau: 1$ Nm", labelpos="E", coordinates="figure"
        )
        plt.colorbar(qs, ax=ax, label=r"$\tau$ [Nm]")
        ax.set_xlabel(r"$q_1$ [rad/m]")
        ax.set_ylabel(r"$q_2$ [rad/m]")
        ax.set_title("Learned potential force in $q$-space")
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "potential_energy_gradient_q.pdf")
        plt.show()

        # compute the ground-truth potential energy landscape in the configuration space
        U_grid = jnp.zeros(q_grid.shape[:2])
        tau_pot_grid = jnp.zeros(q_grid.shape[:2] + (n_tau,))
        robot_potential_energy_fn = jit(partial(auxiliary_fns["potential_energy_fn"], robot_params))
        for i in range(q_grid.shape[0]):
            for j in range(q_grid.shape[1]):
                q = q_grid[i, j]
                U = robot_potential_energy_fn(q)
                tau_pot = -grad(robot_potential_energy_fn)(q)[..., :n_tau]
                U_grid = U_grid.at[i, j].set(U)
                tau_pot_grid = tau_pot_grid.at[i, j, :].set(tau_pot)
        # plot the ground-truth potential energy landscape in the configuration space
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="Ground-truth potential energy landscape in configuration space",
        )
        # contour plot of the potential energy
        cs = ax.contourf(q1_grid, q2_grid, U_grid, levels=100)
        # quiver plot of the potential energy gradient
        qv_skip = 3
        ax.quiver(
            q1_grid[::qv_skip, ::qv_skip],
            q2_grid[::qv_skip, ::qv_skip],
            tau_pot_grid[::qv_skip, ::qv_skip, 0],
            tau_pot_grid[::qv_skip, ::qv_skip, 1],
            angles="xy",
            scale=None,
            scale_units="xy",
            color="white",
        )
        plt.colorbar(cs, ax=ax, label=r"$\mathcal{U}$")
        ax.set_xlabel(r"$q_1$ [rad/m]")
        ax.set_ylabel(r"$q_2$ [rad/m]")
        # axes[0].set_title("Ground-truth potential energy in $q$-space")
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "potential_energy_landscape_q_gt.pdf")
        plt.show()

    def control_fn(
        t: Array, x: Array, control_state: Dict[str, Array]
    ) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
        """
        Control function for the setpoint regulation.
        Args:
            t: current time
            x: current state of the system
            control_state: dictionary with the controller's stateful information. Contains entry with key "e_int" for the integral error.
        Returns:
            tau: control input
            control_state: dictionary with the controller's stateful information. Contains entry with key "e_int" for the integral error.
            control_info: dictionary with control information
        """
        # select the desired setpoint
        _time_idx = (t / control_dt).astype(int)
        _z_des = lax.dynamic_slice(z_des_ts, (_time_idx, 0), (1, n_z)).squeeze(0)

        tau, control_state, control_info = dynamics_model.apply(
            {"params": state.params["dynamics"]},
            x,
            control_state,
            method=dynamics_model.setpoint_regulation_fn,
            dt=control_dt,
            z_des=_z_des,
            kp=kp,
            ki=ki,
            kd=kd,
            gamma=psatid_gamma,
        )
        """
        tau, control_state, control_info = dynamics_model_bound.setpoint_regulation_fn(
            x,
            control_state,
            dt=control_dt,
            z_des=_z_des,
            kp=kp,
            ki=ki,
            kd=kd,
            gamma=psatid_gamma,
        )
        """

        return tau, control_state, control_info

    # render and encode all the target images
    img_des_ps = jnp.zeros((num_setpoints, *img_shape))
    z_des_ps = jnp.zeros((num_setpoints, n_z))
    for setpoint_idx in range(num_setpoints):
        q_des = q_des_ps[setpoint_idx, :]
        # render target image
        img_des = rendering_fn(q_des)
        # normalize the target image
        img_des = jnp.array(
            preprocess_rendering(img_des, grayscale=True, normalize=True)
        )
        # encode the target image
        z_des = nn_model_bound.encode(img_des[None, ...])[0, ...]

        img_des_ps = img_des_ps.at[setpoint_idx].set(img_des)
        z_des_ps = z_des_ps.at[setpoint_idx].set(z_des)

    # generate time sequence of setpoints
    q_des_ts = q_des_ps.repeat(int(sim_duration_per_setpoint / control_dt), axis=0)
    img_des_ts = img_des_ps.repeat(int(sim_duration_per_setpoint / control_dt), axis=0)
    z_des_ts = z_des_ps.repeat(int(sim_duration_per_setpoint / control_dt), axis=0)

    # set initial condition for closed-loop simulation
    x0 = jnp.concatenate([q0, jnp.zeros((n_q,))])

    if simulate_with_learned_dynamics is True:
        # render the initial condition
        img0 = rendering_fn(q0)
        # normalize the initial image
        img0 = jnp.array(preprocess_rendering(img0, grayscale=True, normalize=True))
        # encode the initial condition
        z0 = nn_model_bound.encode(img0[None, ...])[0, ...]

        def learned_ode_fn(t: Array, x: Array, tau: Array) -> Array:
            """
            Learned ODE function for the closed-loop simulation.
            Args:
                t: current time
                x: current state of the system
                tau: control input
            Returns:
                x_d: time derivative of the state
            """
            return dynamics_model_bound(x, tau)

        # start closed-loop simulation of learned dynamics with control
        print("Simulating learned closed-loop dynamics...")
        xi0 = jnp.concatenate([z0, jnp.zeros((n_z,))])
        sim_ts = rollout_ode(
            ode_fn=learned_ode_fn,
            rendering_fn=jit(decode_fn),
            ts=ts,
            sim_dt=sim_dt,
            x0=xi0,
            control_fn=jit(control_fn),
            control_state_init={"e_int": jnp.zeros((n_z,))},
            grayscale_rendering=False,
            normalize_rendering=False,
        )
        xi_ts = sim_ts["x_ts"]
    else:
        # start closed-loop simulation of real dynamics with latent space control
        print("Simulating real closed-loop dynamics...")
        sim_ts = rollout_ode_with_latent_space_control(
            ode_fn=ode_fn,
            rendering_fn=rendering_fn,
            encode_fn=jit(encode_fn),
            ts=ts,
            sim_dt=sim_dt,
            x0=x0,
            input_dim=n_tau,
            latent_dim=n_z,
            control_fn=jit(control_fn),
            control_state_init={"e_int": jnp.zeros((n_z,))},
        )
        xi_ts = sim_ts["xi_ts"]

    # extract both the ground-truth and the statically predicted images
    img_ts = sim_ts["rendering_ts"]
    # add the desired setpoints to the sim_ts dictionary
    sim_ts["q_des_ts"] = q_des_ts
    sim_ts["img_des_ts"] = img_des_ts
    sim_ts["z_des_ts"] = z_des_ts

    if callable(energy_fn):
        if type(dynamics_model) is ConOde:
            energy_fn = partial(
                energy_fn,
                coordinate="zw" if dynamics_model_bound.use_w_coordinates else "z",
            )
            potential_energy_fn = partial(
                potential_energy_fn,
                coordinate="zw" if dynamics_model_bound.use_w_coordinates else "z",
            )
            kinetic_energy_fn = partial(
                kinetic_energy_fn,
                coordinate="zw" if dynamics_model_bound.use_w_coordinates else "z",
            )

        sim_ts["V_ts"] = jax.vmap(energy_fn)(xi_ts)  # total energy
        sim_ts["T_ts"] = jax.vmap(kinetic_energy_fn)(xi_ts)  # kinetic energy
        sim_ts["U_ts"] = jax.vmap(potential_energy_fn)(xi_ts)  # potential energy
        sim_ts["U_des_ts"] = jax.vmap(potential_energy_fn)(z_des_ts)  # desired potential energy

    # save the simulation results
    onp.savez(ckpt_dir / "setpoint_sequence_controlled_rollout.npz", **sim_ts)

    # denormalize the images
    img_ts = jax.vmap(partial(denormalize_img, apply_threshold=True))(img_ts)
    img_des_ts = jax.vmap(partial(denormalize_img, apply_threshold=True))(
        img_des_ts
    )

    # animate the rollout
    print("Animate the rollout...")
    animate_pred_vs_target_image_pyplot(
        onp.array(ts),
        img_pred_ts=onp.array(img_ts),
        img_target_ts=onp.array(img_des_ts),
        filepath=ckpt_dir / "setpoint_sequence_controlled_rollout.mp4",
        step_skip=1,
        show=True,
        label_pred="Actual behavior",
        label_target="Desired behavior",
    )
    animate_image_cv2(
        onp.array(ts),
        onp.array(img_ts),
        filepath=ckpt_dir / "setpoint_sequence_controlled_rollout_actual.mp4",
        step_skip=1,
    )
    animate_image_cv2(
        onp.array(ts),
        onp.array(img_des_ts),
        filepath=ckpt_dir / "setpoint_sequence_controlled_rollout_desired.mp4",
        step_skip=1,
    )

    if simulate_with_learned_dynamics is False:
        # plot evolution of configuration space
        fig, ax = plt.subplots(1, 1, figsize=figsize, num="Configuration vs time")
        for i in range(n_q):
            ax.plot(ts, sim_ts["x_ts"][..., i], color=colors[i], label=f"$q_{i}$")
            ax.plot(
                ts,
                q_des_ts[..., i],
                linestyle="dashed",
                color=colors[i],
                label=rf"$q_{i}^d$",
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Configuration $q$")
        ax.legend()
        ax.set_title("Configuration vs. time")
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "setpoint_sequence_configuration_vs_time.pdf")
        plt.show()

    # plot evolution of latent state
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Latent vs time")
    for i in range(n_z):
        ax.plot(ts, xi_ts[..., i], color=colors[i], label=f"$z_{i}$")
        ax.plot(
            ts,
            z_des_ts[..., i],
            linestyle="dashed",
            color=colors[i],
            label=rf"$z_{i}^d$",
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Latent $z$")
    ax.set_title("Latent vs. time")
    ax.legend()
    plt.grid(True)
    plt.box(True)
    plt.savefig(ckpt_dir / "setpoint_sequence_latent_vs_time.pdf")
    plt.show()
    # plot the estimated latent velocity
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Latent velocity vs time")
    for i in range(n_z):
        ax.plot(ts, xi_ts[..., n_z + i], color=colors[i], label=rf"$\dot{{z}}_{i}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Latent velocity $\dot{z}$")
    ax.set_title("Latent velocity vs. time")
    ax.legend()
    plt.grid(True)
    plt.box(True)
    plt.savefig(ckpt_dir / "setpoint_sequence_latent_velocity_vs_time.pdf")
    plt.show()

    # plot the control inputs
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Control input vs time")
    ax.plot(ts, sim_ts["tau_ts"][..., 0], color=colors[0], label=r"$u_1$")
    ax.plot(ts, sim_ts["tau_ts"][..., 1], color=colors[1], label=r"$u_2$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Control input $u$")
    ax.set_title("Control input vs. time")
    ax.legend()
    plt.grid(True)
    plt.box(True)
    plt.savefig(ckpt_dir / "setpoint_sequence_control_input_vs_time.pdf")
    plt.show()
    # plot the feedforward and feedback torques
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Latent-space torques vs time")
    for i in range(n_z):
        ax.plot(
            ts,
            sim_ts["tau_z_ff_ts"][..., i],
            color=colors[i],
            label=r"$\tau_{z,ff," + str(i) + "}$",
        )
        ax.plot(
            ts,
            sim_ts["tau_z_fb_ts"][..., i],
            linestyle="dotted",
            color=colors[i],
            label=r"$\tau_{z,fb," + str(i) + "}$",
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torques")
    ax.set_title("Torques over time")
    ax.legend()
    plt.grid(True)
    plt.box(True)
    plt.savefig(ckpt_dir / "setpoint_sequence_ff_fb_torques_vs_time.pdf")
    plt.show()

    if "V_ts" in sim_ts:
        # plot the energy over time
        fig, ax = plt.subplots(1, 1, figsize=figsize, num="Energy vs time")
        ax.plot(ts, sim_ts["V_ts"], color=colors[0], label="Energy")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Energy")
        ax.set_title("Energy vs. time")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "setpoint_sequence_energy_vs_time.pdf")
        plt.show()
