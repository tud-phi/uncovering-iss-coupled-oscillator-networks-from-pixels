import flax.linen as nn

from functools import partial
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
from jax import Array, grad, jit, random
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

from src.models.autoencoders import Autoencoder, VAE
from src.models.neural_odes import (
    ConOde,
    ConIaeOde,
    CornnOde,
    LnnOde,
    LinearStateSpaceOde,
    MambaOde,
    MlpOde,
)
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.rendering import preprocess_rendering, render_planar_pcs
from src.rollout import rollout_ode, rollout_ode_with_latent_space_control
from src.training.dataset_utils import load_dataset, load_dummy_neural_network_input
from src.tasks import dynamics_autoencoder
from src.training.train_state_utils import restore_train_state
from src.visualization.img_animation import (
    animate_pred_vs_target_image_pyplot,
)

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

system_type = "pcc_ns-2"
long_horizon_dataset = True
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
dynamics_model_name = "node-con-iae"
# latent space shape
n_z = 2
# number of configuration space dimensions
n_q = 2
# whether to use real or learned dynamics
simulate_with_learned_dynamics = False

# simulation settings
sim_duration = 10.0  # s
# initial configuration
q0 = jnp.pi * jnp.array([2.0, -2.0])
# specify desired configuration
# q_des = jnp.array([jnp.pi, 1.25 * jnp.pi])
q_des = jnp.pi * jnp.array([-1.0, 0.5])
# control settings
apply_feedforward_term = True
apply_feedback_term = True
use_collocated_form = False
# gains
match dynamics_model_name:
    case "node-con" | "node-w-con":
        if simulate_with_learned_dynamics:
            if n_z == 2:
                if apply_feedforward_term is False:
                    kp, ki, kd = 1e-3, 1.3e-1, 5e-3
                    psatid_gamma = 1.0
                else:
                    kp, ki, kd = 1e-3, 1e-3, 1e-3
                    psatid_gamma = 1.0
            else:
                kp, ki, kd = 0e0, 0e0, 0e-2
                psatid_gamma = 1.0
        else:
            if use_collocated_form:
                kp, ki, kd = 1e-3, 0e0, 0e0
                psatid_gamma = 1.0
            else:
                kp, ki, kd = 1e0, 2e0, 0e0
                psatid_gamma = 0.5
    case "node-con-iae" | "node-con-iae-s":
        if simulate_with_learned_dynamics:
            kp, ki, kd = 1e0, 1e0, 0e0
            psatid_gamma = 1.0
        else:
            kp, ki, kd = 1e0, 1e0, 0e0
            psatid_gamma = 1.0

batch_size = 10
norm_layer = nn.LayerNorm
cornn_gamma, cornn_epsilon = 1.0, 1.0
lnn_learn_dissipation = True
diag_shift, diag_eps = 1e-6, 2e-6
if long_horizon_dataset:
    match dynamics_model_name:
        case "node-mechanical-mlp":
            n_z = 8
            experiment_id = "2024-03-08_10-42-05"
            num_mlp_layers, mlp_hidden_dim = 5, 21
            mlp_nonlinearity_name = "tanh"
        case "node-w-con":
            experiment_id = f"2024-03-12_12-53-29/n_z_{n_z}_seed_{seed}"
        case "node-con-iae":
            experiment_id = f"2024-03-15_21-44-34/n_z_{n_z}_seed_{seed}"
            num_mlp_layers, mlp_hidden_dim = 5, 30
        case "node-con-iae-s":
            experiment_id = f"2024-03-17_22-26-44/n_z_{n_z}_seed_{seed}"
            num_mlp_layers, mlp_hidden_dim = 2, 12
        case _:
            raise ValueError(
                f"No experiment_id for dynamics_model_name={dynamics_model_name}"
            )
else:
    if ae_type == "wae":
        raise NotImplementedError
    elif ae_type == "beta_vae":
        if dynamics_model_name == "node-con":
            experiment_id = "2024-02-14_18-34-27"
        elif dynamics_model_name == "node-w-con":
            if n_z == 8:
                experiment_id = "2024-02-21_13-34-53"
            elif n_z == 2:
                experiment_id = "2024-02-22_14-11-21"
            else:
                raise ValueError(f"No experiment_id for n_z={n_z}")
            experiment_id = "2024-02-22_15-01-40"
        else:
            raise NotImplementedError(
                f"beta_vae with node_type '{dynamics_model_name}' not implemented yet."
            )
    else:
        raise NotImplementedError

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
figsize = (8, 6)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


if __name__ == "__main__":
    if long_horizon_dataset:
        dataset_name = f"planar_pcs/{system_type}_32x32px_h-101"
    else:
        dataset_name = f"planar_pcs/{system_type}_32x32px"
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
    if long_horizon_dataset is False:
        robot_params["D"] = 5 * robot_params["D"]
    warnings.warn(
        "The damping parameter D is scaled by 1e1 to improve the numerical stability."
    )
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
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.factory(
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
    if dynamics_model_name == "node-cornn":
        dynamics_model = CornnOde(
            latent_dim=n_z,
            input_dim=n_tau,
            gamma=cornn_gamma,
            epsilon=cornn_epsilon,
        )
    elif dynamics_model_name in ["node-con", "node-w-con"]:
        dynamics_model = ConOde(
            latent_dim=n_z,
            input_dim=n_tau,
            use_w_coordinates=dynamics_model_name == "node-w-con",
            apply_feedforward_term=apply_feedforward_term,
            apply_feedback_term=apply_feedback_term,
        )
    elif dynamics_model_name in ["node-con-iae", "node-con-iae-s"]:
        dynamics_model = ConIaeOde(
            latent_dim=n_z,
            input_dim=n_tau,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            apply_feedforward_term=apply_feedforward_term,
            apply_feedback_term=apply_feedback_term,
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

    if n_z == 2:
        match dynamics_model_name:
            case "node-con" | "node-w-con":
                # plot the potential energy landscape in the original latent space
                fig, ax = plt.subplots(
                    1, 1, figsize=figsize, num="Potential energy landscape in z-coordinates"
                )
                z1_range = jnp.linspace(-1.0, 1.0, 100)
                z2_range = jnp.linspace(-1.0, 1.0, 100)
                z1_grid, z2_grid = jnp.meshgrid(z1_range, z2_range)
                z_grid = jnp.stack([z1_grid, z2_grid], axis=-1)
                xi_grid = jnp.concatenate([z_grid, jnp.zeros_like(z_grid)], axis=-1)
                U_grid = jax.vmap(
                    partial(dynamics_model_bound.energy_fn, coordinate="z"),
                )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
                tau_pot_grid = -jax.vmap(
                    grad(partial(dynamics_model_bound.energy_fn, coordinate="z")),
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
                    1, 1, figsize=figsize, num="Potential energy landscape in zw-coordinates"
                )
                zw1_range = jnp.linspace(-1.0, 1.0, 100)
                zw2_range = jnp.linspace(-1.0, 1.0, 100)
                zw1_grid, zw2_grid = jnp.meshgrid(zw1_range, zw2_range)
                zw_grid = jnp.stack([zw1_grid, zw2_grid], axis=-1)
                xi_grid = jnp.concatenate([zw_grid, jnp.zeros_like(zw_grid)], axis=-1)
                U_grid = jax.vmap(
                    partial(dynamics_model_bound.energy_fn, coordinate="zw"),
                )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
                tau_pot_grid = -jax.vmap(
                    grad(partial(dynamics_model_bound.energy_fn, coordinate="zw")),
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
                xi_grid = jnp.concatenate([zeta_grid, jnp.zeros_like(zeta_grid)], axis=-1)
                U_grid = jax.vmap(
                    partial(dynamics_model_bound.energy_fn, coordinate="zeta"),
                )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
                tau_pot_grid = -jax.vmap(
                    grad(partial(dynamics_model_bound.energy_fn, coordinate="zeta")),
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
                    1, 1, figsize=figsize, num="Potential energy landscape in z-coordinates"
                )
                z1_range = jnp.linspace(-1.0, 1.0, 100)
                z2_range = jnp.linspace(-1.0, 1.0, 100)
                z1_grid, z2_grid = jnp.meshgrid(z1_range, z2_range)
                z_grid = jnp.stack([z1_grid, z2_grid], axis=-1)
                xi_grid = jnp.concatenate([z_grid, jnp.zeros_like(z_grid)], axis=-1)
                U_grid = jax.vmap(
                    partial(dynamics_model_bound.energy_fn),
                )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
                tau_pot_grid = -jax.vmap(
                    grad(partial(dynamics_model_bound.energy_fn)),
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


    if n_q == 2:
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
                        U = dynamics_model_bound.energy_fn(xi, coordinate="zeta")
                        tau_pot = -grad(
                            partial(dynamics_model_bound.energy_fn, coordinate="zeta")
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
                        U = dynamics_model_bound.energy_fn(xi)
                        tau_pot = -grad(dynamics_model_bound.energy_fn)(xi)[..., :n_tau]
                        U_grid = U_grid.at[i, j].set(U)
                        tau_pot_grid = tau_pot_grid.at[i, j, :].set(tau_pot)
            case _:
                raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            num="Learned potential energy landscape in configuration space",
        )
        # contour plot of the potential energy
        cs = axes[0].contourf(q1_grid, q2_grid, U_grid, levels=100)
        plt.colorbar(cs, ax=axes[0])
        axes[0].set_xlabel(r"$q_1$ [rad/m]")
        axes[0].set_ylabel(r"$q_2$ [rad/m]")
        axes[0].set_title("Learned potential energy in $q$-space")
        # quiver plot of the potential energy gradient
        qv_skip = 3
        qs = axes[1].quiver(
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
        qk = axes[1].quiverkey(
            qs, 0.9, 0.9, 1, r"$\tau: 1$ Nm", labelpos="E", coordinates="figure"
        )
        axes[1].set_xlabel(r"$q_1$ [rad/m]")
        axes[1].set_ylabel(r"$q_2$ [rad/m]")
        axes[1].set_title("Learned potential force in $q$-space")
        plt.colorbar(qs, ax=axes[1], label=r"$\tau$ [Nm]")
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "potential_energy_landscape_q.pdf")
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
        if use_collocated_form:
            setpoint_regulation_fn = (
                dynamics_model_bound.setpoint_regulation_collocated_form_fn
            )
        else:
            setpoint_regulation_fn = dynamics_model_bound.setpoint_regulation_fn
        tau, control_state, control_info = setpoint_regulation_fn(
            x,
            control_state,
            dt=control_dt,
            z_des=z_des,
            kp=kp,
            ki=ki,
            kd=kd,
            gamma=psatid_gamma,
        )
        # compute the control input
        """
        tau, control_state, control_info = dynamics_model.apply(
            {"params": state.params["dynamics"]},
            method=setpoint_regulation_fn,
        )
        """
        return tau, control_state, control_info

    # render target image
    target_img = rendering_fn(q_des)
    # normalize the target image
    target_img = jnp.array(
        preprocess_rendering(target_img, grayscale=True, normalize=True)
    )
    # encode the target image
    z_des = nn_model_bound.encode(target_img[None, ...])[0, ...]

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
    img_ts = onp.array(sim_ts["rendering_ts"])
    img_des_ts = onp.array(jnp.tile(target_img, reps=(img_ts.shape[0], 1, 1, 1)))

    # animate the rollout
    print("Animate the rollout...")
    animate_pred_vs_target_image_pyplot(
        onp.array(ts),
        img_pred_ts=img_ts,
        img_target_ts=img_des_ts,
        filepath=ckpt_dir / "controlled_rollout.mp4",
        step_skip=1,
        show=True,
        label_pred="Actual behavior",
        label_target="Desired behavior",
    )

    if simulate_with_learned_dynamics is False:
        # plot evolution of configuration space
        fig, ax = plt.subplots(1, 1, figsize=figsize, num="Configuration vs time")
        q_des_ts = jnp.tile(q_des, reps=(len(ts), 1))
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
        plt.savefig(ckpt_dir / "configuration_vs_time.pdf")
        plt.show()

    # plot evolution of latent state
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Latent vs time")
    z_des_ts = jnp.tile(z_des, reps=(len(ts), 1))
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
    plt.savefig(ckpt_dir / "latent_vs_time.pdf")
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
    plt.savefig(ckpt_dir / "latent_velocity_vs_time.pdf")
    plt.show()

    # plot collocated coordinates
    if "zeta_ts" in sim_ts:
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, num="Actuation coordinate vs time"
        )
        for i in range(n_z):
            ax.plot(
                ts, sim_ts["zeta_ts"][..., i], color=colors[i], label=rf"$\zeta_{i}$"
            )
            ax.plot(
                ts,
                sim_ts["zeta_des_ts"][..., i],
                linestyle="dashed",
                color=colors[i],
                label=rf"$\zeta_{i}^d$",
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"Actuation coordinate $\zeta$")
        ax.set_title("Actuation coordinate vs. time")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "collocated_vs_time.pdf")
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
    plt.savefig(ckpt_dir / "control_input_vs_time.pdf")
    plt.show()
    # plot the feedforward and feedback torques
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Latent-space torques vs time")
    for i in range(n_z):
        if use_collocated_form:
            ax.plot(
                ts,
                sim_ts["tau_zeta_ff_ts"][..., i],
                color=colors[i],
                label=r"$\tau_{\zeta,ff," + str(i) + "}$",
            )
            ax.plot(
                ts,
                sim_ts["tau_zeta_fb_ts"][..., i],
                linestyle="dotted",
                color=colors[i],
                label=r"$\tau_{\zeta,fb," + str(i) + "}$",
            )
        else:
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
    plt.savefig(ckpt_dir / "ff_fb_torques_vs_time.pdf")
    plt.show()

    energy_fn = getattr(dynamics_model_bound, "energy_fn", None)
    if callable(energy_fn):
        if type(dynamics_model) is ConOde:
            energy_fn = partial(
                energy_fn,
                coordinate="zw" if dynamics_model_bound.use_w_coordinates else "z",
            )

        # plot the energy over time
        fig, ax = plt.subplots(1, 1, figsize=figsize, num="Energy vs time")
        V_ts = jax.vmap(energy_fn)(xi_ts)
        ax.plot(ts, V_ts, color=colors[0], label="Energy")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Energy")
        ax.set_title("Energy vs. time")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "energy_vs_time.pdf")
        plt.show()
