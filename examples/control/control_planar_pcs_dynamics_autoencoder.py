import flax.linen as nn

from functools import partial
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")  # set default device to 'cpu'
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
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
dynamics_model_name = "node-w-con"
# latent space shape
n_z = 8
# number of configuration space dimensions
n_q = 2

# initial configuration
q0 = jnp.pi * jnp.array([0.0, -0.0])
# specify desired configuration
# q_des = jnp.array([jnp.pi, 1.25 * jnp.pi])
q_des = jnp.pi * jnp.array([0.0, 0.0])
# control settings
apply_feedforward_term = True
apply_feedback_term = True
use_collocated_form = True
# gains
if use_collocated_form:
    kp, ki, kd = 1e-3, 0e0, 0e0
    psatid_gamma = 1.0
else:
    kp, ki, kd = 1e0, 2e0, 0e0
    psatid_gamma = 0.5

batch_size = 10
norm_layer = nn.LayerNorm
cornn_gamma, cornn_epsilon = 1.0, 1.0
lnn_learn_dissipation = True
diag_shift, diag_eps = 1e-6, 2e-6
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
    datasets, dataset_info, dataset_metadata = load_dataset(
        f"planar_pcs/{system_type}_32x32px",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # extract the robot parameters from the dataset
    robot_params = dataset_metadata["system_params"]
    # TODO: move the damping to the controller
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
    sim_duration = 3.0  # s
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
        init_fn=nn_model.initialize_all_weights,
    )
    dynamics_model_bound = dynamics_model.bind(
        {"params": state.params["dynamics"]}
    )

    if n_z == 2:
        # plot the potential energy landscape in the original latent space
        fig, ax = plt.subplots(1, 1, figsize=figsize, num="Potential energy landscape in z-coordinates")
        z1_range = jnp.linspace(-1.0, 1.0, 100)
        z2_range = jnp.linspace(-1.0, 1.0, 100)
        z1_grid, z2_grid = jnp.meshgrid(z1_range, z2_range)
        z_grid = jnp.stack([z1_grid, z2_grid], axis=-1)
        xi_grid = jnp.concatenate([
            z_grid,
            jnp.zeros_like(z_grid)
        ], axis=-1)
        U_grid = jax.vmap(
            partial(dynamics_model_bound.energy_fn, coordinate="z"),
        )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
        tau_pot = -jax.vmap(
            grad(partial(dynamics_model_bound.energy_fn, coordinate="z")),
        )(xi_grid.reshape(-1, xi_grid.shape[-1]))[..., :n_z].reshape(*xi_grid.shape[:2], -1)
        # contour plot of the potential energy
        cs = ax.contourf(z1_grid, z2_grid, U_grid, levels=100)
        # quiver plot of the potential energy gradient
        ax.quiver(
            z1_grid,
            z2_grid,
            tau_pot[..., 0],
            tau_pot[..., 1],
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
        fig, ax = plt.subplots(1, 1, figsize=figsize, num="Potential energy landscape in zw-coordinates")
        zw1_range = jnp.linspace(-1.0, 1.0, 100)
        zw2_range = jnp.linspace(-1.0, 1.0, 100)
        zw1_grid, zw2_grid = jnp.meshgrid(zw1_range, zw2_range)
        zw_grid = jnp.stack([zw1_grid, zw2_grid], axis=-1)
        xi_grid = jnp.concatenate([
            zw_grid,
            jnp.zeros_like(zw_grid)
        ], axis=-1)
        U_grid = jax.vmap(
            partial(dynamics_model_bound.energy_fn, coordinate="zw"),
        )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
        tau_pot = -jax.vmap(
            grad(partial(dynamics_model_bound.energy_fn, coordinate="zw")),
        )(xi_grid.reshape(-1, xi_grid.shape[-1]))[..., :n_z].reshape(*xi_grid.shape[:2], -1)
        # contour plot of the potential energy
        cs = ax.contourf(zw1_grid, zw2_grid, U_grid, levels=100)
        # quiver plot of the potential energy gradient
        ax.quiver(
            zw1_grid,
            zw2_grid,
            tau_pot[..., 0],
            tau_pot[..., 1],
            angles="xy",
            scale=None,
            scale_units="xy",
            color="white",
        )
        plt.colorbar(cs)
        ax.set_xlabel(r"$z_{w,1}$")
        ax.set_ylabel(r"$z_{w,2}$")
        ax.set_title("Potential energy landscape of learned latent dynamics in w-coordinates")
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "potential_energy_landscape_zw.pdf")
        plt.show()

        # plot the potential energy in the collocated coordinates
        fig, ax = plt.subplots(1, 1, figsize=figsize, num="Potential energy landscape in collocated coordinates")
        zeta1_range = jnp.linspace(-1.0, 1.0, 100)
        zeta2_range = jnp.linspace(-1.0, 1.0, 100)
        zeta1_grid, zeta2_grid = jnp.meshgrid(zeta1_range, zeta2_range)
        zeta_grid = jnp.stack([zeta1_grid, zeta2_grid], axis=-1)
        xi_grid = jnp.concatenate([
            zeta_grid,
            jnp.zeros_like(zeta_grid)
        ], axis=-1)
        U_grid = jax.vmap(
            partial(dynamics_model_bound.energy_fn, coordinate="zeta"),
        )(xi_grid.reshape(-1, xi_grid.shape[-1])).reshape(xi_grid.shape[:2])
        tau_pot = -jax.vmap(
            grad(partial(dynamics_model_bound.energy_fn, coordinate="zeta")),
        )(xi_grid.reshape(-1, xi_grid.shape[-1]))[..., :n_z].reshape(*xi_grid.shape[:2], -1)
        # contour plot of the potential energy
        cs = ax.contourf(zeta1_grid, zeta2_grid, U_grid, levels=100)
        # quiver plot of the potential energy gradient
        ax.quiver(
            zeta1_grid,
            zeta2_grid,
            tau_pot[..., 0],
            tau_pot[..., 1],
            angles="xy",
            scale=None,
            scale_units="xy",
            color="white",
        )
        plt.colorbar(cs)
        ax.set_xlabel(r"$\zeta_1$")
        ax.set_ylabel(r"$\zeta_2$")
        ax.set_title("Potential energy landscape of learned latent dynamics in collocated coordinates")
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "potential_energy_landscape_zeta.pdf")
        plt.show()
    

    def control_fn(t: Array, x: Array, control_state: Dict[str, Array]) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
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
            setpoint_regulation_fn = dynamics_model_bound.setpoint_regulation_collocated_form_fn
        else:
            setpoint_regulation_fn = dynamics_model_bound.setpoint_regulation_control_fn
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
    target_img = jnp.array(preprocess_rendering(target_img, grayscale=True, normalize=True))
    # encode the target image
    target_img_bt = target_img[None, ...]
    z_des_bt = nn_model.apply(
        {"params": state.params}, target_img_bt, method=nn_model.encode
    )
    z_des = z_des_bt[0, :]

    # set initial condition for closed-loop simulation
    x0 = jnp.concatenate([q0, jnp.zeros((n_q,))])

    # start closed-loop simulation
    print("Simulating closed-loop dynamics...")
    sim_ts = rollout_ode_with_latent_space_control(
        ode_fn=ode_fn,
        rendering_fn=rendering_fn,
        encode_fn=jit(
            partial(
                nn_model.apply,
                {"params": state.params},
                method=nn_model.encode,
            )
        ),
        ts=ts,
        sim_dt=sim_dt,
        x0=x0,
        input_dim=n_tau,
        latent_dim=n_z,
        control_fn=jit(control_fn),
        control_state_init={"e_int": jnp.zeros((n_z,))},
    )

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

    # plot evolution of configuration space
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Configuration vs. time")
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
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Latent vs. time")
    z_des_ts = jnp.tile(z_des, reps=(len(ts), 1))
    for i in range(n_z):
        ax.plot(ts, sim_ts["xi_ts"][..., i], color=colors[i], label=f"$z_{i}$")
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

    # plot collocated coordinates
    if "zeta_ts" in sim_ts:
        fig, ax = plt.subplots(1, 1, figsize=figsize, num="Actuation coordinate vs. time")
        for i in range(n_z):
            ax.plot(ts, sim_ts["zeta_ts"][..., i], color=colors[i], label=fr"$\zeta_{i}$")
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
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Control input vs. time")
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
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Latent-space torques vs. time")
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

    # plot the energy over time
    fig, ax = plt.subplots(1, 1, figsize=figsize, num="Energy vs. time")
    V_ts = jax.vmap(
        partial(
            dynamics_model.apply,
            {"params": state.params["dynamics"]},
            method=dynamics_model.energy_fn,
        )
    )(sim_ts["xi_ts"])
    ax.plot(ts, V_ts, color=colors[0], label="Energy")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs. time")
    ax.legend()
    plt.grid(True)
    plt.box(True)
    plt.savefig(ckpt_dir / "energy_vs_time.pdf")
    plt.show()
