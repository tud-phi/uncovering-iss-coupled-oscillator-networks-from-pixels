import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'

from dm_hamiltonian_dynamics_suite.datasets import MASS_SPRING_FRICTION_ACTUATION
import flax.linen as nn
from functools import partial
from jax import Array, debug, grad, jit, lax, random
import jax.numpy as jnp
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
from src.rendering import preprocess_rendering
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

system_type = "mass_spring_friction_actuation"
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
dynamics_model_name = (
    "node-con-iae"  # "node-con-iae", "node-con-iae-s", "node-mechanical-mlp"
)
# latent space shape
n_z = 1
# number of configuration space dimensions
n_q = 1
# whether to use real or learned dynamics
simulate_with_learned_dynamics = False

# simulation settings
# num setpoints
num_setpoints = 7
sim_duration_per_setpoint = 5.0  # s
sim_duration = num_setpoints * sim_duration_per_setpoint  # s
# initial configuration
q0 = jnp.array([0.0])
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
            if apply_feedforward_term:
                kp, ki, kd = 2e1, 2.0e0, 4e0
            else:
                kp, ki, kd = 2e1, 2.0e0, 4e0
            psatid_gamma = 1.0
    case "node-mechanical-mlp":
        kp, ki, kd = 1e-3, 2e-2, 1e-5
        psatid_gamma = 1.0
    case _:
        raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")
print(f"Control gains: kp={kp}, ki={ki}, kd={kd}, gamma={psatid_gamma}")

batch_size = 10
norm_layer = nn.LayerNorm
diag_shift, diag_eps = 1e-6, 2e-6
match dynamics_model_name:
    case "node-con-iae":
        experiment_id = f"2024-09-25_16-15-11/n_z_{n_z}_seed_{seed}"
        num_mlp_layers, mlp_hidden_dim = 5, 30
    case "node-mechanical-mlp":
        raise NotImplementedError
        experiment_id = f"2024-05-21_07-45-14/n_z_{n_z}_seed_{seed}"
        num_mlp_layers, mlp_hidden_dim = 5, 30
    case _:
        raise ValueError(
            f"No experiment_id for dynamics_model_name={dynamics_model_name}"
        )

# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in ["node", "discrete"], f"Unknown dynamics_type: {dynamics_type}"

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
    dataset_name = f"toy_physics/{system_type}_dt_0_05"
    print(f"Loading dataset: {dataset_name}")
    datasets, dataset_info, dataset_metadata = load_dataset(
        dataset_name,
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
        dataset_type="dm_hamiltonian_dynamics_suite"
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # initialize the system
    system_cls, config_fn = globals().get(system_type.upper())
    system_config = config_fn()
    system = system_cls(**system_config)
    # print(f"System: {system}", f"System configuration: {system_config}")

    # extract the robot parameters from the dataset
    robot_params = dict(
        m=system_config["m_range"].max,
        k=system_config["k_range"].max,
        d=system_config["friction"],
    )
    print(f"Robot parameters: {robot_params}")
    # size of torques
    n_tau = train_ds.element_spec["tau"].shape[-1]  # dimension of the control input
    print(f"Control input dimension: {n_tau}")
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape
    print(f"Image shape: {img_shape}")
    # limits of the configuration space
    q0_min, q0_max = dataset_metadata["x0_min"][:n_q], dataset_metadata["x0_max"][:n_q]
    print(f"Configuration space limits: {q0_min}, {q0_max}")

    # define the ode and potential energy functions of the system

    def system_ode_fn(t: float, x: Array, tau: Array) -> Array:
        q, q_d = jnp.split(x, 2, axis=-1)
        x_d = jnp.concatenate([
            q_d,
            (tau - robot_params["d"] * q_d - robot_params["k"] * q) / robot_params["m"]
        ], axis=-1)
        return x_d

    def system_potential_energy_fn(q: Array) -> Array:
        return 0.5 * robot_params["k"] * q ** 2


    # initialize the rendering function
    def rendering_fn(q: Array) -> Array:
        params = dict(
            m=robot_params["m"],
            k=robot_params["k"],
        )
        imgs, extra = system.render_trajectories(q[None, None, ...], params=params, rng_key=rng)
        img = jnp.array(imgs[0, 0, ...])
        return img

    # configure the rendering preprocessor
    preprocess_rendering_kwargs = dict(
        grayscale=True,
        normalize=True,
        img_min_val=dataset_metadata["rendering"]["img_min_val"],
        img_max_val=dataset_metadata["rendering"]["img_max_val"],
    )
    preprocess_rendering_fn = partial(preprocess_rendering, **preprocess_rendering_kwargs)

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

    # generate a random setpoint sequence
    rng_setpoint = random.PRNGKey(seed=1)
    q_des_ps = random.uniform(
        rng_setpoint, shape=(num_setpoints, n_q), minval=q0_min, maxval=q0_max
    )

    # define settings for the closed-loop simulation
    control_dt = 1e-2  # control and time step of 1e-2 s
    sim_dt = 5e-4 * control_dt  # simulation time step of 1e-5 s
    ts = jnp.linspace(0.0, sim_duration, num=int(sim_duration / control_dt))
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
    potential_energy_fn: callable = getattr(
        dynamics_model_bound, "potential_energy_fn", None
    )
    kinetic_energy_fn: callable = getattr(
        dynamics_model_bound, "kinetic_energy_fn", None
    )

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
    img_q0_max = preprocess_rendering_fn(img_q0_max)
    z0_max = jnp.abs(nn_model_bound.encode(img_q0_max[None, ...])[0, ...])

    if n_z == 1 and dynamics_model_name in ["node-con-iae", "node-con-iae-s"]:
        z_ps = jnp.linspace(-z0_max[0], z0_max[0], 100)[:, None]
        xi_ps = jnp.concatenate([z_ps, jnp.zeros_like(z_ps)], axis=-1)

        # evaluate the potential energy on the grid of latent variables
        Uz_ps = jax.vmap(potential_energy_fn)(xi_ps)

        # plot the potential energy landscape in the original latent space
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="Potential energy landscape in z-coordinates",
        )
        ax.plot(z_ps, Uz_ps, linewidth=2.0, label=r"$\mathcal{U}(z)$")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\mathcal{U}$")
        plt.grid(True)
        plt.legend()
        plt.savefig(ckpt_dir / "potential_energy_landscape_z.pdf")
        plt.show()

    if callable(potential_energy_fn) and n_q == 1 and dynamics_model_name in ["node-con-iae", "node-con-iae-s"]:
        q_ps = jnp.linspace(q0_min[0], q0_max[0], 100)[:, None]
        xi_ps = []
        Uq_hat_ps, Uq_gt_ps = [], []
        for i in range(q_ps.shape[0]):
            q = q_ps[i]
            img = rendering_fn(q)
            img = preprocess_rendering_fn(img)
            z = nn_model_bound.encode(img[None, ...])[0, ...]
            xi = jnp.concatenate([z, jnp.zeros((n_z,))])

            # compute the learned potential energy landscape in the configuration coordinates
            Uq_hat = potential_energy_fn(xi)
            # compute the ground-truth potential energy landscape in the configuration space
            Uq_gt = system_potential_energy_fn(q)

            xi_ps.append(xi)
            Uq_hat_ps.append(Uq_hat)
            Uq_gt_ps.append(Uq_gt)
        xi_ps = jnp.stack(xi_ps, axis=0)
        Uq_hat_ps, Uq_gt_ps = jnp.stack(Uq_hat_ps, axis=0), jnp.stack(Uq_gt_ps, axis=0)

        # plot the mapping from configurtion space to latent space
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="Mapping from configuration to latent space",
        )
        ax.plot(q_ps, xi_ps[..., 0], linewidth=2.0, label=r"$z(q)$")
        ax.set_xlabel(r"$q$ [m]")
        ax.set_ylabel(r"$z$")
        plt.grid(True)
        plt.legend()
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.savefig(ckpt_dir / "mapping_configuration_to_latent_space.pdf")
        plt.show()

        # plot the potential energy landscape in the configuration space
        """
        fig, ax1 = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="Learned potential energy landscape in configuration space",
        )
        ax2 = ax1.twinx()
        ax1.plot(q_ps, Uq_hat_ps, color=colors[0], label=r"$\hat{\mathcal{U}}(q)$")
        ax2.plot(q_ps, Uq_gt_ps, color=colors[1], label=r"$\mathcal{U}_{\mathrm{gt}}(q)$")
        ax1.set_xlabel(r"$q$")
        ax1.set_ylabel(r"$\hat{\mathcal{U}}$")
        ax2.set_ylabel(r"$\mathcal{U}_{\mathrm{gt}}$")
        plt.grid(True)
        ax1.legend()
        ax2.legend()
        plt.savefig(ckpt_dir / "potential_energy_landscape_q.pdf")
        plt.show()
        """
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="Learned potential energy landscape in configuration space",
        )
        ax.plot(q_ps, Uq_gt_ps, linewidth=2.5, color=colors[0], label=r"$\mathcal{U}_{\mathrm{gt}}(q)$")
        ax.plot(q_ps, Uq_hat_ps, linewidth=2.0, color=colors[1], label=r"$\hat{\mathcal{U}}(q)$")
        ax.set_xlabel(r"$q$ [m]")
        ax.set_ylabel(r"$\mathcal{U}$")
        plt.grid(True)
        ax.legend()
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

        return tau, control_state, control_info

    # render and encode all the target images
    img_des_ps = jnp.zeros((num_setpoints, *img_shape))
    z_des_ps = jnp.zeros((num_setpoints, n_z))
    for setpoint_idx in range(num_setpoints):
        q_des = q_des_ps[setpoint_idx, :]
        # render target image
        img_des = rendering_fn(q_des)
        # normalize the target image
        img_des = preprocess_rendering_fn(img_des)
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
        img0 = preprocess_rendering_fn(img0)
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
            preprocess_rendering_kwargs=preprocess_rendering_kwargs,
        )
        xi_ts = sim_ts["x_ts"]
    else:
        # start closed-loop simulation of real dynamics with latent space control
        print("Simulating real closed-loop dynamics...")
        sim_ts = rollout_ode_with_latent_space_control(
            ode_fn=system_ode_fn,
            rendering_fn=rendering_fn,
            encode_fn=jit(encode_fn),
            ts=ts,
            sim_dt=sim_dt,
            x0=x0,
            input_dim=n_tau,
            latent_dim=n_z,
            control_fn=jit(control_fn),
            control_state_init={"e_int": jnp.zeros((n_z,))},
            preprocess_rendering_kwargs=preprocess_rendering_kwargs,
        )
        xi_ts = sim_ts["xi_ts"]

    # extract both the ground-truth and the statically predicted images
    img_ts = sim_ts["rendering_ts"]
    # add the desired setpoints to the sim_ts dictionary
    sim_ts["q_des_ts"] = q_des_ts
    sim_ts["img_des_ts"] = img_des_ts
    sim_ts["z_des_ts"] = z_des_ts

    if callable(energy_fn):
        sim_ts["V_ts"] = jax.vmap(energy_fn)(xi_ts)  # total energy
        sim_ts["T_ts"] = jax.vmap(kinetic_energy_fn)(xi_ts)  # kinetic energy
        sim_ts["U_ts"] = jax.vmap(potential_energy_fn)(xi_ts)  # potential energy
        sim_ts["U_des_ts"] = jax.vmap(potential_energy_fn)(
            z_des_ts
        )  # desired potential energy

    # save the simulation results
    onp.savez(ckpt_dir / "setpoint_sequence_controlled_rollout.npz", **sim_ts)

    # denormalize the images
    img_ts = jax.vmap(partial(denormalize_img, apply_threshold=True))(img_ts)
    img_des_ts = jax.vmap(partial(denormalize_img, apply_threshold=True))(img_des_ts)

    # animate the rollout
    print("Animate the rollout...")
    animate_pred_vs_target_image_pyplot(
        onp.array(ts),
        img_pred_ts=onp.array(img_ts),
        img_target_ts=onp.array(img_des_ts),
        filepath=ckpt_dir / "setpoint_sequence_controlled_rollout.mp4",
        skip_step=1,
        show=True,
        label_pred="Actual behavior",
        label_target="Desired behavior",
    )
    animate_image_cv2(
        onp.array(ts),
        onp.array(img_ts).astype(onp.uint8),
        filepath=ckpt_dir / "setpoint_sequence_controlled_rollout_actual.mp4",
        skip_step=2,
    )
    animate_image_cv2(
        onp.array(ts),
        onp.array(img_des_ts).astype(onp.uint8),
        filepath=ckpt_dir / "setpoint_sequence_controlled_rollout_desired.mp4",
        skip_step=2,
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
