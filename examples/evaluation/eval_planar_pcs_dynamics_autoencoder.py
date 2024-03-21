import flax.linen as nn
from functools import partial
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", ["cpu"])  # set default device to 'cpu'
from jax import Array, jit, random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
import tensorflow as tf

from src.models.autoencoders import Autoencoder, VAE
from src.models.discrete_forward_dynamics import (
    DiscreteLssDynamics,
    DiscreteMambaDynamics,
    DiscreteMlpDynamics,
    DiscreteRnnDynamics,
)
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
from src.rendering import render_planar_pcs
from src.rollout import rollout_ode
from src.training.dataset_utils import load_dataset, load_dummy_neural_network_input
from src.training.loops import run_eval
from src.tasks import dynamics_autoencoder
from src.training.train_state_utils import restore_train_state
from src.visualization.img_animation import (
    animate_pred_vs_target_image_cv2,
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
""" dynamics_model_name in [
    "node-general-mlp", "node-mechanical-mlp", "node-mechanical-mlp-s", 
    "node-cornn", "node-con", "node-w-con", "node-con-iae", "node-con-iae-s", "node-dcon", "node-lnn", 
    "node-hippo-lss", "node-mamba",
    "discrete-mlp", "discrete-elman-rnn", "discrete-gru-rnn", "discrete-general-lss", "discrete-hippo-lss", "discrete-mamba",
]
"""
dynamics_model_name = "node-con-iae-s"
# latent space shape
n_z = 30
# simulation time step
sim_dt = None

batch_size = 10
loss_weights = dict(mse_q=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0)
start_time_idx = 1
num_past_timesteps = 2

norm_layer = nn.LayerNorm
num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 20, "leaky_relu"
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
        if dynamics_model_name == "node-mechanical-mlp":
            experiment_id = "2024-02-13_16-27-39"
            num_mlp_layers, mlp_hidden_dim = 4, 52
            mlp_nonlinearity_name = "elu"
        elif dynamics_model_name == "node-cornn":
            experiment_id = "2024-02-14_18-17-49"
            cornn_gamma, cornn_epsilon = 35.60944428175452, 0.05125440449424828
        elif dynamics_model_name == "node-con":
            experiment_id = "2024-02-14_18-34-27"
        elif dynamics_model_name == "node-w-con":
            match n_z:
                case 2:
                    experiment_id = "2024-02-22_14-11-21"
                case 4:
                    experiment_id = "2024-02-14_22-52-37"
                case 8:
                    experiment_id = "2024-02-21_13-34-53"
                case _:
                    raise ValueError(f"No experiment_id for n_z={n_z}")
        elif dynamics_model_name == "discrete-mlp":
            experiment_id = "2024-02-14_17-45-30"
            num_mlp_layers, mlp_hidden_dim = 4, 95
            mlp_nonlinearity_name = "elu"
        elif dynamics_model_name == "discrete-elman-rnn":
            experiment_id = "2024-02-13_17-19-57"
        elif dynamics_model_name == "discrete-gru-rnn":
            experiment_id = "2024-02-13_17-28-13"
        elif dynamics_model_name == "discrete-mamba":
            experiment_id = "2024-02-13_17-42-29"
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
    if dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp"]:
        dynamics_model = MlpOde(
            latent_dim=n_z,
            input_dim=n_tau,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
            mechanical_system=True
            if dynamics_model_name == "node-mechanical-mlp"
            else False,
        )
    elif dynamics_model_name == "node-cornn":
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
        )
    elif dynamics_model_name in ["node-con-iae", "node-con-iae-s"]:
        dynamics_model = ConIaeOde(
            latent_dim=n_z,
            input_dim=n_tau,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
        )
    elif dynamics_model_name == "node-lnn":
        dynamics_model = LnnOde(
            latent_dim=n_z,
            input_dim=n_tau,
            learn_dissipation=lnn_learn_dissipation,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
            diag_shift=diag_shift,
            diag_eps=diag_eps,
        )
    elif dynamics_model_name in [
        "node-general-lss",
        "node-mechanical-lss",
        "node-hippo-lss",
    ]:
        dynamics_model = LinearStateSpaceOde(
            latent_dim=n_z,
            input_dim=n_tau,
            transition_matrix_init=dynamics_model_name.split("-")[
                1
            ],  # "general", "mechanical", or "hippo"
        )
    elif dynamics_model_name == "discrete-mlp":
        dynamics_model = DiscreteMlpDynamics(
            state_dim=num_past_timesteps * n_z,
            input_dim=num_past_timesteps * n_tau,
            output_dim=n_z,
            dt=dataset_metadata["dt"],
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
        )
    elif dynamics_model_name in ["discrete-elman-rnn", "discrete-gru-rnn"]:
        dynamics_model = DiscreteRnnDynamics(
            state_dim=num_past_timesteps * n_z,
            input_dim=num_past_timesteps * n_tau,
            output_dim=n_z,
            rnn_method=dynamics_model_name.split("-")[1],  # "elman" or "gru"
        )
    elif dynamics_model_name == "discrete-mamba":
        dynamics_model = DiscreteMambaDynamics(
            state_dim=num_past_timesteps * n_z,
            input_dim=num_past_timesteps * n_tau,
            output_dim=n_z,
            dt=dataset_metadata["dt"],
        )
    else:
        raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")
    nn_model = DynamicsAutoencoder(
        autoencoder=autoencoder_model,
        dynamics=dynamics_model,
        dynamics_type=dynamics_type,
        num_past_timesteps=num_past_timesteps,
    )

    # import solver class from diffrax
    # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    solver_class = getattr(
        __import__("diffrax", fromlist=[dataset_metadata["solver_class"]]),
        dataset_metadata["solver_class"],
    )

    # call the factory function for the dynamics autoencoder task
    task_callables, metrics_collection_cls = dynamics_autoencoder.task_factory(
        system_type,
        nn_model,
        ts=dataset_metadata["ts"],
        sim_dt=dataset_metadata["sim_dt"] if sim_dt is None else sim_dt,
        loss_weights=loss_weights,
        ae_type=ae_type,
        dynamics_type=dynamics_type,
        start_time_idx=start_time_idx,
        solver=solver_class(),
        latent_velocity_source="image-space-finite-differences",
        num_past_timesteps=num_past_timesteps,
        compute_psnr=True,
        compute_ssim=True,
    )

    # load the neural network dummy input
    nn_dummy_input = load_dummy_neural_network_input(test_ds, task_callables)
    # load the training state from the checkpoint directory
    state = restore_train_state(
        rng=rng,
        ckpt_dir=ckpt_dir,
        nn_model=nn_model,
        nn_dummy_input=nn_dummy_input,
        metrics_collection_cls=metrics_collection_cls,
        init_fn=nn_model.forward_all_layers,
    )

    # bind params to the models
    nn_model_bound = nn_model.bind({"params": state.params})
    dynamics_model_bound = dynamics_model.bind({"params": state.params["dynamics"]})

    print("Run testing...")
    state, test_history = run_eval(test_ds, state, task_callables)
    test_metrics = state.metrics.compute()
    print(
        "\n"
        f"Final test metrics:\n"
        f"rmse_rec_static={test_metrics['rmse_rec_static']:.4f}, "
        f"rmse_rec_dynamic={test_metrics['rmse_rec_dynamic']:.4f}, "
        f"psnr_rec_static={test_metrics['psnr_rec_static']:.4f}, "
        f"psnr_rec_dynamic={test_metrics['psnr_rec_dynamic']:.4f}, "
        f"ssim_rec_static={test_metrics['ssim_rec_static']:.4f}, "
        f"ssim_rec_dynamic={test_metrics['ssim_rec_dynamic']:.4f}"
    )

    # define settings for the rollout
    rollout_duration = 3.0  # s
    rollout_fps = 30  # frames per second
    rollout_dt = 1 / rollout_fps  # s
    rollout_sim_dt = 1e-3 * rollout_dt  # simulation time step of 1e-5 s
    ts_rollout = jnp.linspace(
        0.0, rollout_duration, num=int(rollout_duration / rollout_dt)
    )
    ode_rollout_fn = partial(
        rollout_ode,
        ode_fn=ode_fn,
        ts=ts_rollout,
        sim_dt=rollout_sim_dt,
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
        ts=ts_rollout,
        sim_dt=rollout_sim_dt,
        loss_weights=loss_weights,
        ae_type=ae_type,
        dynamics_type=dynamics_type,
        start_time_idx=start_time_idx,
        solver=solver_class(),
        latent_velocity_source="image-space-finite-differences",
        num_past_timesteps=num_past_timesteps,
    )
    forward_fn_learned = jit(task_callables_rollout_learned.forward_fn)

    # rollout dynamics
    print("Rollout...")
    q0 = (
        0.5
        * jnp.tile(jnp.array([1.0, -1.0]), reps=int(jnp.ceil(n_q / 2)))[:n_q]
        * dataset_metadata["x0_max"][:n_q]
    )
    x0 = jnp.concatenate([q0, jnp.zeros((n_q,))])
    # tau = jnp.zeros((n_tau,))
    tau = -0.5 * dataset_metadata["tau_max"]
    print("x0:", x0, "tau:", tau)
    sim_ts = ode_rollout_fn(x0=x0, tau=tau)
    rollout_batch = dict(
        t_ts=ts_rollout[None, ...],
        x_ts=sim_ts["x_ts"][None, ...],
        tau=tau[None, ...],
        rendering_ts=sim_ts["rendering_ts"][None, :],
    )
    preds = forward_fn_learned(rollout_batch, state.params)
    # extract both the target and the predicted images
    img_pred_ts = preds["img_dynamic_ts"][0]
    img_target_ts = sim_ts["rendering_ts"][start_time_idx:]
    # extract the latent state trajectory
    xi_ts = preds["xi_dynamic_ts"][0]

    # animate the rollout
    print("Animate the rollout...")
    animate_pred_vs_target_image_pyplot(
        onp.array(ts_rollout),
        img_pred_ts=img_pred_ts,
        img_target_ts=img_target_ts,
        filepath=ckpt_dir / "rollout.mp4",
        step_skip=1,
        show=True,
    )

    energy_fn = getattr(dynamics_model_bound, "energy_fn", None)
    if callable(energy_fn):
        if type(dynamics_model) is ConOde:
            energy_fn = partial(
                energy_fn,
                coordinate="zw" if dynamics_model_bound.use_w_coordinates else "z",
            )

        # plot the energy over time
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), num="Energy vs. time")
        V_ts = jax.vmap(energy_fn)(xi_ts)
        ax.plot(ts_rollout[start_time_idx:], V_ts, label="Energy")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Energy")
        ax.set_title("Energy vs. time")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(ckpt_dir / "energy_vs_time.pdf")
        plt.show()
