import flax.linen as nn
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'
from jax import Array, jit, random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import planar_pcs
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
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
""" dynamics_model_name in [
    "node-general-mlp", "node-mechanical-mlp", "node-cornn", "node-con", "node-w-con", "node-lnn", "node-hippo-lss", "mambda-ode",
    "discrete-mlp", "discrete-elman-rnn", "discrete-gru-rnn", "discrete-general-lss", "discrete-hippo-lss", "discrete-mamba",
]
"""
dynamics_model_name = "discrete-mlp"
# latent space shape
n_z = 4
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
if ae_type == "wae":
    raise NotImplementedError
elif ae_type == "beta_vae":
    if dynamics_model_name == "node-mechanical-mlp":
        experiment_id = "2024-02-13_16-27-39"
        num_mlp_layers, mlp_hidden_dim = 4, 52
        mlp_nonlinearity_name = "elu"
    elif dynamics_model_name == "node-con":
        experiment_id = "2024-02-13_12-57-25"
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
        nn_model = DiscreteRnnDynamics(
            state_dim=num_past_timesteps * n_z,
            input_dim=num_past_timesteps * n_tau,
            output_dim=n_z,
            rnn_method=dynamics_model_name.split("-")[1],  # "elman" or "gru"
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
        init_fn=nn_model.initialize_all_weights,
    )

    print("Run testing...")
    state, test_history = run_eval(test_ds, state, task_callables)
    test_metrics = state.metrics.compute()
    print(
        "\n"
        f"Final test metrics:\n"
        f"rmse_rec_static={test_metrics['rmse_rec_static']:.4f}, "
        f"rmse_rec_dynamic={test_metrics['rmse_rec_dynamic']:.4f}"
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

    test_batch = next(test_ds.as_numpy_iterator())

    # rollout dynamics
    print("Rollout...")
    x0 = jnp.concatenate([dataset_metadata["x0_max"][:n_q], jnp.zeros((n_q,))])
    tau = jnp.zeros((n_tau,))
    print("x0:", x0)
    rollout_data_ts = ode_rollout_fn(x0=x0, tau=tau)
    rollout_batch = dict(
        t_ts=ts_rollout[None, ...],
        x_ts=rollout_data_ts["x_ts"][None, ...],
        tau=tau[None, ...],
        rendering_ts=rollout_data_ts["rendering_ts"][None, :],
    )
    preds = forward_fn_learned(rollout_batch, state.params)
    # extract both the target and the predicted images
    rendering_pred_ts = preds["rendering_dynamic_ts"][0]
    rendering_target_ts = rollout_data_ts["rendering_ts"][start_time_idx:]

    # animate the rollout
    print("Animate the rollout...")
    animate_pred_vs_target_image_pyplot(
        onp.array(ts_rollout),
        img_pred_ts=rendering_pred_ts,
        img_target_ts=rendering_target_ts,
        filepath=ckpt_dir / "rollout.mp4",
        step_skip=1,
        show=True,
    )
