import flax.linen as nn
from functools import partial
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
from jax import Array, jit, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
import tensorflow as tf

from src.models.autoencoders import Autoencoder, VAE
from src.models.discrete_forward_dynamics import (
    DiscreteConIaeCfaDynamics,
    DiscreteCornn,
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
    MlpOde,
)
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.training.dataset_utils import load_dataset, load_dummy_neural_network_input
from src.training.loops import run_eval
from src.tasks import dynamics_autoencoder
from src.training.train_state_utils import restore_train_state
from src.visualization.img_animation import (
    animate_image_cv2,
    animate_pred_vs_target_image_cv2,
    animate_pred_vs_target_image_pyplot,
)
from src.visualization.utils import denormalize_img

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

# set the system type in [
# "cc", "cs", "pcc_ns-2",
# "mass_spring_friction", "mass_spring_friction_actuation", "pendulum_friction", "double_pendulum_friction",
# "single_pendulum", "reaction_diffusion_default"]
system_type = "pcc_ns-2"
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
""" dynamics_model_name in [
    "node-general-mlp", "node-mechanical-mlp", "node-mechanical-mlp-s",
    "node-cornn", "node-con", "node-w-con", "node-con-iae", "node-con-iae-s", "node-lnn",
    "node-hippo-lss",
    "discrete-mlp", "discrete-elman-rnn", "discrete-gru-rnn", "discrete-general-lss", "discrete-hippo-lss", "discrete-mamba",
    "ar-con-iae-cfa", "ar-elman-rnn", "ar-gru-rnn", "ar-cornn"
]
"""
dynamics_model_name = "node-con-iae"
# simulation time step
if system_type in ["cc", "cs", "pcc_ns-2", "pcc_ns-3", "pcc_ns-4"]:
    sim_dt = 1e-2
elif system_type in [
    "single_pendulum",
    "double_pendulum",
    "mass_spring_friction",
    "mass_spring_friction_actuation",
    "pendulum_friction",
    "double_pendulum_friction",
    "reaction_diffusion_default",
]:
    sim_dt = 2.5e-2
else:
    raise ValueError(f"Unknown system_type: {system_type}")

batch_size = 5
loss_weights = dict(mse_q=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0)
start_time_idx = 1
num_past_timesteps = 2

norm_layer = nn.LayerNorm
num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 20, "leaky_relu"
cornn_gamma, cornn_epsilon = 1.0, 1.0
lnn_learn_dissipation = True
diag_shift, diag_eps = 1e-6, 2e-6
grayscale = True
match system_type:
    case "mass_spring_friction":
        n_z = 4  # latent space dimension
        match dynamics_model_name:
            case "node-con-iae":
                experiment_id = f"2024-08-05_18-31-04/n_z_{n_z}_seed_{seed}"
                num_mlp_layers, mlp_hidden_dim = 5, 30
            case _:
                raise ValueError(
                    f"No experiment_id for dynamics_model_name={dynamics_model_name}"
                )
    case "mass_spring_friction_actuation":
        n_z = 1  # latent space dimension
        match dynamics_model_name:
            case "node-con-iae":
                experiment_id = f"2024-09-26_16-00-56/n_z_{n_z}_seed_{seed}"
                num_mlp_layers, mlp_hidden_dim = 5, 30
            case "node-mechanical-mlp":
                experiment_id = f"2024-09-26_05-16-30/n_z_{n_z}_seed_{seed}"
                num_mlp_layers, mlp_hidden_dim = 5, 30
            case _:
                raise ValueError(
                    f"No experiment_id for dynamics_model_name={dynamics_model_name}"
                )
    case "pendulum_friction":
        n_z = 4  # latent space dimension
        match dynamics_model_name:
            case "node-con-iae":
                experiment_id = f"2024-08-06_02-23-26/n_z_{n_z}_seed_{seed}"
                num_mlp_layers, mlp_hidden_dim = 5, 30
            case _:
                raise ValueError(
                    f"No experiment_id for dynamics_model_name={dynamics_model_name}"
                )
    case "double_pendulum_friction":
        n_z = 12  # latent space dimension
        grayscale = False
        match dynamics_model_name:
            case "node-con-iae":
                experiment_id = f"2024-08-06_15-00-51/n_z_{n_z}_seed_{seed}"
                num_mlp_layers, mlp_hidden_dim = 5, 30
            case _:
                raise ValueError(
                    f"No experiment_id for dynamics_model_name={dynamics_model_name}"
                )
    case "reaction_diffusion_default":
        # raise NotImplementedError("Reaction-diffusion system not implemented yet.")
        n_z = 4  # latent space dimension
        grayscale = False
        match dynamics_model_name:
            case "node-con-iae":
                experiment_id = f"2024-10-09_17-13-09/n_z_{n_z}_seed_{seed}"
                # the dataset doesn't consider inputs
                num_mlp_layers, mlp_hidden_dim = 0, 0
            case _:
                raise ValueError(
                    f"No experiment_id for dynamics_model_name={dynamics_model_name}"
                )


# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in [
    "node",
    "discrete",
    "ar",
], f"Unknown dynamics_type: {dynamics_type}"


ckpt_dir = (
    Path("logs").resolve() / f"{system_type}_dynamics_autoencoder" / experiment_id
)


if __name__ == "__main__":
    dynamics_order = 2
    if system_type in ["cc", "cs", "pcc_ns-2", "pcc_ns-3", "pcc_ns-4"]:
        dataset_type = "planar_pcs"
    elif system_type in ["single_pendulum", "double_pendulum"]:
        dataset_type = "pendulum"
    elif system_type == "reaction_diffusion_default":
        dataset_type = "reaction_diffusion"
        dynamics_order = 1
    elif system_type in [
        "mass_spring_friction",
        "mass_spring_friction_actuation",
        "pendulum_friction",
    ]:
        dataset_type = "toy_physics"
    elif system_type == "double_pendulum_friction":
        dataset_type = "toy_physics"
    else:
        raise ValueError(f"Unknown system_type: {system_type}")

    dataset_name_postfix = ""
    if not dataset_type in ["reaction_diffusion"]:
        if dataset_type == "toy_physics":
            dataset_name_postfix += f"_dt_0_05"
        else:
            dataset_name_postfix += f"_32x32px"
        if dataset_type != "toy_physics":
            dataset_name_postfix += f"_h-101"
    dataset_name = f"{dataset_type}/{system_type}{dataset_name_postfix}"
    if dataset_type == "toy_physics":
        load_dataset_type = "dm_hamiltonian_dynamics_suite"
    elif dataset_type == "reaction_diffusion":
        load_dataset_type = "reaction_diffusion"
    else:
        load_dataset_type = "jsrm"
    datasets, dataset_info, dataset_metadata = load_dataset(
        dataset_name,
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=grayscale,
        dataset_type=load_dataset_type,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # size of torques
    n_tau = train_ds.element_spec["tau"].shape[-1]  # dimension of the control input
    if system_type in ["reaction_diffusion_default"]:
        n_tau = 0
        print(f"n_tau: {n_tau}")
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape

    # initialize the neural networks
    if ae_type == "beta_vae":
        autoencoder_model = VAE(
            latent_dim=n_z, img_shape=img_shape, norm_layer=norm_layer
        )
    else:
        autoencoder_model = Autoencoder(
            latent_dim=n_z, img_shape=img_shape, norm_layer=norm_layer
        )
    state_dim = n_z if dynamics_order == 1 else 2 * n_z
    if dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp"]:
        dynamics_model = MlpOde(
            latent_dim=n_z,
            input_dim=n_tau,
            dynamics_order=dynamics_order,
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
            dynamics_order=dynamics_order,
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
            dynamics_order=dynamics_order,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
        )
    elif dynamics_model_name == "node-lnn":
        dynamics_model = LnnOde(
            latent_dim=n_z,
            input_dim=n_tau,
            dynamics_order=dynamics_order,
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
    elif dynamics_model_name == "ar-con-iae-cfa":
        dynamics_model = DiscreteConIaeCfaDynamics(
            latent_dim=n_z,
            input_dim=n_tau,
            dt=sim_dt,
            dynamics_order=dynamics_order,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
        )
    elif dynamics_model_name in ["ar-elman-rnn", "ar-gru-rnn"]:
        dynamics_model = DiscreteRnnDynamics(
            state_dim=state_dim,
            input_dim=n_tau,
            output_dim=state_dim,
            rnn_method=dynamics_model_name.split("-")[1],  # "elman" or "gru"
        )
    elif dynamics_model_name == "ar-cornn":
        dynamics_model = DiscreteCornn(
            latent_dim=n_z,
            input_dim=n_tau,
            dynamics_order=dynamics_order,
            dt=sim_dt,
            gamma=cornn_gamma,
            epsilon=cornn_epsilon,
        )
    else:
        raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")
    nn_model = DynamicsAutoencoder(
        autoencoder=autoencoder_model,
        dynamics=dynamics_model,
        dynamics_type=dynamics_type,
        dynamics_order=dynamics_order,
        num_past_timesteps=num_past_timesteps,
    )

    solver_class_name = dataset_metadata.get("solver_class", "Dopri5")
    # import solver class from diffrax
    # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    solver_class = getattr(
        __import__("diffrax", fromlist=[solver_class_name]),
        solver_class_name,
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
        dynamics_order=dynamics_order,
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

    num_rollouts = 10
    inference_forward_fn = jax.jit(
        partial(task_callables.forward_fn, nn_params=state.params)
    )
    for batch_idx, batch in enumerate(test_ds.as_numpy_iterator()):
        pred = inference_forward_fn(batch)

        ts = batch["t_ts"][0, start_time_idx:]
        img_pred_ts = pred["img_dynamic_ts"][0]
        img_target_ts = batch["rendering_ts"][0, start_time_idx:]

        if dataset_type == "reaction_diffusion":
            # add third channel that consists of zeros
            img_pred_ts = jnp.concatenate(
                [img_pred_ts, jnp.zeros_like(img_pred_ts[..., 0:1])], axis=-1
            )
            img_target_ts = jnp.concatenate(
                [img_target_ts, jnp.zeros_like(img_target_ts[..., 0:1])], axis=-1
            )

            # denormalize the images
            img_pred_ts = jax.vmap(partial(denormalize_img, apply_threshold=False))(
                img_pred_ts
            )
            img_target_ts = jax.vmap(partial(denormalize_img, apply_threshold=False))(
                img_target_ts
            )
        else:
            # denormalize the images
            img_pred_ts = jax.vmap(partial(denormalize_img, apply_threshold=True))(
                img_pred_ts
            )
            img_target_ts = jax.vmap(partial(denormalize_img, apply_threshold=True))(
                img_target_ts
            )

        # animate the rollout
        print(f"Animate rollout {batch_idx + 1} / {num_rollouts}...")
        animate_pred_vs_target_image_pyplot(
            onp.array(ts),
            img_pred_ts=img_pred_ts,
            img_target_ts=img_target_ts,
            filepath=ckpt_dir / f"rollout_{batch_idx}.mp4",
            skip_step=1,
            show=False,
            label_target="Ground-truth",
        )
        animate_image_cv2(
            onp.array(ts),
            onp.array(img_target_ts).astype(onp.uint8),
            filepath=ckpt_dir / f"rollout_{batch_idx}_target.mp4",
            skip_step=1,
        )
        animate_image_cv2(
            onp.array(ts),
            onp.array(img_pred_ts).astype(onp.uint8),
            filepath=ckpt_dir / f"rollout_{batch_idx}_pred.mp4",
            skip_step=1,
        )

        if batch_idx == num_rollouts - 1:
            break
