from datetime import datetime
import dill
import flax.linen as nn
import jax

# jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
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
from src.tasks import dynamics_autoencoder
from src.training.dataset_utils import load_dataset, load_dummy_neural_network_input
from src.training.loops import run_eval, run_training
from src.training.train_state_utils import (
    count_number_of_trainable_params,
    restore_train_state,
)


def concat_or_none(x, y, **kwargs):
    if x is None:
        return y
    return onp.concatenate([x, y], **kwargs)


# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# set sweep settings
# set the range of latent dimensions
n_z_range = onp.arange(2, 34, 2)
# set the range of random seeds
seed_range = onp.array([0])

system_type = "nb-2"
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
""" dynamics_model_name in [
    "node-general-mlp", "node-mechanical-mlp", "node-mechanical-mlp-s", 
    "node-cornn", "node-con", "node-w-con", "node-con-iae",  "node-con-iae-s", "node-lnn", 
    "node-hippo-lss", "node-mamba",
    "discrete-mlp", "discrete-elman-rnn", "discrete-gru-rnn", "discrete-general-lss", "discrete-hippo-lss", "discrete-mamba",
]
"""
dynamics_model_name = "node-mechanical-mlp"
# simulation time step
sim_dt = 1e-2

batch_size = 100
num_epochs = 50
warmup_epochs = 5
start_time_idx = 1
num_past_timesteps = 2

latent_velocity_source = "image-space-finite-differences"
num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 20, "leaky_relu"
cornn_gamma, cornn_epsilon = 1.0, 1.0
lnn_learn_dissipation = True
diag_shift, diag_eps = 1e-6, 2e-6
if ae_type == "beta_vae":
    match dynamics_model_name:
        case "node-mechanical-mlp":
            # optimized for n_z=8
            base_lr = 0.019107078623753257
            loss_weights = dict(
                mse_z=0.386566816622383,
                mse_rec_static=1.0,
                mse_rec_dynamic=43.30612446330905,
                beta=0.0002863423459906223,
            )
            weight_decay = 1.2975197419490978e-05
            num_mlp_layers, mlp_hidden_dim = 4, 8
            mlp_nonlinearity_name = "tanh"
        case _:
            raise NotImplementedError(
                f"beta_vae with dynamics_model_name '{dynamics_model_name}' not implemented yet."
            )
else:
    raise NotImplementedError(f"ae_type '{ae_type}' not implemented yet.")


# identify the number of bodies
num_bodies = int(system_type.split("-")[-1])
print(f"Number of segments: {num_bodies}")

# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in ["node", "discrete"], f"Unknown dynamics_type: {dynamics_type}"

now = datetime.now()
logdir = (
    Path("logs").resolve()
    / f"{system_type}_dynamics_autoencoder"
    / f"{now:%Y-%m-%d_%H-%M-%S}"
)
logdir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # initialize dictionary with experimental results
    zero_array = None
    sweep_results = dict(
        n_z=zero_array,
        seed=zero_array,
        num_trainable_params=dict(
            total=zero_array,
            autoencoder=zero_array,
            dynamics=zero_array,
        ),
        train=dict(
            rmse_rec_static=zero_array,
            rmse_rec_dynamic=zero_array,
            psnr_rec_static=zero_array,
            psnr_rec_dynamic=zero_array,
            ssim_rec_static=zero_array,
            ssim_rec_dynamic=zero_array,
        ),
        test=dict(
            rmse_rec_static=zero_array,
            rmse_rec_dynamic=zero_array,
            psnr_rec_static=zero_array,
            psnr_rec_dynamic=zero_array,
            ssim_rec_static=zero_array,
            ssim_rec_dynamic=zero_array,
        ),
    )
    for n_z in n_z_range:
        for seed in seed_range:
            # initialize the pseudo-random number generator
            rng = random.PRNGKey(seed=seed)
            tf.random.set_seed(seed=seed)

            # specify the folder
            logdir_run = logdir / f"n_z_{n_z}_seed_{seed}"

            datasets, dataset_info, dataset_metadata = load_dataset(
                f"n_body_problem/{system_type}_h-101_32x32px",
                seed=seed,
                batch_size=batch_size,
                normalize=True,
                grayscale=True,
            )
            train_ds, val_ds, test_ds = (
                datasets["train"],
                datasets["val"],
                datasets["test"],
            )

            # extract the robot parameters from the dataset
            robot_params = dataset_metadata["system_params"]
            # size of torques
            n_tau = train_ds.element_spec["tau"].shape[
                -1
            ]  # dimension of the control input=
            # image shape
            img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape

            # initialize the neural networks
            if ae_type == "beta_vae":
                autoencoder_model = VAE(
                    latent_dim=n_z, img_shape=img_shape, norm_layer=nn.LayerNorm
                )
            else:
                autoencoder_model = Autoencoder(
                    latent_dim=n_z, img_shape=img_shape, norm_layer=nn.LayerNorm
                )
            if dynamics_model_name in [
                "node-general-mlp",
                "node-mechanical-mlp",
                "node-mechanical-mlp-s",
            ]:
                dynamics_model = MlpOde(
                    latent_dim=n_z,
                    input_dim=n_tau,
                    num_layers=num_mlp_layers,
                    hidden_dim=mlp_hidden_dim,
                    nonlinearity=getattr(nn, mlp_nonlinearity_name),
                    mechanical_system=True
                    if dynamics_model_name.split("-")[1] == "mechanical"
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
            solver_class = __import__("diffrax", fromlist=[dataset_metadata.get("solver_class", "Dopri5")])

            # call the factory function for the dynamics autoencoder task
            task_callables, metrics_collection_cls = dynamics_autoencoder.task_factory(
                system_type,
                nn_model,
                ts=dataset_metadata["ts"],
                sim_dt=sim_dt,
                loss_weights=loss_weights,
                ae_type=ae_type,
                dynamics_type=dynamics_type,
                start_time_idx=start_time_idx,
                solver=solver_class(),
                latent_velocity_source=latent_velocity_source,
                num_past_timesteps=num_past_timesteps,
                compute_psnr=True,
                compute_ssim=True,
            )

            # run the training loop
            print(f"Run training for n_z={n_z}, seed={seed}...")
            (state, train_history, elapsed) = run_training(
                rng=rng,
                train_ds=train_ds,
                val_ds=val_ds,
                task_callables=task_callables,
                metrics_collection_cls=metrics_collection_cls,
                num_epochs=num_epochs,
                nn_model=nn_model,
                init_fn=nn_model.forward_all_layers,
                base_lr=base_lr,
                warmup_epochs=warmup_epochs,
                weight_decay=weight_decay,
                logdir=logdir_run,
            )
            train_metrics = state.metrics.compute()
            print(
                f"Final training metrics for n_z={n_z}, seed={seed}:\n", train_metrics
            )

            # count the number of trainable parameters
            params_count = count_number_of_trainable_params(state, verbose=False)

            # load the neural network dummy input
            nn_dummy_input = load_dummy_neural_network_input(test_ds, task_callables)
            # load the training state from the checkpoint directory
            state = restore_train_state(
                rng=rng,
                ckpt_dir=logdir_run,
                nn_model=nn_model,
                nn_dummy_input=nn_dummy_input,
                metrics_collection_cls=metrics_collection_cls,
                init_fn=nn_model.forward_all_layers,
            )

            print(f"Run testing for n_z={n_z}, seed={seed}...")
            state, test_history = run_eval(test_ds, state, task_callables)
            test_metrics = state.metrics.compute()
            print(
                "\n"
                f"Final test metrics for n_z={n_z}, seed={seed}:\n"
                f"rmse_rec_static={test_metrics['rmse_rec_static']:.4f}, "
                f"rmse_rec_dynamic={test_metrics['rmse_rec_dynamic']:.4f}, "
                f"psnr_rec_static={test_metrics['psnr_rec_static']:.4f}, "
                f"psnr_rec_dynamic={test_metrics['psnr_rec_dynamic']:.4f}, "
                f"ssim_rec_static={test_metrics['ssim_rec_static']:.4f}, "
                f"ssim_rec_dynamic={test_metrics['ssim_rec_dynamic']:.4f}"
            )

            # update sweep results
            sweep_results["n_z"] = concat_or_none(
                sweep_results["n_z"], onp.array(n_z)[None, ...], axis=0
            )
            sweep_results["seed"] = concat_or_none(
                sweep_results["seed"], onp.array(seed)[None, ...], axis=0
            )
            sweep_results["num_trainable_params"]["total"] = concat_or_none(
                sweep_results["num_trainable_params"]["total"],
                onp.array(params_count["total"])[None, ...],
                axis=0,
            )
            sweep_results["num_trainable_params"]["autoencoder"] = concat_or_none(
                sweep_results["num_trainable_params"]["autoencoder"],
                onp.array(params_count["autoencoder"])[None, ...],
                axis=0,
            )
            sweep_results["num_trainable_params"]["dynamics"] = concat_or_none(
                sweep_results["num_trainable_params"]["dynamics"],
                onp.array(params_count["dynamics"])[None, ...],
                axis=0,
            )
            sweep_results["train"]["rmse_rec_static"] = concat_or_none(
                sweep_results["train"]["rmse_rec_static"],
                train_metrics["rmse_rec_static"][None, ...],
                axis=0,
            )
            sweep_results["train"]["rmse_rec_dynamic"] = concat_or_none(
                sweep_results["train"]["rmse_rec_dynamic"],
                train_metrics["rmse_rec_dynamic"][None, ...],
                axis=0,
            )
            sweep_results["train"]["psnr_rec_static"] = concat_or_none(
                sweep_results["train"]["psnr_rec_static"],
                train_metrics["psnr_rec_static"][None, ...],
                axis=0,
            )
            sweep_results["train"]["psnr_rec_dynamic"] = concat_or_none(
                sweep_results["train"]["psnr_rec_dynamic"],
                train_metrics["psnr_rec_dynamic"][None, ...],
                axis=0,
            )
            sweep_results["train"]["ssim_rec_static"] = concat_or_none(
                sweep_results["train"]["ssim_rec_static"],
                train_metrics["ssim_rec_static"][None, ...],
                axis=0,
            )
            sweep_results["train"]["ssim_rec_dynamic"] = concat_or_none(
                sweep_results["train"]["ssim_rec_dynamic"],
                train_metrics["ssim_rec_dynamic"][None, ...],
                axis=0,
            )
            sweep_results["test"]["rmse_rec_static"] = concat_or_none(
                sweep_results["test"]["rmse_rec_static"],
                test_metrics["rmse_rec_static"][None, ...],
                axis=0,
            )
            sweep_results["test"]["rmse_rec_dynamic"] = concat_or_none(
                sweep_results["test"]["rmse_rec_dynamic"],
                test_metrics["rmse_rec_dynamic"][None, ...],
                axis=0,
            )
            sweep_results["test"]["psnr_rec_static"] = concat_or_none(
                sweep_results["test"]["psnr_rec_static"],
                test_metrics["psnr_rec_static"][None, ...],
                axis=0,
            )
            sweep_results["test"]["psnr_rec_dynamic"] = concat_or_none(
                sweep_results["test"]["psnr_rec_dynamic"],
                test_metrics["psnr_rec_dynamic"][None, ...],
                axis=0,
            )
            sweep_results["test"]["ssim_rec_static"] = concat_or_none(
                sweep_results["test"]["ssim_rec_static"],
                test_metrics["ssim_rec_static"][None, ...],
                axis=0,
            )
            sweep_results["test"]["ssim_rec_dynamic"] = concat_or_none(
                sweep_results["test"]["ssim_rec_dynamic"],
                test_metrics["ssim_rec_dynamic"][None, ...],
                axis=0,
            )

            # save the experimental results
            with open(logdir / "sweep_results.dill", "wb") as f:
                dill.dump(sweep_results, f)
