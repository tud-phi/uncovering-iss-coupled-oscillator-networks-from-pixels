import os
# restrict to using one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



from datetime import datetime
import dill
import flax.linen as nn
import jax

jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
import logging
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import tensorflow as tf
from src.models.autoencoders import Autoencoder, VAE
from src.models.discrete_forward_dynamics import (
    DiscreteConIaeCfaDynamics,
    DiscreteLssDynamics,
    DiscreteMambaDynamics,
    DiscreteMlpDynamics,
    DiscreteRnnDynamics,
)
from src.models.neural_odes import (
    ConOde,
    ConIaeOde,
    CornnOde,
    DconOde,
    LnnOde,
    LinearStateSpaceOde,
    MambaOde,
    MlpOde,
)
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.tasks import dynamics_autoencoder
from src.training.callbacks import OptunaPruneCallback
from src.training.dataset_utils import load_dataset
from src.training.loops import run_training
import sys

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

system_type = "pcc_ns-2"  # "cc", "cs", "pcc_ns-2", "pcc_ns-3"
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
""" dynamics_model_name in [
    "node-general-mlp", "node-mechanical-mlp", "node-mechanical-mlp-s", 
    "node-cornn", "node-con", "node-w-con", "node-con-iae", "node-dcon", "node-lnn", 
    "node-hippo-lss", "node-mamba",
    "discrete-mlp", "discrete-elman-rnn", "discrete-gru-rnn", "discrete-general-lss", "discrete-hippo-lss", "discrete-mamba",
    "dsim-con-iae-cfa"
]
"""
dynamics_model_name = "dsim-con-iae-cfa"
# latent space shape
n_z = 8
# simulation time step
sim_dt = 1e-2

# identify the number of segments
if system_type in ["cc", "cs"]:
    num_segments = 1
elif system_type.split("_")[0] == "pcc":
    num_segments = int(system_type.split("-")[-1])
else:
    raise ValueError(f"Unknown system_type: {system_type}")
print(f"Number of segments: {num_segments}")

# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in [
    "node",
    "discrete",
    "dsim",
], f"Unknown dynamics_type: {dynamics_type}"

max_num_epochs = 50
warmup_epochs = 5
batch_size = 80

now = datetime.now()
experiment_name = f"{system_type}_dynamics_autoencoder"
datetime_str = f"{now:%Y-%m-%d_%H-%M-%S}"
study_id = f"study-{experiment_name}-{datetime_str}"  # Unique identifier of the study.
logdir = Path("logs").resolve() / experiment_name / datetime_str
logdir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # define the objective function for hyperparameter tuning
    def objective(trial):
        # re-seed tensorflow
        tf.random.set_seed(seed=seed)

        # Sample hyperparameters
        base_lr = trial.suggest_float("base_lr", 1e-3, 2e-2, log=True)
        # loss weights
        mse_z_weight = trial.suggest_float("mse_z_weight", 1e-1, 5e-1, log=True)
        mse_rec_static_weight = 1.0
        mse_rec_dynamic_weight = trial.suggest_float(
            "mse_rec_dynamic_weight", 1e0, 1e2, log=True
        )
        weight_decay = trial.suggest_float("weight_decay", 5e-6, 5e-5, log=True)
        # latent_velocity_source = trial.suggest_categorical(
        #     "latent_velocity_source",
        #     ["latent-space-finite-differences", "image-space-finite-differences"],
        # )
        latent_velocity_source = "image-space-finite-differences"
        # initialize the loss weights
        loss_weights = dict(
            mse_z=mse_z_weight,
            mse_rec_static=mse_rec_static_weight,
            mse_rec_dynamic=mse_rec_dynamic_weight,
        )

        # start_time_idx = trial.suggest_int("start_time_idx", 1, 7)
        start_time_idx = 1
        num_past_timesteps = 2

        if ae_type == "beta_vae":
            beta = trial.suggest_float("beta", 1e-4, 1e-3, log=True)
            loss_weights["beta"] = beta
        elif ae_type == "wae":
            mmd = trial.suggest_float("mmd", 1e-4, 1e1, log=True)
            loss_weights["mmd"] = mmd

        datasets, dataset_info, dataset_metadata = load_dataset(
            f"planar_pcs/{system_type}_32x32px_h-101",
            seed=seed,
            batch_size=batch_size,
            normalize=True,
            grayscale=True,
        )
        train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

        # extract the robot parameters from the dataset
        robot_params = dataset_metadata["system_params"]
        print(f"Robot parameters: {robot_params}")

        # size of torques
        n_tau = train_ds.element_spec["tau"].shape[-1]  # dimension of the control input
        print(f"Control input dimension: {n_tau}")
        # image shape
        img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]

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
            if dynamics_model_name == "node-mechanical-mlp-s":
                # small mlp model
                num_mlp_layers, mlp_hidden_dim = 2, 24
                mlp_nonlinearity_name = "elu"
            else:
                num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 6)
                mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 96)
                mlp_nonlinearity_name = trial.suggest_categorical(
                    "mlp_nonlinearity",
                    ["tanh", "sigmoid", "softplus"],
                )
            mlp_nonlinearity = getattr(nn, mlp_nonlinearity_name)

            dynamics_model = MlpOde(
                latent_dim=n_z,
                input_dim=n_tau,
                num_layers=num_mlp_layers,
                hidden_dim=mlp_hidden_dim,
                nonlinearity=mlp_nonlinearity,
                mechanical_system=True
                if dynamics_model_name.split("-")[1] == "mechanical"
                else False,
            )
        elif dynamics_model_name == "node-cornn":
            cornn_gamma = trial.suggest_float("cornn_gamma", 1e-2, 1e2, log=True)
            cornn_epsilon = trial.suggest_float("cornn_epsilon", 1e-2, 1e2, log=True)
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
            # loss_weights["mse_tau_rec"] = trial.suggest_float("mse_tau_rec_weight", 1e-1, 1e3, log=True)
            loss_weights["mse_tau_rec"] = 1e1
            # num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 6)
            # mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 96)
            if dynamics_model_name == "node-con-iae-s":
                num_mlp_layers, mlp_hidden_dim = 2, 12
            else:
                num_mlp_layers, mlp_hidden_dim = 5, 30
            dynamics_model = ConIaeOde(
                latent_dim=n_z,
                input_dim=n_tau,
                num_layers=num_mlp_layers,
                hidden_dim=mlp_hidden_dim,
            )
        elif dynamics_model_name == "node-dcon":
            dcon_gamma = trial.suggest_float("dcon_gamma", 1e-2, 1e2, log=True)
            dcon_epsilon = trial.suggest_float("dcon_epsilon", 1e-2, 1e2, log=True)
            num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 6)
            mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 96)
            dynamics_model = DconOde(
                latent_dim=n_z,
                input_dim=n_tau,
                gamma=dcon_gamma,
                epsilon=dcon_epsilon,
                num_layers=num_mlp_layers,
                hidden_dim=mlp_hidden_dim,
            )
        elif dynamics_model_name == "node-lnn":
            learn_dissipation = trial.suggest_categorical(
                "learn_dissipation", [True, False]
            )
            num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 6)
            mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 72)
            # attention: we are not allowed to use non-continuous activation functions like ReLU, leaky ReLU, or ELU
            # the second derivative of the activation function must be continuous
            # otherwise, the second derivative of the potential is not continuous and with that the Hessian
            # of the potential is not symmetric
            mlp_nonlinearity_name = trial.suggest_categorical(
                "mlp_nonlinearity",
                ["tanh", "sigmoid", "softplus"],
            )
            diag_shift = trial.suggest_float("diag_shift", 1e-6, 1e-2, log=True)
            diag_eps = trial.suggest_float("diag_eps", 1e-6, 1e-2, log=True)
            dynamics_model = LnnOde(
                latent_dim=n_z,
                input_dim=n_tau,
                learn_dissipation=learn_dissipation,
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
        elif dynamics_model_name == "node-mamba":
            dynamics_model = MambaOde(
                latent_dim=n_z,
                input_dim=n_tau,
            )
        elif dynamics_model_name == "discrete-mlp":
            num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 6)
            mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 96)
            mlp_nonlinearity_name = trial.suggest_categorical(
                "mlp_nonlinearity",
                ["leaky_relu", "relu", "tanh", "sigmoid", "elu", "selu"],
            )
            mlp_nonlinearity = getattr(nn, mlp_nonlinearity_name)

            dynamics_model = DiscreteMlpDynamics(
                state_dim=num_past_timesteps * n_z,
                input_dim=num_past_timesteps * n_tau,
                output_dim=n_z,
                dt=dataset_metadata["dt"],
                num_layers=num_mlp_layers,
                hidden_dim=mlp_hidden_dim,
                nonlinearity=mlp_nonlinearity,
            )
        elif dynamics_model_name in ["discrete-elman-rnn", "discrete-gru-rnn"]:
            dynamics_model = DiscreteRnnDynamics(
                state_dim=num_past_timesteps * n_z,
                input_dim=num_past_timesteps * n_tau,
                output_dim=n_z,
                rnn_method=dynamics_model_name.split("-")[1],
            )
        elif dynamics_model_name in ["discrete-general-lss", "discrete-hippo-lss"]:
            dynamics_model = DiscreteLssDynamics(
                state_dim=num_past_timesteps * n_z,
                input_dim=num_past_timesteps * n_tau,
                output_dim=n_z,
                dt=dataset_metadata["dt"],
                transition_matrix_init=dynamics_model_name.split("-")[
                    1
                ],  # "general", or "hippo"
            )
        elif dynamics_model_name == "discrete-mamba":
            dynamics_model = DiscreteMambaDynamics(
                state_dim=num_past_timesteps * n_z,
                input_dim=num_past_timesteps * n_tau,
                output_dim=n_z,
                dt=dataset_metadata["dt"],
            )
        elif dynamics_model_name == "dsim-con-iae-cfa":
            loss_weights["mse_tau_rec"] = 1e1
            # num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 6)
            # mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 96)
            num_mlp_layers, mlp_hidden_dim = 5, 30
            sim_dt = trial.suggest_categorical("sim_dt", [1e-2, 5e-3, 2.5e-3])
            dynamics_model = DiscreteConIaeCfaDynamics(
                latent_dim=n_z,
                input_dim=n_tau,
                dt=sim_dt,
                num_layers=num_mlp_layers,
                hidden_dim=mlp_hidden_dim,
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

        # call the factory function for the task
        print(
            "Dataset dt:",
            dataset_metadata["dt"],
            "dataset sim_dt:",
            dataset_metadata["sim_dt"],
            "actually using sim_dt",
            sim_dt,
        )
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
        )

        # add the optuna prune callback
        prune_callback = OptunaPruneCallback(trial, metric_name="rmse_rec_dynamic_val")
        callbacks = [prune_callback]

        print(f"Running trial {trial.number}...")
        (state, history, elapsed) = run_training(
            rng=rng,
            train_ds=train_ds,
            val_ds=val_ds,
            task_callables=task_callables,
            metrics_collection_cls=metrics_collection_cls,
            num_epochs=max_num_epochs,
            nn_model=nn_model,
            init_fn=nn_model.forward_all_layers,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            weight_decay=weight_decay,
            callbacks=callbacks,
            logdir=None,
            show_pbar=True,
        )

        (
            val_loss_stps,
            val_rmse_rec_static_stps,
            val_rmse_rec_dynamic_stps,
        ) = history.collect(
            "loss_val",
            "rmse_rec_static_val",
            "rmse_rec_dynamic_val",
        )
        print(
            f"Trial {trial.number} finished after {elapsed.steps} training steps with "
            f"validation loss: {val_loss_stps[-1]:.5f}, "
            f"rmse_rec_static: {val_rmse_rec_static_stps[-1]:.5f}, and rmse_rec_dynamic: {val_rmse_rec_dynamic_stps[-1]:.5f}"
        )

        return val_rmse_rec_dynamic_stps[-1]

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage_name = f"sqlite:///{logdir}/optuna_study.db"
    sampler = TPESampler(seed=seed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(
        study_name=study_id,
        sampler=sampler,
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=500),
        storage=storage_name,
    )  # Create a new study.

    print(f"Run hyperparameter tuning with storage in {storage_name}...")
    study.optimize(
        objective, n_trials=1000
    )  # Invoke optimization of the objective function.

    with open(logdir / "optuna_study.dill", "wb") as f:
        dill.dump(study, f)
