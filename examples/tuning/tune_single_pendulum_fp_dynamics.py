from datetime import datetime
import dill
from jax import random
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jsrm.integration import ode_factory
from jsrm.systems.pendulum import factory, normalize_joint_angles
import logging
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import tensorflow as tf

from src.neural_networks.simple_cnn import Autoencoder
from src.neural_networks.vae import VAE
from src.tasks import fp_dynamics
from src.training.callbacks import OptunaPruneCallback
from src.training.load_dataset import load_dataset
from src.training.loops import run_training, run_eval
import sys

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

ae_type = "wae"  # "None", "beta_vae", "wae"

max_num_epochs = 50
warmup_epochs = 5
batch_size = 100

now = datetime.now()
experiment_name = "single_pendulum_fp_dynamics"
datetime_str = f"{now:%Y-%m-%d_%H-%M-%S}"
study_id = f"study-{experiment_name}-{datetime_str}"  # Unique identifier of the study.
logdir = Path("logs") / experiment_name / datetime_str
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = Path("symbolic_expressions") / "single_pendulum.dill"

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        "mechanical_system/single_pendulum_64x64px",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # extract the robot parameters from the dataset
    robot_params = dataset_metadata["system_params"]
    print(f"Robot parameters: {robot_params}")

    # number of generalized coordinates
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    # latent space shape
    latent_dim = 2 * n_q
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]

    # get the dynamics function
    forward_kinematics_fn, dynamical_matrices_fn = factory(sym_exp_filepath)

    # initialize the model
    if ae_type == "beta_vae":
        nn_model = VAE(
            latent_dim=latent_dim,
            img_shape=img_shape,
        )
    else:
        nn_model = Autoencoder(latent_dim=latent_dim, img_shape=img_shape)

    # define the objective function for hyperparameter tuning
    def objective(trial):
        # Sample hyperparameters
        base_lr = trial.suggest_float("base_lr", 1e-3, 1e-2, log=True)
        # loss weights
        mse_q_weight = trial.suggest_float("mse_q_weight", 5e-3, 5e-1, log=True)
        mse_rec_static_weight = 1.0
        mse_rec_dynamic_weight = trial.suggest_float(
            "mse_rec_dynamic_weight", 1e0, 5e2, log=True
        )
        b1 = 0.9
        b2 = 0.999
        weight_decay = trial.suggest_float("weight_decay", 5e-6, 2e-4, log=True)
        # fp dynamics settings
        # configuration_velocity_source = trial.suggest_categorical(
        #     "configuration_velocity_source",
        #     ["direct-finite-differences", "image-space-finite-differences"],
        # )
        configuration_velocity_source = "direct-finite-differences" # works generally better 
        # initialize the loss weights
        loss_weights = dict(
            mse_q=mse_q_weight,
            mse_rec_static=mse_rec_static_weight,
            mse_rec_dynamic=mse_rec_dynamic_weight,
        )
        start_time_idx = trial.suggest_int("start_time_idx", 1, 7)
        if ae_type == "beta_vae":
            beta = trial.suggest_float("beta", 1e-4, 1e1, log=True)
            loss_weights["beta"] = beta
        elif ae_type == "wae":
            mmd = trial.suggest_float("mmd", 1e-4, 1e1, log=True)
            loss_weights["mmd"] = mmd

        # call the factory function for the task
        task_callables, metrics = fp_dynamics.task_factory(
            "pendulum",
            nn_model,
            ode_fn=ode_factory(
                dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))
            ),
            loss_weights=loss_weights,
            solver=dataset_metadata["solver_class"](),
            configuration_velocity_source=configuration_velocity_source,
            start_time_idx=start_time_idx,
            ae_type=ae_type,
        )

        # add the optuna prune callback
        prune_callback = OptunaPruneCallback(trial, metric_name="rmse_q_static_val")
        callbacks = [prune_callback]

        print(f"Running trial {trial.number}...")
        (state, history, elapsed) = run_training(
            rng=rng,
            train_ds=train_ds,
            val_ds=val_ds,
            task_callables=task_callables,
            metrics=metrics,
            num_epochs=max_num_epochs,
            nn_model=nn_model,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            callbacks=callbacks,
            logdir=None,
            show_pbar=True,
        )

        (
            val_loss_stps,
            val_rmse_q_static_stps,
            val_rmse_q_dynamic_stps,
            val_rmse_rec_static_stps,
            val_rmse_rec_dynamic_stps,
        ) = history.collect("loss_val", "rmse_q_static_val", "rmse_q_dynamic_val", "rmse_rec_static_val", "rmse_rec_dynamic_val")
        print(
            f"Trial {trial.number} finished after {elapsed.steps} training steps with "
            f"validation loss: {val_loss_stps[-1]:.5f}, rmse_q_static: {val_rmse_q_static_stps[-1]:.5f}, rmse_q_dynamic: {val_rmse_q_dynamic_stps[-1]:.5f}, "
            f"rmse_rec_static: {val_rmse_rec_static_stps[-1]:.5f}, and rmse_rec_dynamic: {val_rmse_rec_dynamic_stps[-1]:.5f}"
        )

        return val_rmse_q_static_stps[-1]

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
        objective, n_trials=250
    )  # Invoke optimization of the objective function.

    with open(logdir / "optuna_study.dill", "wb") as f:
        dill.dump(study, f)
