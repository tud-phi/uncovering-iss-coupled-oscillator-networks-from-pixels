from datetime import datetime
import dill
from jax import random
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems.pendulum import factory
import logging
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import tensorflow as tf

from src.autoencoders.simple_cnn import Autoencoder
from src.autoencoders.vae import VAE
from src.tasks import fp_dynamics_sindy_loss
from src.training.callbacks import OptunaPruneCallback
from src.training.dataset_utils import load_dataset
from src.training.loops import run_training
import sys

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

ae_type = "None"  # "None", "beta_vae", "wae"

max_num_epochs = 40
warmup_epochs = 5
batch_size = 25

now = datetime.now()
experiment_name = "single_pendulum_fp_dynamics_sindy_loss"
datetime_str = f"{now:%Y-%m-%d_%H-%M-%S}"
study_id = f"study-{experiment_name}-{datetime_str}"  # Unique identifier of the study.
logdir = Path("logs").resolve() / experiment_name / datetime_str
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"pendulum_nl-1.dill"
)

if __name__ == "__main__":
    # get the dynamics function
    forward_kinematics_fn, dynamical_matrices_fn = factory(sym_exp_filepath)

    # define the objective function for hyperparameter tuning
    def objective(trial):
        # re-seed tensorflow
        tf.random.set_seed(seed=seed)

        # Sample hyperparameters
        base_lr = trial.suggest_float("base_lr", 1e-4, 1e-2, log=True)
        # loss weights
        mse_rec_weight = 1.0
        mse_sindy_q_dd_weight = trial.suggest_float(
            "mse_sindy_q_dd_weight", 1e-7, 1e-1, log=True
        )
        mse_sindy_rendering_dd_weight = 0.0
        """
        mse_sindy_rendering_dd_weight = trial.suggest_float(
            "mse_sindy_rendering_dd_weight", 1e-10, 1e-5, log=True
        )
        """
        b1 = 0.9
        b2 = 0.999
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
        # initialize the loss weights
        loss_weights = dict(
            mse_rec=mse_rec_weight,
            mse_sindy_q_dd=mse_sindy_q_dd_weight,
            mse_sindy_rendering_dd=mse_sindy_rendering_dd_weight,
        )
        if ae_type == "beta_vae":
            beta = trial.suggest_float("beta", 1e-4, 1e1, log=True)
            loss_weights["beta"] = beta
        elif ae_type == "wae":
            mmd = trial.suggest_float("mmd", 1e-4, 1e1, log=True)
            loss_weights["mmd"] = mmd

        datasets, dataset_info, dataset_metadata = load_dataset(
            "pendulum/single_pendulum_24x24px",
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

        # initialize the model
        if ae_type == "beta_vae":
            nn_model = VAE(
                latent_dim=latent_dim,
                img_shape=img_shape,
            )
        else:
            nn_model = Autoencoder(latent_dim=latent_dim, img_shape=img_shape)

        # call the factory function for the task
        task_callables, metrics_collection_cls = fp_dynamics_sindy_loss.task_factory(
            "pendulum",
            nn_model,
            ts=dataset_metadata["ts"],
            ode_fn=ode_with_forcing_factory(dynamical_matrices_fn, robot_params),
            loss_weights=loss_weights,
            ae_type=ae_type,
        )

        # add the optuna prune callback
        prune_callback = OptunaPruneCallback(trial, metric_name="rmse_q_dd_val")
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
            val_rmse_rec_stps,
            val_rmse_q_stps,
            val_rmse_q_mirror_stps,
            val_rmse_q_d_stps,
            val_rmse_q_dd_stps,
        ) = history.collect(
            "loss_val",
            "rmse_rec_val",
            "rmse_q_val",
            "rmse_q_mirror_val",
            "rmse_q_d_val",
            "rmse_q_dd_val"
        )
        print(
            f"Trial {trial.number} finished after {elapsed.steps} training steps with "
            f"validation loss: {val_loss_stps[-1]:.5f}, rmse_rec: {val_rmse_rec_stps[-1]:.5f}, "
            f"rmse_q: {val_rmse_q_stps[-1]:.5f}, rmse_q_mirror: {val_rmse_q_mirror_stps[-1]:.5f}, "
            f"rmse_q_d: {val_rmse_q_d_stps[-1]:.5f}, and rmse_q_dd: {val_rmse_q_dd_stps[-1]:.5f}"
        )

        return val_rmse_q_dd_stps[-1]

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
