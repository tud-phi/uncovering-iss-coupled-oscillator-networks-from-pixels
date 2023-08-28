from datetime import datetime
from jax import random
from jax import config as jax_config
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from pathlib import Path
import optuna
import tensorflow as tf

from src.neural_networks.convnext import ConvNeXtAutoencoder
from src.neural_networks.simple_cnn import Autoencoder
from src.neural_networks.vae import VAE
from src.tasks import autoencoding
from src.training.load_dataset import load_dataset
from src.training.loops import run_training, run_eval

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

ae_type = "beta_vae"

latent_dim = 2
normalize_latent_space = True
num_epochs = 50
batch_size = 30

now = datetime.now()
logdir = Path("logs") / "single_pendulum_autoencoding" / f"{now:%Y-%m-%d_%H-%M-%S}"
logdir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        "mechanical_system/single_pendulum_64x64px",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # dimension of the latent space
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]

    # initialize the model
    if ae_type == "beta_vae":
        nn_model = VAE(latent_dim=latent_dim, img_shape=img_shape)
    else:
        nn_model = Autoencoder(latent_dim=latent_dim, img_shape=img_shape)

    # run the training loop
    print("Run training...")
    def objective(trial):
        # Sample hyperparameters
        warmup_epochs = 5
        base_lr = trial.suggest_float("base_lr", 1e-5, 1e-2, log=True)
        beta = trial.suggest_float("beta", 1e-5, 1e1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

        train_loss_weights = dict(mse_q=0.0, mse_rec=1.0, beta=beta)
        val_loss_weights = dict(mse_q=0.0, mse_rec=1.0, beta=1e0)

        if ae_type != "beta_vae":
            raise ValueError("Only beta_vae is supported for now")

        # call the factory function for the sensing task
        train_task_callables, train_metrics = autoencoding.task_factory(
            "pendulum",
            nn_model,
            loss_weights=train_loss_weights,
            normalize_latent_space=normalize_latent_space,
            # weight_on_foreground=0.15,
            ae_type=ae_type,
            eval=False
        )

        (
            state,
            train_history,
        ) = run_training(
            rng=rng,
            train_ds=train_ds,
            val_ds=val_ds,
            task_callables=train_task_callables,
            metrics=train_metrics,
            num_epochs=num_epochs,
            nn_model=nn_model,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            weight_decay=weight_decay,
            logdir=None,
            show_pbar=False
        )

        # run validation
        val_task_callables, val_metrics = autoencoding.task_factory(
            "pendulum",
            nn_model,
            loss_weights=val_loss_weights,
            normalize_latent_space=normalize_latent_space,
            ae_type=ae_type,
            eval=True
        )
        val_history = run_eval(val_ds, state, val_task_callables, show_pbar=False)
        val_loss_stps, val_rmse_rec_stps = val_history.collect("loss", "rmse_rec")

        print(f"Trial {trial.number} finished with validation loss: {val_loss_stps[-1]}, rmse_rec: {val_rmse_rec_stps[-1]}")

        return val_loss_stps[-1]
    

    study = optuna.create_study()  # Create a new study.
    study.optimize(objective, n_trials=2)  # Invoke optimization of the objective function.
