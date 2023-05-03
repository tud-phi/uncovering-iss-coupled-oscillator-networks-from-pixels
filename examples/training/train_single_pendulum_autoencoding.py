from datetime import datetime
from jax import random
from jax import config as jax_config
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from pathlib import Path
import tensorflow as tf


from src.neural_networks.simple_cnn import Autoencoder
from src.neural_networks.convnext import ConvNeXtAutoencoder
from src.tasks import autoencoding
from src.training.load_dataset import load_dataset
from src.training.loops import run_training, run_eval

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

use_wae = False
sigma_z = 1.0

num_epochs = 25

if use_wae:
    latent_dim = 3
    normalize_latent_space = False
    batch_size = 10
    loss_weights = dict(mse_q=0.0, mse_rec=5.0, mmd=1e-3)
    base_lr = 2e-3
    warmup_epochs = 5
else:
    latent_dim = 2
    normalize_latent_space = True
    batch_size = 8
    loss_weights = dict(mse_q=1.0, mse_rec=5.0)
    base_lr = 5e-3
    warmup_epochs = 3

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
    nn_model = Autoencoder(latent_dim=latent_dim, img_shape=img_shape)

    # call the factory function for the sensing task
    task_callables, metrics = autoencoding.task_factory(
        "pendulum",
        nn_model,
        loss_weights=loss_weights,
        normalize_latent_space=normalize_latent_space,
        use_wae=use_wae,
        sigma_z=sigma_z,
    )

    # run the training loop
    print("Run training...")
    (state, train_history,) = run_training(
        rng=rng,
        train_ds=train_ds,
        val_ds=val_ds,
        task_callables=task_callables,
        metrics=metrics,
        num_epochs=num_epochs,
        nn_model=nn_model,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=0.0,
        logdir=logdir,
    )
    print("Final training metrics:\n", state.metrics.compute())
