from datetime import datetime
import flax.linen as nn
from jax import random
from jax import config as jax_config
import jax.numpy as jnp
from pathlib import Path
import tensorflow as tf

from src.autoencoders.simple_cnn import Autoencoder
from src.autoencoders.vae import VAE
from src.tasks import autoencoding
from src.training.load_dataset import load_dataset
from src.training.loops import run_training
from src.visualization.latent_space import (
    visualize_mapping_from_configuration_to_latent_space,
)

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

system_type = "cc"
ae_type = "beta_vae"
encourage_time_alignment = True

if system_type == "cc":
    latent_dim = 1
elif system_type == "pcc_ns-2":
    latent_dim = 2
elif system_type == "pcc_ns-3":
    latent_dim = 3
else:
    raise ValueError(f"Unknown system type: {system_type}!")

batch_size = 100
num_epochs = 50
warmup_epochs = 5
weight_decay = 0.0
conv_strides = (1, 1)

if ae_type == "wae":
    loss_weights = dict(mse_q=0.0, mse_rec=1.0, mmd=1e-1)
    base_lr = 5e-3
    warmup_epochs = 5
elif ae_type == "beta_vae":
    loss_weights = dict(mse_q=0.0, mse_rec=1.0, beta=5e-3)
    if encourage_time_alignment:
        loss_weights["time_alignment"] = 2e1
    base_lr = 1e-3
    weight_decay = 1e-4
elif ae_type == "triplet":
    loss_weights = dict(mse_q=0.0, mse_rec=1.0, triplet=1e2)
    base_lr = 2e-3
else:
    loss_weights = dict(mse_q=0.2, mse_rec=1.0)
    base_lr = 2e-3

now = datetime.now()
logdir = Path("logs") / f"{system_type}_autoencoding" / f"{now:%Y-%m-%d_%H-%M-%S}"
logdir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        f"planar_pcs/{system_type}_64x64px",
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
        nn_model = VAE(
            latent_dim=latent_dim,
            img_shape=img_shape,
            strides=conv_strides,
            norm_layer=nn.LayerNorm,
        )
    else:
        nn_model = Autoencoder(
            latent_dim=latent_dim,
            img_shape=img_shape,
            strides=conv_strides,
            norm_layer=nn.LayerNorm,
        )

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = autoencoding.task_factory(
        system_type,
        nn_model,
        loss_weights=loss_weights,
        ae_type=ae_type,
        margin=1e-2
    )

    # run the training loop
    print("Run training...")
    (state, train_history, elapsed) = run_training(
        rng=rng,
        train_ds=train_ds,
        val_ds=val_ds,
        task_callables=task_callables,
        metrics_collection_cls=metrics_collection_cls,
        num_epochs=num_epochs,
        nn_model=nn_model,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        logdir=logdir,
    )
    print("Final training metrics:\n", state.metrics.compute())

    visualize_mapping_from_configuration_to_latent_space(
        test_ds, state, task_callables, rng=rng
    )
