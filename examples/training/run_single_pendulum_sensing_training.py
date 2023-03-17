from datetime import datetime
from jax import random
from jax import config as jax_config
import jax.numpy as jnp
from pathlib import Path
import tensorflow as tf


from src.neural_networks.simple_cnn import Encoder
from src.training.tasks import sensing
from src.training.load_dataset import load_dataset
from src.training.loops import run_training, run_eval

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

num_epochs = 25
batch_size = 8
base_lr = 5e-4
warmup_epochs = 2

now = datetime.now()
logdir = Path("logs") / "single_pendulum_sensing" / f"{now:%Y-%m-%d_%H-%M-%S}"
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
    nn_model = Encoder(
        latent_dim=n_q,
        img_shape=img_shape
    )

    # call the factory function for the sensing task
    task_callables, metrics = sensing.task_factory("pendulum", nn_model)

    # run the training loop
    print("Run training...")
    (state, train_history,) = run_training(
        rng=rng,
        train_ds=train_ds,
        val_ds=val_ds,
        nn_model=nn_model,
        task_callables=task_callables,
        metrics=metrics,
        num_epochs=num_epochs,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=0.0,
        logdir=logdir,
    )
    print("Final training metrics:\n", state.metrics.compute())

    print("Run testing...")
    test_history = run_eval(test_ds, state, task_callables)
    rmse_q_stps = train_history.collect("rmse_q")
    print(f"Final test metrics: rmse_q={rmse_q_stps[-1]:.3f}")
