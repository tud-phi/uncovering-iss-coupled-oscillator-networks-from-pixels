from datetime import datetime
from jax import random
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from pathlib import Path
import tensorflow as tf

from src.autoencoders.simple_cnn import Autoencoder
from src.tasks import autoencoding
from src.training.load_dataset import load_dataset
from src.training.loops import run_training, run_eval

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

num_epochs = 25
batch_size = 8
base_lr = 5e-4
warmup_epochs = 2
loss_weights = dict(mse_q=1.0, mse_rec=5.0)

now = datetime.now()
logdir = Path("logs") / "double_pendulum_autoencoding" / f"{now:%Y-%m-%d_%H-%M-%S}"
logdir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        "mechanical_system/double_pendulum_64x64px",
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
    nn_model = Autoencoder(latent_dim=n_q, img_shape=img_shape)

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = autoencoding.task_factory(
        "pendulum", nn_model, loss_weights=loss_weights
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
        weight_decay=0.0,
        logdir=logdir,
    )
    print("Final training metrics:\n", state.metrics.compute())

    print("Run testing...")
    state, test_history = run_eval(test_ds, state, task_callables)
    rmse_q_stps, rmse_rec_stps = train_history.collect("rmse_q", "rmse_rec")
    print(
        f"Final test metrics: rmse_q={rmse_q_stps[-1]:.3f}, rmse_rec={rmse_rec_stps[-1]:.3f}"
    )

    test_batch = next(test_ds.as_numpy_iterator())
    test_preds = task_callables.forward_fn(test_batch, state.params)

    import matplotlib.pyplot as plt

    for i in range(test_batch["x_ts"].shape[0]):
        print("test sample:", i)
        q_gt = test_batch["x_ts"][i, 0, :n_q] / jnp.pi * 180
        q_pred = test_preds["q_ts"][i, 0, :n_q] / jnp.pi * 180
        error_q = normalize_joint_angles(
            test_preds["q_ts"][i, 0, :n_q] - test_batch["x_ts"][i, 0, :n_q]
        )
        print(
            "Ground-truth q:",
            q_gt,
            "deg",
            "Predicted q:",
            q_pred,
            "deg",
            "Error:",
            error_q / jnp.pi * 180,
            "deg",
        )

        img_gt = (128 * (1.0 + test_batch["rendering_ts"][i, 0])).astype(jnp.uint8)
        img_rec = (128 * (1.0 + test_preds["rendering_ts"][i, 0])).astype(jnp.uint8)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        img_gt_plot = axes[0].imshow(img_gt, vmin=0, vmax=255)
        plt.colorbar(img_gt_plot, ax=axes[0])
        axes[0].set_title("Original")
        img_rec_plot = axes[1].imshow(img_rec, vmin=0, vmax=255)
        plt.colorbar(img_rec_plot, ax=axes[1])
        axes[1].set_title("Reconstruction")
        plt.show()
