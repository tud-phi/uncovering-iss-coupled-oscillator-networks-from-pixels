from jax import random
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
import tensorflow as tf

from src.neural_networks.simple_cnn import Autoencoder
from src.training.tasks import autoencoding
from src.training.load_dataset import load_dataset
from src.training.loops import run_training

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

num_epochs = 25
batch_size = 8
base_lr = 5e-4
warmup_epochs = 2
loss_weights = dict(mse_q=1.0, mse_rec=5.0)

if __name__ == "__main__":
    datasets = load_dataset(
        "mechanical_system/double_pendulum_64x64px",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # initialize the model
    nn_model = Autoencoder(latent_dim=2, img_shape=(64, 64, 1))

    # call the factory function for the sensing task
    task_callables = autoencoding.task_factory(nn_model, loss_weights=loss_weights)

    # run the training loop
    (
        val_loss_history,
        train_metrics_history,
        val_metrics_history,
        best_state,
    ) = run_training(
        rng=rng,
        train_ds=train_ds,
        val_ds=val_ds,
        nn_model=nn_model,
        task_callables=task_callables,
        num_epochs=num_epochs,
        batch_size=batch_size,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=0.0,
        verbose=True,
    )

    print("Final validation metrics:\n", val_metrics_history[-1])

    test_batch = next(test_ds.as_numpy_iterator())
    test_preds = task_callables.predict_fn(test_batch, best_state.params)

    import matplotlib.pyplot as plt

    for i in range(test_batch["x_ts"].shape[0]):
        print("test sample:", i)
        n_q = test_batch["x_ts"].shape[-1] // 2
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
