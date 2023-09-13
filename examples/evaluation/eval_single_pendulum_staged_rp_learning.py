from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import Array, jacfwd, jacrev, random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_factory
from jsrm.systems import pendulum
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

from src.autoencoders.simple_cnn import Autoencoder
from src.autoencoders.staged_autoencoder import StagedAutoencoder
from src.training.load_dataset import load_dataset
from src.training.loops import run_eval
from src.tasks import fp_dynamics
from src.training.train_state_utils import restore_train_state

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

latent_dim = 4
config_dim = 2

batch_size = 8
loss_weights = dict(mse_q=0.0, mse_rec_static=5.0, mse_rec_dynamic=5.0)

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"pendulum_nl-1.dill"
)
ckpt_dir = (
    Path("logs")
    / "single_pendulum_staged_rp_learning"
    / "2023-05-06_18-02-19"
    / "dynamic_learning"
)


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
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    # initialize the model
    backbone = Autoencoder(latent_dim=latent_dim, img_shape=img_shape)
    nn_model = StagedAutoencoder(backbone=backbone, config_dim=2 * n_q)

    # call the factory function for the first-principle dynamics task
    task_callables, metrics_collection_cls = fp_dynamics.task_factory(
        "pendulum",
        nn_model,
        ts=dataset_metadata["ts"],
        sim_dt=dataset_metadata["sim_dt"],
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))),
        loss_weights=loss_weights,
        solver=dataset_metadata["solver_class"](),
        configuration_velocity_source="direct-finite-differences",
    )

    state = restore_train_state(rng, ckpt_dir, nn_model, metrics_collection_cls)

    print("Run testing...")
    state, test_history = run_eval(test_ds, state, task_callables)

    test_metrics = state.metrics.compute()
    print(
        "\n"
        f"Final test metrics: "
        f"rmse_q_static={test_metrics['rmse_q_static']:.4f}, "
        f"rmse_q_dynamic={test_metrics['rmse_q_dynamic']:.4f}, "
        f"rmse_rec_static={test_metrics['rmse_rec_static']:.4f}, "
        f"rmse_rec_dynamic={test_metrics['rmse_rec_dynamic']:.4f}"
    )

    test_batch = next(test_ds.as_numpy_iterator())
    test_preds = task_callables.forward_fn(test_batch, state.params)

    for i in range(test_batch["x_ts"].shape[0]):
        print("Trajectory:", i)
        for t in range(test_batch["x_ts"].shape[1]):
            print("Time step:", t)
            q_gt = test_batch["x_ts"][i, t, :n_q] / jnp.pi * 180
            q_pred = test_preds["q_dynamic_ts"][i, t, :n_q] / jnp.pi * 180
            error_q = pendulum.normalize_joint_angles(
                test_preds["q_dynamic_ts"][i, t, :n_q] - test_batch["x_ts"][i, t, :n_q]
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

            img_gt = (128 * (1.0 + test_batch["rendering_ts"][i, t])).astype(jnp.uint8)
            img_rec = (128 * (1.0 + test_preds["rendering_dynamic_ts"][i, t])).astype(
                jnp.uint8
            )

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            img_gt_plot = axes[0].imshow(img_gt, vmin=0, vmax=255)
            plt.colorbar(img_gt_plot, ax=axes[0])
            axes[0].set_title("Original")
            img_rec_plot = axes[1].imshow(img_rec, vmin=0, vmax=255)
            plt.colorbar(img_rec_plot, ax=axes[1])
            axes[1].set_title("Reconstruction")
            plt.show()
