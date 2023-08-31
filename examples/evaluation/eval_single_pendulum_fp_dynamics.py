from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import Array, jacfwd, jacrev, random
import jax.numpy as jnp
from jsrm.integration import ode_factory
from jsrm.systems import pendulum
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

from src.autoencoders.simple_cnn import Autoencoder
from src.autoencoders.vae import VAE
from src.training.load_dataset import load_dataset
from src.training.loops import run_eval
from src.tasks import fp_dynamics
from src.training.train_state_utils import restore_train_state
from src.visualization.latent_space import (
    visualize_mapping_from_configuration_to_latent_space,
)

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

batch_size = 10
loss_weights = dict(mse_q=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0)
start_time_idx = 1
ae_type = "beta_vae"  # "None", "beta_vae", "wae"

sym_exp_filepath = Path("symbolic_expressions") / "single_pendulum.dill"
ckpt_dir = Path("logs") / "single_pendulum_fp_dynamics" / "2023-08-31_14-10-16"


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
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape

    # get the dynamics function
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    # initialize the model
    if ae_type == "beta_vae":
        nn_model = VAE(
            latent_dim=latent_dim,
            img_shape=img_shape,
        )
    else:
        nn_model = Autoencoder(latent_dim=latent_dim, img_shape=img_shape)

    # call the factory function for the sensing task
    task_callables, metrics = fp_dynamics.task_factory(
        "pendulum",
        nn_model,
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))),
        loss_weights=loss_weights,
        solver=dataset_metadata["solver_class"](),
        start_time_idx=start_time_idx,
        configuration_velocity_source="direct-finite-differences",
    )

    state = restore_train_state(rng, ckpt_dir, nn_model, metrics)

    print("Run testing...")
    test_history = run_eval(test_ds, state, task_callables)
    (
        rmse_q_static_stps,
        rmse_q_dynamic_stps,
        rmse_rec_static_stps,
        rmse_rec_dynamic_stps,
    ) = test_history.collect(
        "rmse_q_static", "rmse_q_dynamic", "rmse_rec_static", "rmse_rec_dynamic"
    )
    print(
        "\n"
        f"Final test metrics: rmse_q_static_stps={rmse_q_static_stps[-1]:.3f}, "
        f"rmse_q_dynamic_stps={rmse_q_dynamic_stps[-1]:.3f}, "
        f"rmse_rec_static_stps={rmse_rec_static_stps[-1]:.3f}, "
        f"rmse_rec_dynamic_stps={rmse_rec_dynamic_stps[-1]:.3f}"
    )

    visualize_mapping_from_configuration_to_latent_space(
        test_ds, state, task_callables, rng=rng
    )

    test_batch = next(test_ds.as_numpy_iterator())
    test_preds = task_callables.forward_fn(test_batch, state.params)

    for i in range(test_batch["x_ts"].shape[0]):
        print("Trajectory:", i)

        print(
            f"Estimated initial velocity: {test_preds['x_dynamic_ts'][i, 0, n_q:]}rad/s, actual initial velocity: {test_batch['x_ts'][i, start_time_idx, n_q:]}rad/s"
        )

        for t in range(start_time_idx, test_batch["x_ts"].shape[1]):
            print("Time step:", t)
            q_gt = test_batch["x_ts"][i, t, :n_q] / jnp.pi * 180
            q_pred = (
                test_preds["q_dynamic_ts"][i, t - start_time_idx, :n_q] / jnp.pi * 180
            )
            error_q = pendulum.normalize_joint_angles(
                test_preds["q_dynamic_ts"][i, t - start_time_idx, :n_q]
                - test_batch["x_ts"][i, t, :n_q]
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
            img_rec = (
                128 * (1.0 + test_preds["rendering_dynamic_ts"][i, t - start_time_idx])
            ).astype(jnp.uint8)

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            img_gt_plot = axes[0].imshow(img_gt, vmin=0, vmax=255)
            plt.colorbar(img_gt_plot, ax=axes[0])
            axes[0].set_title("Original")
            img_rec_plot = axes[1].imshow(img_rec, vmin=0, vmax=255)
            plt.colorbar(img_rec_plot, ax=axes[1])
            axes[1].set_title("Reconstruction")
            plt.show()
