import flax.linen as nn
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import Array, random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import planar_pcs
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

system_type = "cc"
ae_type = "beta_vae"  # "None", "beta_vae", "wae"

if ae_type == "beta_vae":
    experiment_id = "2023-09-18_20-43-40"
else:
    # ae_type == "None"
    experiment_id = "2023-09-18_11-21-53"
ckpt_dir = Path("logs") / f"{system_type}_fp_dynamics" / experiment_id

batch_size = 10
loss_weights = dict(mse_q=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0)
start_time_idx = 1
norm_layer = nn.LayerNorm

if ae_type == "beta_vae":
    loss_weights["beta"] = 1.0

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        f"planar_pcs/{system_type}_64x64px",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # extract the robot parameters from the dataset
    robot_params = dataset_metadata["system_params"]
    print(f"Robot parameters: {robot_params}")
    num_segments = dataset_metadata.get("num_segments", 1)
    # number of generalized coordinates
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    # latent space shape
    latent_dim = n_q
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape

    sym_exp_filepath = (
        Path(jsrm.__file__).parent
        / "symbolic_expressions"
        / f"planar_pcs_ns-{num_segments}.dill"
    )

    # initialize the system
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.factory(
        sym_exp_filepath, dataset_metadata["strain_selector"]
    )

    # initialize the model
    if ae_type == "beta_vae":
        nn_model = VAE(
            latent_dim=latent_dim, img_shape=img_shape, norm_layer=norm_layer
        )
    else:
        nn_model = Autoencoder(
            latent_dim=latent_dim, img_shape=img_shape, norm_layer=norm_layer
        )

    # call the factory function for the fp dynamics task
    task_callables, metrics_collection_cls = fp_dynamics.task_factory(
        system_type,
        nn_model,
        ts=dataset_metadata["ts"],
        sim_dt=dataset_metadata["sim_dt"],
        ode_fn=ode_with_forcing_factory(dynamical_matrices_fn, robot_params),
        loss_weights=loss_weights,
        ae_type=ae_type,
        solver=dataset_metadata["solver_class"](),
        start_time_idx=start_time_idx,
        configuration_velocity_source="direct-finite-differences",
    )

    state = restore_train_state(rng, ckpt_dir, nn_model, metrics_collection_cls)

    print("Run testing...")
    state, test_history = run_eval(test_ds, state, task_callables)
    test_metrics = state.metrics.compute()
    print(
        "\n"
        f"Final test metrics:\n"
        f"rmse_q_static={test_metrics['rmse_q_static']:.4f}, "
        f"rmse_q_dynamic={test_metrics['rmse_q_dynamic']:.4f}, "
        f"rmse_q_d_dynamic={test_metrics['rmse_q_d_dynamic']:.4f},\n"
        f"rmse_rec_static={test_metrics['rmse_rec_static']:.4f}, "
        f"rmse_rec_dynamic={test_metrics['rmse_rec_dynamic']:.4f}"
    )

    visualize_mapping_from_configuration_to_latent_space(
        test_ds, state, task_callables, rng=rng
    )

    test_batch = next(test_ds.as_numpy_iterator())
    test_preds = task_callables.forward_fn(test_batch, state.params)

    for i in range(test_batch["x_ts"].shape[0]):
        print("Trajectory:", i)

        print(
            f"Estimated initial velocity: {test_preds['x_dynamic_ts'][i, 0, n_q:]}rad/(ms), actual initial velocity: {test_batch['x_ts'][i, start_time_idx, n_q:]}rad/(ms)"
        )

        for t in range(start_time_idx, test_batch["x_ts"].shape[1]):
            print("Time step:", t)
            q_gt = test_batch["x_ts"][i, t, :n_q]
            q_pred = test_preds["q_dynamic_ts"][i, t - start_time_idx, :n_q]
            error_q = (
                test_preds["q_dynamic_ts"][i, t - start_time_idx, :n_q]
                - test_batch["x_ts"][i, t, :n_q]
            )
            print(
                "Ground-truth q:",
                q_gt,
                "rad/m",
                "Predicted q:",
                q_pred,
                "rad/m",
                "Error:",
                error_q,
                "rad/m",
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
