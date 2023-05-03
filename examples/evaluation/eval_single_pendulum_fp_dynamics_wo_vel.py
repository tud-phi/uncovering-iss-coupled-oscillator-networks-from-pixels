from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import Array, jacfwd, random
import jax.numpy as jnp
from jsrm.integration import ode_factory
from jsrm.systems import pendulum
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

from src.neural_networks.simple_cnn import Autoencoder
from src.training.load_dataset import load_dataset
from src.training.loops import run_eval
from src.tasks import fp_dynamics_wo_vel
from src.training.train_state_utils import restore_train_state

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)

batch_size = 8
loss_weights = dict(mse_q=1.0, mse_rec_static=5.0, mse_rec_dynamic=5.0)

sym_exp_filepath = Path("symbolic_expressions") / "single_pendulum.dill"
ckpt_dir = Path("logs") / "single_pendulum_fp_dynamics_wo_vel" / "2023-05-02_10-20-40"


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
    nn_model = Autoencoder(latent_dim=2 * n_q, img_shape=img_shape)

    # call the factory function for the sensing task
    task_callables, metrics = fp_dynamics_wo_vel.task_factory(
        "pendulum",
        nn_model,
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))),
        loss_weights=loss_weights,
        solver=dataset_metadata["solver_class"](),
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

    # Experiment with predicting the latent-space velocity
    batch_idx = 0  # batch index of sample for which we want to predict the latent-space velocity
    time_idx = 1  # time index of sample for which we want to predict the latent-space velocity
    t_ts = test_batch["t_ts"][batch_idx, :]
    img_gt_ts = test_batch["rendering_ts"][batch_idx, :]
    x_gt_ts = test_batch["x_ts"][batch_idx, :]
    q_gt_ts = x_gt_ts[:, :n_q]
    q_d_gt_ts = x_gt_ts[:, n_q:]

    print("q", q_gt_ts[time_idx, :], "q_d_0", q_d_gt_ts[time_idx, :])


    def pendulum_encode(_img_ts: Array):
        _encoder_output = nn_model.apply(
            {"params": state.params}, _img_ts, method=nn_model.encode
        )

        # if the system is a pendulum, we interpret the encoder output as sin(theta) and cos(theta) for each joint
        # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
        # output of arctan2 will be in the range [-pi, pi]
        _q_ts = jnp.arctan2(
            _encoder_output[..., :n_q], _encoder_output[..., n_q:]
        )
        return _q_ts


    jac_fn = jacfwd(pendulum_encode)
    z_pred = pendulum_encode(img_gt_ts[time_idx:time_idx + 1, ...]).squeeze(0)
    print("z_pred:", z_pred)
    dz_dimg = jac_fn(img_gt_ts[time_idx:time_idx + 1, ...]).squeeze((0, 1))

    # use finite differences to compute the velocity in image space
    dt = t_ts[time_idx] - t_ts[time_idx - 1]
    # img_d_fd = (img_gt_ts[time_idx, ...] - img_gt_ts[time_idx - 1, ...]) / dt  # naive finite differences
    img_d_fd = jnp.gradient(img_gt_ts, dt, axis=0)[time_idx, ...]
    # apply the chain rule to compute the velocity in latent space
    # flatten so we can do matrix multiplication
    dz_dimg_flat = dz_dimg.reshape((dz_dimg.shape[0], -1))
    img_d_fd_flat = img_d_fd.flatten()
    z_d_hat_flat = jnp.matmul(dz_dimg_flat, img_d_fd_flat)
    z_d_hat = z_d_hat_flat.reshape(z_pred.shape)
    print("Estimated latent-space velocity:", z_d_hat)
