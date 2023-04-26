import jax.numpy as jnp
from jsrm.integration import ode_factory
from jsrm.systems import euler_lagrangian, pendulum
from pathlib import Path
import tensorflow as tf

from src.neural_networks.simple_cnn import Autoencoder
from src.training.load_dataset import load_dataset
from src.training.loops import run_training, run_eval
from src.training.tasks import fp_dynamics_wo_vel
from src.training.train_state_utils import restore_train_state

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
batch_size = 8
loss_weights = dict(mse_q=1.0, mse_rec_static=5.0, mse_rec_dynamic=5.0)

sym_exp_filepath = Path("symbolic_expressions") / "single_pendulum.dill"
ckpt_dir = Path("logs") / "single_pendulum_fp_dynamics_wo_vel" / "2023-04-26_11-59-21"


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
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    # initialize the model
    nn_model = Autoencoder(
        latent_dim=n_q,
        img_shape=img_shape
    )

    # call the factory function for the sensing task
    task_callables, metrics = fp_dynamics_wo_vel.task_factory(
        "pendulum",
        nn_model,
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))),
        loss_weights=loss_weights,
        solver=dataset_metadata["solver_class"](),
    )

    state = restore_train_state(ckpt_dir, nn_model, metrics)

    print("Run testing...")
    test_history = run_eval(test_ds, state, task_callables)
    (
        rmse_q_static_stps,
        rmse_q_dynamic_stps,
        rmse_rec_static_stps,
        rmse_rec_dynamic_stps
    ) = test_history.collect("rmse_q_static", "rmse_q_dynamic", "rmse_rec_static", "rmse_rec_dynamic")
    print(
        "\n"
        f"Final test metrics: rmse_q_static_stps={rmse_q_static_stps[-1]:.3f}, "
        f"rmse_q_dynamic_stps={rmse_q_dynamic_stps[-1]:.3f}, "
        f"rmse_rec_static_stps={rmse_rec_static_stps[-1]:.3f}, "
        f"rmse_rec_dynamic_stps={rmse_rec_dynamic_stps[-1]:.3f}"
    )