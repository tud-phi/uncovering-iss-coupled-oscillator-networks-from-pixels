from datetime import datetime
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
from jsrm.integration import ode_factory
from jsrm.systems import pendulum
import optax
from pathlib import Path
import tensorflow as tf

# jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'

from src.neural_networks.simple_cnn import Autoencoder
from src.tasks import fp_dynamics
from src.training.load_dataset import load_dataset
from src.training.loops import run_training

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

num_epochs = 100
batch_size = 10
base_lr = 2e-3
warmup_epochs = 5
loss_weights = dict(mse_q=0.0, mse_rec_static=5.0, mse_rec_dynamic=35.0)

now = datetime.now()
logdir = Path("logs") / "single_pendulum_fp_dynamics" / f"{now:%Y-%m-%d_%H-%M-%S}"
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = Path("symbolic_expressions") / "single_pendulum.dill"

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        "mechanical_system/single_pendulum_64x64px",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    steps_per_epoch = len(train_ds)

    # extract the robot parameters from the dataset
    robot_params = dataset_metadata["system_params"]
    print(f"Robot parameters: {robot_params}")
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    # initialize the model
    nn_model = Autoencoder(latent_dim=2 * n_q, img_shape=img_shape)

    # initialize the schedule for the configuration velocity source
    direct_finite_differences_weight_ratio_scheduler = optax.linear_schedule(
        init_value=0.0, end_value=1.0, transition_steps=num_epochs * steps_per_epoch
    )

    # call the factory function for the sensing task
    task_callables, metrics = fp_dynamics.task_factory(
        "pendulum",
        nn_model,
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))),
        loss_weights=loss_weights,
        solver=dataset_metadata["solver_class"](),
        configuration_velocity_source="combined",
        direct_finite_differences_weight_ratio_scheduler=direct_finite_differences_weight_ratio_scheduler,
    )

    # run the training loop
    print("Run training...")
    (
        state,
        train_history,
    ) = run_training(
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
