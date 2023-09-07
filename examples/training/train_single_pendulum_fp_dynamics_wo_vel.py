from datetime import datetime
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_factory
from jsrm.systems import pendulum
from pathlib import Path
import tensorflow as tf

# jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'

from src.autoencoders.simple_cnn import Autoencoder
from src.tasks import fp_dynamics_wo_vel
from src.training.load_dataset import load_dataset
from src.training.loops import run_training

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

num_epochs = 50
batch_size = 8
base_lr = 2e-3
warmup_epochs = 3
loss_weights = dict(mse_q=0.0, mse_rec_static=5.0, mse_rec_dynamic=25.0)

now = datetime.now()
logdir = (
    Path("logs") / "single_pendulum_fp_dynamics_wo_vel" / f"{now:%Y-%m-%d_%H-%M-%S}"
)
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"pendulum_nl-1.dill"
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
    nn_model = Autoencoder(latent_dim=2 * n_q, img_shape=img_shape)

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = fp_dynamics_wo_vel.task_factory(
        "pendulum",
        nn_model,
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))),
        loss_weights=loss_weights,
        solver=dataset_metadata["solver_class"](),
        sim_dt=dataset_metadata.get("sim_dt"),
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
