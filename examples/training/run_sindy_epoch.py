from datetime import datetime
import flax.linen as nn
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import pendulum
from pathlib import Path
import tensorflow as tf

# jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'

from src.autoencoders.simple_cnn import Autoencoder
from src.tasks import fp_dynamics_sindy_loss
from src.training.load_dataset import load_dataset
from src.training.loops import run_training
from src.training.optim import create_learning_rate_fn
from src.training.train_state_utils import initialize_train_state

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"pendulum_nl-1.dill"
)

# set hyperparameters
batch_size = 10

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        "pendulum/single_pendulum_64x64px",
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
    nn_model = Autoencoder(
        latent_dim=latent_dim, img_shape=img_shape, norm_layer=nn.LayerNorm
    )

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = fp_dynamics_sindy_loss.task_factory(
        "pendulum",
        nn_model,
        ode_fn=ode_with_forcing_factory(dynamical_matrices_fn, robot_params),
        ts=dataset_metadata["ts"],
        x0_min=dataset_metadata["x0_min"],
        x0_max=dataset_metadata["x0_max"],
    )

    # extract dummy batch from dataset
    nn_dummy_batch = next(train_ds.as_numpy_iterator())
    # assemble input for dummy batch
    nn_dummy_input = task_callables.assemble_input_fn(nn_dummy_batch)

    # create learning rate schedule
    lr_fn = create_learning_rate_fn(
        0,
        steps_per_epoch=len(train_ds),
        base_lr=1e-4,
        warmup_epochs=0,
    )

    # initialize the train state
    state = initialize_train_state(
        rng,
        nn_model,
        nn_dummy_input=nn_dummy_input,
        metrics_collection_cls=metrics_collection_cls,
        learning_rate_fn=lr_fn,
    )

    for batch in train_ds:
        task_callables.forward_fn(batch, state.params)
        break

    # run the training loop
    """
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
        weight_decay=weight_decay,
        logdir=logdir,
    )
    print("Final training metrics:\n", state.metrics.compute())
    """
