from datetime import datetime
from flax import traverse_util
from flax.core.frozen_dict import freeze
import flax.linen as nn
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import pendulum
from pathlib import Path
import optax
import tensorflow as tf

# jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'

from src.autoencoders.simple_cnn import Autoencoder
from src.autoencoders.staged_autoencoder import StagedAutoencoder
from src.tasks import autoencoding, fp_dynamics
from src.training.load_dataset import load_dataset
from src.training.loops import run_training
from src.training.optim import create_learning_rate_fn

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

latent_dim = 8
config_dim = 2

batch_size = 10

hyperparams = [
    # WAE
    dict(
        num_epochs=10,
        base_lr=5e-4,
        warmup_epochs=3,
        loss_weights=dict(mse_q=0.0, mse_rec=5.0, mmd=1e-1),
    ),
    # dynamic learning of configuration space
    dict(
        num_epochs=50,
        base_lr=2e-3,
        warmup_epochs=5,
        loss_weights=dict(mse_q=0.0, mse_rec_static=5.0, mse_rec_dynamic=35.0),
    ),
]

now = datetime.now()
logdir = (
    Path("logs") / "single_pendulum_staged_rp_learning" / f"{now:%Y-%m-%d_%H-%M-%S}"
)
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"pendulum_nl-1.dill"
)

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
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    # initialize the model
    backbone = Autoencoder(
        latent_dim=latent_dim, img_shape=img_shape, norm_layer=nn.LayerNorm
    )
    nn_model = StagedAutoencoder(backbone=backbone, config_dim=2 * n_q)

    # call the factory function for the sensing task
    (
        autoencoding_task_callables,
        autoencoding_metrics_collection_cls,
    ) = autoencoding.task_factory(
        "pendulum",
        nn_model,
        normalize_latent_space=False,
        encode_kwargs=dict(use_head=False),
        decode_kwargs=dict(use_head=False),
        loss_weights=hyperparams[0]["loss_weights"],
        weight_on_foreground=0.15,
        ae_type="wae",
    )

    # run the WAE training loop
    print("Run WAE training...")
    (state, train_history, elapsed) = run_training(
        rng=rng,
        train_ds=train_ds,
        val_ds=val_ds,
        task_callables=autoencoding_task_callables,
        metrics_collection_cls=autoencoding_metrics_collection_cls,
        num_epochs=hyperparams[0]["num_epochs"],
        nn_model=nn_model,
        init_kwargs=dict(use_head=True),
        base_lr=hyperparams[0]["base_lr"],
        warmup_epochs=hyperparams[0]["warmup_epochs"],
        weight_decay=0.0,
        logdir=logdir / "wae",
    )
    print("Final WAE training metrics:\n", state.metrics.compute())

    # import solver class from diffrax
    # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    solver_class = getattr(__import__(
        "diffrax", fromlist=[dataset_metadata["solver_class"]]),
        dataset_metadata["solver_class"]
    )

    # call the factory function for the dynamic learning of the configuration space
    fp_dynamics_task_callables, fp_dynamics_metrics = fp_dynamics.task_factory(
        "pendulum",
        nn_model,
        ode_fn=ode_with_forcing_factory(dynamical_matrices_fn, robot_params),
        ts=dataset_metadata["ts"],
        sim_dt=dataset_metadata["sim_dt"],
        loss_weights=hyperparams[1]["loss_weights"],
        solver=solver_class(),
        configuration_velocity_source="image-space-finite-differences",
    )

    # initialize the learning rate scheduler
    lr_fn = create_learning_rate_fn(
        num_epochs=hyperparams[1]["num_epochs"],
        steps_per_epoch=len(train_ds),
        base_lr=hyperparams[1]["base_lr"],
        warmup_epochs=hyperparams[1]["warmup_epochs"],
    )
    # initialize the Adam with weight decay optimizer for both neural networks
    partition_optimizers = {
        "trainable": optax.adamw(lr_fn),
        "frozen": optax.set_to_zero(),
    }
    param_partitions = freeze(
        traverse_util.path_aware_map(
            lambda path, v: "frozen" if "backbone" in path else "trainable",
            state.params,
        )
    )
    fp_dynamics_tx = optax.multi_transform(partition_optimizers, param_partitions)
    # reset training step
    state = state.replace(step=0)

    # run the dynamic learning training loop
    print("Run dynamic learning of configuration space...")
    (state, train_history, elapsed) = run_training(
        rng=rng,
        train_ds=train_ds,
        val_ds=val_ds,
        task_callables=fp_dynamics_task_callables,
        metrics_collection_cls=fp_dynamics_metrics,
        num_epochs=hyperparams[1]["num_epochs"],
        state=state,
        tx=fp_dynamics_tx,
        learning_rate_fn=lr_fn,
        logdir=logdir / "dynamic_learning",
    )
    print(
        "Final dynamic learning of configuration space metrics:\n",
        state.metrics.compute(),
    )
