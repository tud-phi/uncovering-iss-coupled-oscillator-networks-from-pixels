from datetime import datetime
from flax import traverse_util
from flax.core.frozen_dict import freeze, unfreeze
import flax.linen as nn
import jax

jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import planar_pcs
import optax
from pathlib import Path
import tensorflow as tf

from src.models.autoencoders.simple_cnn import Autoencoder
from src.models.autoencoders.staged_autoencoder import StagedAutoencoder
from src.models.autoencoders.vae import VAE
from src.tasks import fp_dynamics_autoencoder
from src.training.dataset_utils import load_dataset
from src.training.loops import run_training
from src.training.optim import create_learning_rate_fn
from src.training.train_state_utils import initialize_train_state, restore_train_state
from src.visualization.dataset_distribution import (
    plot_acting_forces_distribution,
    plot_basic_distribution,
)
from src.visualization.latent_space import (
    visualize_mapping_from_configuration_to_latent_space,
)

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

system_type = "cc"
pretrained_ae_type = "beta_vae"  # "None", "beta_vae", "wae"
ckpt_dir = (
    Path("logs").resolve() / f"{system_type}_autoencoding" / f"2023-09-24_14-41-33"
)

num_epochs = 40
warmup_epochs = 5
batch_size = 100
start_time_idx = 0
configuration_velocity_source = "ground-truth"

base_lr = 1e-1
loss_weights = dict(mse_q=1e-2, mse_rec_static=0.0, mse_rec_dynamic=100)
weight_decay = 0.0

now = datetime.now()
logdir = (
    Path("logs").resolve() / f"{system_type}_fp_dynamics" / f"{now:%Y-%m-%d_%H-%M-%S}"
)
logdir.mkdir(parents=True, exist_ok=True)

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

    # get the dynamics function
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = planar_pcs.factory(
        sym_exp_filepath, strain_selector=dataset_metadata["strain_selector"]
    )

    # plot training dataset distribution
    # plot_basic_distribution(train_ds)
    # plot_acting_forces_distribution(train_ds, system_type, robot_params, dynamical_matrices_fn)

    # initialize the model
    if pretrained_ae_type == "beta_vae":
        backbone = VAE(
            latent_dim=latent_dim, img_shape=img_shape, norm_layer=nn.LayerNorm
        )
    else:
        backbone = Autoencoder(
            latent_dim=latent_dim, img_shape=img_shape, norm_layer=nn.LayerNorm
        )
    nn_model = StagedAutoencoder(backbone=backbone, config_dim=n_q, mirror_head=True)

    # import solver class from diffrax
    # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    solver_class = getattr(
        __import__("diffrax", fromlist=[dataset_metadata["solver_class"]]),
        dataset_metadata["solver_class"],
    )

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = fp_dynamics_autoencoder.task_factory(
        system_type,
        nn_model,
        ode_fn=ode_with_forcing_factory(dynamical_matrices_fn, robot_params),
        ts=dataset_metadata["ts"],
        sim_dt=dataset_metadata["sim_dt"],
        loss_weights=loss_weights,
        ae_type="None",
        solver=solver_class(),
        start_time_idx=start_time_idx,
        configuration_velocity_source=configuration_velocity_source,
    )

    # extract dummy batch from dataset
    nn_dummy_batch = next(train_ds.as_numpy_iterator())
    # assemble input for dummy batch
    nn_dummy_input = task_callables.assemble_input_fn(nn_dummy_batch)

    # initialize the learning rate scheduler
    lr_fn = create_learning_rate_fn(
        num_epochs=num_epochs,
        steps_per_epoch=len(train_ds),
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
    )
    pretrained_state = restore_train_state(
        rng, ckpt_dir, backbone, metrics_collection_cls=metrics_collection_cls
    )
    # initialize the train state
    state = initialize_train_state(
        rng,
        nn_model,
        nn_dummy_input=nn_dummy_input,
        metrics_collection_cls=metrics_collection_cls,
        init_fn=nn_model.__call__,
        learning_rate_fn=lr_fn,
        weight_decay=weight_decay,
    )
    params = unfreeze(state.params)
    # copy the pretrained parameters into the new state
    params["backbone"] = pretrained_state.params
    # make sure that the kernel of the head is positive definite
    params["head"]["kernel"] = jnp.abs(params["head"]["kernel"])
    print("head params:\n", params["head"])
    # freeze the parameters again and save to the new state
    state = state.replace(step=0, params=freeze(params))
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
    # print("param_partitions:\n", param_partitions)
    fp_dynamics_tx = optax.multi_transform(partition_optimizers, param_partitions)

    # run the training loop
    print("Run training...")
    (state, train_history, elapsed) = run_training(
        rng=rng,
        train_ds=train_ds,
        val_ds=val_ds,
        task_callables=task_callables,
        metrics_collection_cls=metrics_collection_cls,
        num_epochs=num_epochs,
        state=state,
        nn_model=nn_model,
        tx=fp_dynamics_tx,
        base_lr=base_lr,
        logdir=logdir,
    )
    print("Final training metrics:\n", state.metrics.compute())
    print("Final params of head:\n", state.params["head"])

    visualize_mapping_from_configuration_to_latent_space(
        test_ds, state, task_callables, rng=rng
    )
