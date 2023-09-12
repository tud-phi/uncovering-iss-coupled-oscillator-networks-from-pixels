from datetime import datetime
import flax.linen as nn
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_factory
from jsrm.systems import planar_pcs
from pathlib import Path
import tensorflow as tf

# jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'

from src.autoencoders.simple_cnn import Autoencoder
from src.autoencoders.vae import VAE
from src.tasks import fp_dynamics
from src.training.load_dataset import load_dataset
from src.training.loops import run_training

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

system_type = "cc"
ae_type = "beta_vae"  # "None", "beta_vae", "wae"

num_epochs = 50
warmup_epochs = 5
batch_size = 100
configuration_velocity_source = "direct-finite-differences"

if ae_type == "wae":
    base_lr = 0.005
    loss_weights = dict(
        mse_q=0.25,
        mse_rec_static=1.0,
        mse_rec_dynamic=20.0,
        mmd=0.40,
    )
    weight_decay = 0.0001660747175371815
    start_time_idx = 1
elif ae_type == "beta_vae":
    base_lr = 0.002
    loss_weights = dict(
        mse_q=0.02,
        mse_rec_static=1.0,
        mse_rec_dynamic=120,
        beta=0.03,
    )
    weight_decay = 5e-5
    start_time_idx = 1
else:
    # ae_type == "None"
    base_lr = 0.004
    loss_weights = dict(
        mse_q=0.70, mse_rec_static=1.0, mse_rec_dynamic=77.0
    )
    weight_decay = 1.7e-05
    start_time_idx = 1

now = datetime.now()
logdir = Path("logs") / "{system_type}_autoencoding" / f"{now:%Y-%m-%d_%H-%M-%S}"
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
        Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"
    )

    # get the dynamics function
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.factory(
        sym_exp_filepath, 
        strain_selector=dataset_metadata["strain_selector"]
    )

    # initialize the model
    if ae_type == "beta_vae":
        nn_model = VAE(
            latent_dim=latent_dim,
            img_shape=img_shape,
            norm_layer=nn.LayerNorm
        )
    else:
        nn_model = Autoencoder(latent_dim=latent_dim, img_shape=img_shape, norm_layer=nn.LayerNorm)

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = fp_dynamics.task_factory(
        system_type,
        nn_model,
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))),
        loss_weights=loss_weights,
        solver=dataset_metadata["solver_class"](),
        sim_dt=dataset_metadata.get("sim_dt"),
        start_time_idx=start_time_idx,
        configuration_velocity_source=configuration_velocity_source,
        ae_type=ae_type,
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
        weight_decay=weight_decay,
        logdir=logdir,
    )
    print("Final training metrics:\n", state.metrics.compute())
