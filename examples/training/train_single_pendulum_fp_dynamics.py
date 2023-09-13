from datetime import datetime
import flax.linen as nn
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

num_epochs = 50
warmup_epochs = 5
batch_size = 100
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
configuration_velocity_source = "direct-finite-differences"

if ae_type == "wae":
    # with rmse_q_static objective
    # base_lr = 0.002645501263921337
    # loss_weights = dict(
    #     mse_q=0.17923446274288507,
    #     mse_rec_static=1.0,
    #     mse_rec_dynamic=44.77484360640797,
    #     mmd=0.11675524982544401,
    # )
    # weight_decay = 2.5017288074367157e-05
    # start_time_idx = 2
    # with rmse_rec_dynamic objective
    base_lr = 0.005172703488133171
    loss_weights = dict(
        mse_q=0.2695465212029283,
        mse_rec_static=1.0,
        mse_rec_dynamic=22.30264086175223,
        mmd=0.4348682813029425,
    )
    weight_decay = 0.0001660747175371815
    start_time_idx = 1
elif ae_type == "beta_vae":
    # # with rmse_q_static objective
    # base_lr = 0.0016071488813346846
    # loss_weights = dict(
    #     mse_q=0.05458926808374876,
    #     mse_rec_static=1.0,
    #     mse_rec_dynamic=35.57369398107525,
    #     beta=0.001505505029022702,
    # )
    # weight_decay = 2.755442935349405e-05
    # start_time_idx = 1
    # with rmse_rec_dynamic objective
    base_lr = 0.002353419319997345
    loss_weights = dict(
        mse_q=0.019601416440053525,
        mse_rec_static=1.0,
        mse_rec_dynamic=118.94483974763628,
        beta=0.028185805822357596,
    )
    weight_decay = 4.972203024278259e-05
    start_time_idx = 1
else:
    # ae_type == "None"
    base_lr = 0.00396567508177101
    loss_weights = dict(
        mse_q=0.7013219779945796, mse_rec_static=1.0, mse_rec_dynamic=77.11768972549937
    )
    weight_decay = 1.7240460099242286e-05
    start_time_idx = 7

now = datetime.now()
logdir = Path("logs") / "single_pendulum_fp_dynamics" / f"{now:%Y-%m-%d_%H-%M-%S}"
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"pendulum_nl-1.dill"
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
            latent_dim=latent_dim, img_shape=img_shape, norm_layer=nn.LayerNorm
        )
    else:
        nn_model = Autoencoder(
            latent_dim=latent_dim, img_shape=img_shape, norm_layer=nn.LayerNorm
        )

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = fp_dynamics.task_factory(
        "pendulum",
        nn_model,
        ts=dataset_metadata["ts"],
        sim_dt=dataset_metadata["sim_dt"],
        ode_fn=ode_factory(dynamical_matrices_fn, robot_params, tau=jnp.zeros((n_q,))),
        loss_weights=loss_weights,
        solver=dataset_metadata["solver_class"](),
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
