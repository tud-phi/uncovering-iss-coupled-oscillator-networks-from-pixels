from datetime import datetime
import flax.linen as nn
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.systems import pendulum
from pathlib import Path
import tensorflow as tf

# jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'

from src.models.discrete_forward_dynamics import DiscreteMlpDynamics
from src.models.neural_odes import ConOde, CornnOde, LnnOde, LinearStateSpaceOde, MlpOde
from src.tasks import state_space_dynamics
from src.training.dataset_utils import load_dataset
from src.training.loops import run_training

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

# dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp", "node-cornn", "node-con", "node-lnn", "node-hippo-lss", "discrete-mlp"]
dynamics_model_name = "node-mechanical-mlp"

batch_size = 100
num_epochs = 50
warmup_epochs = 5
start_time_idx = 1

num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 20, "leaky_relu"
cornn_gamma, cornn_epsilon = 1.0, 1.0

base_lr = 1e-3
loss_weights = dict(
    mse_q=1.0,
    mse_q_d=1.0,
)
weight_decay = 0.0
num_mlp_layers = 4
mlp_hidden_dim = 40

# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in ["node", "discrete"], f"Unknown dynamics_type: {dynamics_type}"

now = datetime.now()
logdir = (
    Path("logs").resolve()
    / "single_pendulum_state_space_dynamics"
    / f"{now:%Y-%m-%d_%H-%M-%S}"
)
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"pendulum_nl-1.dill"
)

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        "pendulum/single_pendulum_32x32px",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # extract the robot parameters from the dataset
    robot_params = dataset_metadata["system_params"]
    print(f"Robot parameters: {robot_params}")
    n_tau = train_ds.element_spec["tau"].shape[-1]  # dimension of the control input
    n_q = (
        train_ds.element_spec["x_ts"].shape[-1] // 2
    )  # dimension of the configuration space

    # get the dynamics function
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    if dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp"]:
        nn_model = MlpOde(
            latent_dim=n_q,
            input_dim=n_tau,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
            mechanical_system=True
            if dynamics_model_name == "node-mechanical-mlp"
            else False,
        )
    elif dynamics_model_name == "node-cornn":
        nn_model = CornnOde(
            latent_dim=n_q,
            input_dim=n_tau,
            gamma=cornn_gamma,
            epsilon=cornn_epsilon,
        )
    elif dynamics_model_name == "node-con":
        nn_model = ConOde(
            latent_dim=n_q,
            input_dim=n_tau,
        )
    elif dynamics_model_name == "node-lnn":
        nn_model = LnnOde(
            latent_dim=n_q,
            input_dim=n_tau,
        )
    elif dynamics_model_name in [
        "node-general-lss",
        "node-mechanical-lss",
        "node-hippo-lss",
    ]:
        nn_model = LinearStateSpaceOde(
            latent_dim=n_q,
            input_dim=n_tau,
            transition_matrix_init=dynamics_model_name.split("-")[
                1
            ],  # "general", "mechanical", or "hippo"
        )
    elif dynamics_model_name == "discrete-mlp":
        nn_model = DiscreteMlpDynamics(
            input_dim=n_tau,
            output_dim=2 * n_q,
            dt=dataset_metadata["dt"],
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
        )
    else:
        raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")

    # import solver class from diffrax
    # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    solver_class = getattr(
        __import__("diffrax", fromlist=[dataset_metadata["solver_class"]]),
        dataset_metadata["solver_class"],
    )

    # call the factory function for the state space dynamics task
    task_callables, metrics_collection_cls = state_space_dynamics.task_factory(
        "pendulum",
        ts=dataset_metadata["ts"],
        sim_dt=dataset_metadata["sim_dt"],
        x0_min=dataset_metadata["x0_min"],
        x0_max=dataset_metadata["x0_max"],
        loss_weights=loss_weights,
        dynamics_type=dynamics_type,
        nn_model=nn_model,
        solver=solver_class(),
        start_time_idx=start_time_idx,
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
