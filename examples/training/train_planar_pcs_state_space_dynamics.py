from datetime import datetime
import flax.linen as nn
import jax

# jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
from pathlib import Path
import tensorflow as tf

from src.models.discrete_forward_dynamics import (
    DiscreteLssDynamics,
    DiscreteMambaDynamics,
    DiscreteMlpDynamics,
    DiscreteRnnDynamics,
)
from src.models.neural_odes import (
    ConOde,
    CornnOde,
    LnnOde,
    LinearStateSpaceOde,
    MambaOde,
    MlpOde,
)
from src.tasks import state_space_dynamics
from src.training.dataset_utils import load_dataset
from src.training.loops import run_training

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

system_type = "pcc_ns-2"
""" dynamics_model_name in [
    "node-general-mlp", "node-mechanical-mlp", "node-cornn", "node-con", "node-lnn", "node-hippo-lss", "node-mamba",
    "discrete-mlp", "discrete-elman-rnn", "discrete-gru-rnn", "discrete-general-lss", "discrete-hippo-lss", "discrete-mamba",
]
"""
dynamics_model_name = "discrete-gru-rnn"

batch_size = 100
num_epochs = 50
warmup_epochs = 5
start_time_idx = 0

base_lr = 0.0
loss_weights = dict(
    mse_q=0.0,
    mse_q_d=1.0,
)
weight_decay = 0.0
num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 20, "leaky_relu"
cornn_gamma, cornn_epsilon = 1.0, 1.0

if dynamics_model_name in ["node-mechanical-mlp"]:
    base_lr = 0.03323371435041385
    loss_weights = dict(
        mse_q=0.0003462995467520171,
        mse_q_d=1.0,
    )
    weight_decay = 1.8252275472841628e-05
    num_mlp_layers = 3
    mlp_hidden_dim = 81
    mlp_nonlinearity_name = "selu"
elif dynamics_model_name == "node-lnn":
    base_lr = 0.014497133990714495
    loss_weights = dict(
        mse_q=0.9347261979172878,
        mse_q_d=1.0,
    )
    weight_decay = 5.4840283002626335e-05
    num_mlp_layers = 5
    mlp_hidden_dim = 15
    mlp_nonlinearity_name = "softplus"
    diag_shift, diag_eps = 8.271283131006865e-05, 0.005847971857910474
elif dynamics_model_name == "node-con":
    base_lr = 0.05464221872891958
    loss_weights = dict(
        mse_q=0.0003253194073232259,
        mse_q_d=1.0,
    )
    weight_decay = 0.00019740817308270745
    nonlinearity_name = "sigmoid"
    diag_shift, diag_eps = 7.892200115980268e-06, 1.3090986307629387e-06
elif dynamics_model_name == "discrete-mlp":
    base_lr = 0.0001
    loss_weights = dict(
        mse_q=0.0,
        mse_q_d=1.0,
    )
    weight_decay = 0.0
    num_mlp_layers = 5
    mlp_hidden_dim = 20
    mlp_nonlinearity_name = "leaky_relu"
elif dynamics_model_name in [
    "discrete-elman-rnn",
    "discrete-gru-rnn",
    "discrete-general-lss",
    "discrete-hippo-lss",
]:
    base_lr = 0.0001
    loss_weights = dict(
        mse_q=0.0,
        mse_q_d=1.0,
    )
    weight_decay = 0.0
else:
    raise NotImplementedError(f"Unknown dynamics_model_name: {dynamics_model_name}")

# identify the number of segments
if system_type == "cc":
    num_segments = 1
elif system_type.split("_")[0] == "pcc":
    num_segments = int(system_type.split("-")[-1])
else:
    raise ValueError(f"Unknown system_type: {system_type}")
print(f"Number of segments: {num_segments}")

# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in ["node", "discrete"], f"Unknown dynamics_type: {dynamics_type}"

now = datetime.now()
logdir = (
    Path("logs").resolve()
    / f"{system_type}_state_space_dynamics"
    / f"{now:%Y-%m-%d_%H-%M-%S}"
)
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_pcs_ns-{num_segments}.dill"
)

if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        f"planar_pcs/{system_type}_32x32px",
        seed=seed,
        batch_size=batch_size,
        num_epochs=num_epochs,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # extract the robot parameters from the dataset
    robot_params = dataset_metadata["system_params"]
    print(f"Robot parameters: {robot_params}")
    print("Strain selector:", dataset_metadata["strain_selector"])
    # dimension of the configuration space
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    # dimension of the control input
    n_tau = train_ds.element_spec["tau"].shape[-1]

    # get the dynamics function
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
        planar_pcs.factory(
            sym_exp_filepath, strain_selector=dataset_metadata["strain_selector"]
        )
    )

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
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
        )
    elif dynamics_model_name == "node-con":
        nn_model = ConOde(
            latent_dim=n_q,
            input_dim=n_tau,
            nonlinearity=getattr(nn, nonlinearity_name),
            diag_shift=diag_shift,
            diag_eps=diag_eps,
        )
    elif dynamics_model_name == "node-lnn":
        nn_model = LnnOde(
            latent_dim=n_q,
            input_dim=n_tau,
            learn_dissipation=True,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
            diag_shift=diag_shift,
            diag_eps=diag_eps,
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
    elif dynamics_model_name == "node-mamba":
        nn_model = MambaOde(
            latent_dim=n_q,
            input_dim=n_tau,
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
    elif dynamics_model_name in ["discrete-elman-rnn", "discrete-gru-rnn"]:
        nn_model = DiscreteRnnDynamics(
            state_dim=2 * n_q,
            input_dim=n_tau,
            output_dim=2 * n_q,
            rnn_method=dynamics_model_name.split("-")[1],  # "elman" or "gru"
        )
    elif dynamics_model_name in ["discrete-general-lss", "discrete-hippo-lss"]:
        nn_model = DiscreteLssDynamics(
            state_dim=2 * n_q,
            input_dim=n_tau,
            output_dim=2 * n_q,
            dt=dataset_metadata["dt"],
            transition_matrix_init=dynamics_model_name.split("-")[
                1
            ],  # "general", or "hippo"
        )
    elif dynamics_model_name == "discrete-mamba":
        nn_model = DiscreteMambaDynamics(
            state_dim=2 * n_q,
            input_dim=n_tau,
            output_dim=2 * n_q,
            dt=dataset_metadata["dt"],
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
        system_type,
        ts=dataset_metadata["ts"],
        sim_dt=jnp.min(jnp.diff(dataset_metadata["ts"])).item() / 4,
        x0_min=dataset_metadata["x0_min"],
        x0_max=dataset_metadata["x0_max"],
        loss_weights=loss_weights,
        dynamics_type=dynamics_type,
        nn_model=nn_model,
        normalize_loss=True,
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
