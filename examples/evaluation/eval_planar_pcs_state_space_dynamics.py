import flax.linen as nn
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import Array, random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

from src.models.neural_odes import ConOde, CornnOde, LnnOde, LinearStateSpaceOde, MlpOde
from src.training.dataset_utils import load_dataset, load_dummy_neural_network_input
from src.training.loops import run_eval
from src.tasks import state_space_dynamics
from src.training.train_state_utils import restore_train_state
from src.visualization.latent_space import (
    visualize_mapping_from_configuration_to_latent_space,
)

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

system_type = "pcc_ns-2"
# dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp", "node-cornn", "node-con", "node-lnn", "node-hippo-lss", "discrete-mlp"]
dynamics_model_name = "node-mechanical-mlp"
normalize_loss = False

batch_size = 10
loss_weights = dict(mse_q=0.0, mse_q_d=1.0)
start_time_idx = 1
num_past_timesteps = 2

num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 20, "leaky_relu"
cornn_gamma, cornn_epsilon = 1.0, 1.0

if dynamics_model_name == "node-mechanical-mlp":
    """ Training stats
    {'loss': Array(0.15468458, dtype=float64), 'rmse_q_norm': Array(0.00099309, dtype=float64), 'rmse_q_d_norm': Array(0.01251912, dtype=float64)}
    """
    experiment_id = "2024-02-08_10-45-33"
    num_mlp_layers, mlp_hidden_dim = 3, 81
    mlp_nonlinearity_name = "selu"
elif dynamics_model_name == "node-lnn":
    """ Training stats
    {'loss': Array(0.4666356, dtype=float64), 'rmse_q_norm': Array(0.00123107, dtype=float64), 'rmse_q_d_norm': Array(0.02174395, dtype=float64)}
    """
    experiment_id = "2024-02-08_11-06-41"
    num_mlp_layers, mlp_hidden_dim = 5, 15
    mlp_nonlinearity_name = "elu"
    diag_shift, diag_eps = 8.271283131006865e-05, 0.005847971857910474
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

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"
)
ckpt_dir = Path("logs").resolve() / f"{system_type}_state_space_dynamics" / experiment_id


if __name__ == "__main__":
    datasets, dataset_info, dataset_metadata = load_dataset(
        f"planar_pcs/{system_type}_32x32px",
        seed=seed,
        batch_size=batch_size,
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
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.factory(
        sym_exp_filepath, strain_selector=dataset_metadata["strain_selector"]
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
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
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
    elif dynamics_model_name in ["node-general-lss", "node-mechanical-lss", "node-hippo-lss"]:
        nn_model = LinearStateSpaceOde(
            latent_dim=n_q,
            input_dim=n_tau,
            transition_matrix_init=dynamics_model_name.split("-")[1],  # "general", "mechanical", or "hippo"
        )
    elif dynamics_model_name == "discrete-mlp":
        nn_model = DiscreteMlpDynamics(
            latent_dim=n_q,
            input_dim=n_tau,
            output_dim=n_q,
            dt=dataset_metadata["dt"],
            num_past_timesteps=num_past_timesteps,
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

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = state_space_dynamics.task_factory(
        system_type,
        nn_model,
        ts=dataset_metadata["ts"],
        sim_dt=jnp.min(jnp.diff(dataset_metadata["ts"])).item() / 4,
        x0_min=dataset_metadata["x0_min"],
        x0_max=dataset_metadata["x0_max"],
        loss_weights=loss_weights,
        dynamics_type=dynamics_type,
        normalize_loss=normalize_loss,
        solver=solver_class(),
        start_time_idx=start_time_idx,
        num_past_timesteps=num_past_timesteps,
    )

    # load the neural network dummy input
    nn_dummy_input = load_dummy_neural_network_input(test_ds, task_callables)
    # load the training state from the checkpoint directory
    state = restore_train_state(
        rng, ckpt_dir, nn_model, nn_dummy_input, metrics_collection_cls
    )

    print("Run testing...")
    state, test_history = run_eval(train_ds, state, task_callables)
    test_metrics = state.metrics.compute()
    print(
        "\n"
        f"Final {'normalized ' if normalize_loss else ''}test metrics:\n{test_metrics}"
    )

    # define ground-truth dynamics
    gt_ode_fn = ode_with_forcing_factory(dynamical_matrices_fn, robot_params)

    # rollout dynamics
