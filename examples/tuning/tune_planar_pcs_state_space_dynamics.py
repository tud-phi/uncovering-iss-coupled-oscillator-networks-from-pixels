from datetime import datetime
import dill
import flax.linen as nn
import jax

jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import logging
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import tensorflow as tf

from src.models.discrete_forward_dynamics import DiscreteMlpDynamics
from src.models.neural_odes import ConOde, CornnOde, LinearStateSpaceOde, LnnOde, MlpOde
from src.tasks import state_space_dynamics
from src.training.callbacks import OptunaPruneCallback
from src.training.dataset_utils import load_dataset
from src.training.loops import run_training
import sys

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

system_type = "pcc_ns-2"
# dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp", "node-cornn", "node-con",
# "node-lnn", "node-general-lss", "node-mechanical-lss", "discrete-mlp"]
dynamics_model_name = "node-con"

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

max_num_epochs = 50
warmup_epochs = 5
batch_size = 100

now = datetime.now()
experiment_name = f"{system_type}_state_space_dynamics"
datetime_str = f"{now:%Y-%m-%d_%H-%M-%S}"
study_id = f"study-{experiment_name}-{datetime_str}"  # Unique identifier of the study.
logdir = Path("logs").resolve() / experiment_name / datetime_str
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_pcs_ns-{num_segments}.dill"
)

if __name__ == "__main__":
    # define the objective function for hyperparameter tuning
    def objective(trial):
        # re-seed tensorflow
        tf.random.set_seed(seed=seed)

        # Sample hyperparameters
        base_lr = trial.suggest_float("base_lr", 1e-5, 1e-1, log=True)
        # loss weights
        mse_q_weight = trial.suggest_float("mse_q_weight", 1e-8, 1e3, log=True)
        # weight decay
        weight_decay = trial.suggest_float("weight_decay", 5e-6, 2e-4, log=True)
        # initialize the loss weights
        loss_weights = dict(
            mse_q=mse_q_weight,
            mse_q_d=1.0,
        )

        start_time_idx = 0

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

        # dimension of the configuration space
        n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
        # dimension of the control input
        n_tau = train_ds.element_spec["tau"].shape[-1]

        # get the dynamics function
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = planar_pcs.factory(
            sym_exp_filepath, strain_selector=dataset_metadata["strain_selector"]
        )

        # initialize the neural networks
        if dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp"]:
            num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 6)
            mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 96)
            mlp_nonlinearity_name = trial.suggest_categorical(
                "mlp_nonlinearity",
                ["leaky_relu", "relu", "tanh", "sigmoid", "elu", "selu"],
            )
            mlp_nonlinearity = getattr(nn, mlp_nonlinearity_name)

            nn_model = MlpOde(
                latent_dim=n_q,
                input_dim=n_tau,
                num_layers=num_mlp_layers,
                hidden_dim=mlp_hidden_dim,
                nonlinearity=mlp_nonlinearity,
                mechanical_system=True
                if dynamics_model_name == "node-mechanical-mlp"
                else False,
            )
        elif dynamics_model_name == "node-cornn":
            cornn_gamma = trial.suggest_float("cornn_gamma", 1e-2, 1e2, log=True)
            cornn_epsilon = trial.suggest_float("cornn_epsilon", 1e-2, 1e2, log=True)
            nn_model = CornnOde(
                latent_dim=n_q,
                input_dim=n_tau,
                gamma=cornn_gamma,
                epsilon=cornn_epsilon,
            )
        elif dynamics_model_name == "node-con":
            nonlinearity_name = trial.suggest_categorical(
                "nonlinearity",
                ["leaky_relu", "relu", "tanh", "sigmoid", "elu", "selu", "softplus"],
            )
            diag_shift = trial.suggest_float("diag_shift", 1e-6, 1e-2, log=True)
            diag_eps = trial.suggest_float("diag_eps", 1e-6, 1e-2, log=True)
            nn_model = ConOde(
                latent_dim=n_q,
                input_dim=n_tau,
                nonlinearity=getattr(nn, nonlinearity_name),
                diag_shift=diag_shift,
                diag_eps=diag_eps,
            )
        elif dynamics_model_name == "node-lnn":
            # learn_dissipation = trial.suggest_categorical(
            #     "learn_dissipation", [True, False]
            # )
            learn_dissipation = True
            num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 6)
            mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 72)
            # attention: we are not allowed to use non-continuous activation functions like ReLU, leaky ReLU, or ELU
            # the second derivative of the activation function must be continuous
            # otherwise, the second derivative of the potential is not continuous and with that the Hessian
            # of the potential is not symmetric
            mlp_nonlinearity_name = trial.suggest_categorical(
                "mlp_nonlinearity",
                ["tanh", "sigmoid", "softplus"],
            )
            diag_shift = trial.suggest_float("diag_shift", 1e-6, 1e-2, log=True)
            diag_eps = trial.suggest_float("diag_eps", 1e-6, 1e-2, log=True)
            nn_model = LnnOde(
                latent_dim=n_q,
                input_dim=n_tau,
                learn_dissipation=learn_dissipation,
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
        elif dynamics_model_name == "discrete-mlp":
            num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 6)
            mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 4, 96)
            mlp_nonlinearity_name = trial.suggest_categorical(
                "mlp_nonlinearity",
                ["leaky_relu", "relu", "tanh", "sigmoid", "elu", "selu"],
            )
            mlp_nonlinearity = getattr(nn, mlp_nonlinearity_name)

            nn_model = DiscreteMlpDynamics(
                state_dim=2 * n_q,
                input_dim=n_tau,
                output_dim=2 * n_q,
                dt=dataset_metadata["dt"],
                num_layers=num_mlp_layers,
                hidden_dim=mlp_hidden_dim,
                nonlinearity=mlp_nonlinearity,
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

        # add the optuna prune callback
        prune_callback = OptunaPruneCallback(trial, metric_name="rmse_q_d_norm_val")
        callbacks = [prune_callback]

        print(f"Running trial {trial.number}...")
        (state, history, elapsed) = run_training(
            rng=rng,
            train_ds=train_ds,
            val_ds=val_ds,
            task_callables=task_callables,
            metrics_collection_cls=metrics_collection_cls,
            num_epochs=max_num_epochs,
            nn_model=nn_model,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            weight_decay=weight_decay,
            callbacks=callbacks,
            logdir=None,
            show_pbar=True,
        )

        (
            val_loss_stps,
            val_rmse_q_norm_stps,
            val_rmse_q_d_norm_stps,
        ) = history.collect(
            "loss_val",
            "rmse_q_norm",
            "rmse_q_d_norm",
        )
        print(
            f"Trial {trial.number} finished after {elapsed.steps} training steps with "
            f"validation loss: {val_loss_stps[-1]:.5f}, "
            f"rmse_q_norm: {val_rmse_q_norm_stps[-1]:.5f}, and rmse_q_d_norm: {val_rmse_q_d_norm_stps[-1]:.5f}"
        )

        return val_rmse_q_d_norm_stps[-1]

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage_name = f"sqlite:///{logdir}/optuna_study.db"
    sampler = TPESampler(seed=seed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(
        study_name=study_id,
        sampler=sampler,
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=500),
        storage=storage_name,
    )  # Create a new study.

    print(f"Run hyperparameter tuning with storage in {storage_name}...")
    study.optimize(
        objective, n_trials=1000
    )  # Invoke optimization of the objective function.

    with open(logdir / "optuna_study.dill", "wb") as f:
        dill.dump(study, f)
