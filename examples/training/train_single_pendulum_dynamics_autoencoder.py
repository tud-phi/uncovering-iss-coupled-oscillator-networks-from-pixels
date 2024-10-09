from datetime import datetime
import flax.linen as nn
import jax

# jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from jsrm.systems import pendulum
from pathlib import Path
import tensorflow as tf

from src.models.autoencoders import Autoencoder, VAE
from src.models.discrete_forward_dynamics import (
    DiscreteConIaeCfaDynamics,
    DiscreteLssDynamics,
    DiscreteMambaDynamics,
    DiscreteMlpDynamics,
    DiscreteRnnDynamics,
)
from src.models.neural_odes import (
    ConOde,
    ConIaeOde,
    CornnOde,
    LnnOde,
    LinearStateSpaceOde,
    MambaOde,
    MlpOde,
)
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.tasks import dynamics_autoencoder
from src.training.dataset_utils import load_dataset
from src.training.loops import run_training

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

long_horizon_dataset = True
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
""" dynamics_model_name in [
    "node-general-mlp", "node-mechanical-mlp", "node-mechanical-mlp-s", 
    "node-cornn", "node-con", "node-w-con", "node-con-iae", "node-lnn", 
    "node-hippo-lss", "node-mamba",
    "discrete-mlp", "discrete-elman-rnn", "discrete-gru-rnn", "discrete-general-lss", "discrete-hippo-lss", "discrete-mamba",
    "ar-con-iae-cfa"
]
"""
dynamics_model_name = "discrete-mlp"
# size of latent space
n_z = 2

batch_size = 100
num_epochs = 50
warmup_epochs = 5
start_time_idx = 1
num_past_timesteps = 2

latent_velocity_source = "image-space-finite-differences"
num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 20, "leaky_relu"
cornn_gamma, cornn_epsilon = 1.0, 1.0
lnn_learn_dissipation = True
diag_shift, diag_eps = 1e-6, 2e-6
if long_horizon_dataset:
    # Attention: these hyperparameters are not optimized specifically for the single pendulum dataset
    if ae_type == "beta_vae":
        match dynamics_model_name:
            case "node-mechanical-mlp":
                n_z = 8
                base_lr = 0.007137268676917664
                loss_weights = dict(
                    mse_z=0.17701201082200202,
                    mse_rec_static=1.0,
                    mse_rec_dynamic=50.808302047597074,
                    beta=0.002678889167847793,
                )
                weight_decay = 4.5818408762378344e-05
                num_mlp_layers, mlp_hidden_dim = 5, 21
                mlp_nonlinearity_name = "tanh"
            case "node-w-con":
                """the following params might work even a bit better
                n_z = 32
                base_lr = 0.009903075976738526
                loss_weights = dict(
                    mse_z=0.35815329016032293,
                    mse_rec_static=1.0,
                    mse_rec_dynamic=61.967763503662795,
                    beta=0.00041128402956180964,
                )
                weight_decay = 1.4235324198580345e-05
                """
                n_z = 32
                base_lr = 0.009793849772267547
                loss_weights = dict(
                    mse_z=0.40568126085978073,
                    mse_rec_static=1.0,
                    mse_rec_dynamic=64.68788840647458,
                    beta=0.0002437097576124702,
                )
                weight_decay = 1.3691415073322272e-05
            case "node-con-iae":
                # optimized for n_z=8
                base_lr = 0.018486990918444367
                loss_weights = dict(
                    mse_z=0.3733687489479885,
                    mse_rec_static=1.0,
                    mse_rec_dynamic=83.7248326772002,
                    beta=0.00020068384639167935,
                    mse_tau_rec=1e1,
                )
                weight_decay = 5.5340117045438595e-06
                num_mlp_layers, mlp_hidden_dim = 5, 30
            case "ar-con-iae-cfa":
                # optimized for n_z=8
                base_lr = 0.018088317332901616
                loss_weights = dict(
                    mse_z=0.10824911140537369,
                    mse_rec_static=1.0,
                    mse_rec_dynamic=80.45564515992584,
                    beta=0.00010659152931072577,
                    mse_tau_rec=1e1,
                )
                weight_decay = 2.6404635847920316e-05
                num_mlp_layers, mlp_hidden_dim = 5, 30
            case _:
                raise NotImplementedError(
                    f"beta_vae with dynamics_model_name '{dynamics_model_name}' not implemented yet."
                )
    else:
        raise NotImplementedError(f"ae_type '{ae_type}' not implemented yet.")
else:
    if ae_type == "wae":
        raise NotImplementedError(f"ae_type '{ae_type}' not implemented yet.")
    elif ae_type == "beta_vae":
        if dynamics_model_name == "node-general-mlp":
            base_lr = 0.004245278743015398
            loss_weights = dict(
                mse_z=0.011179320698028615,
                mse_rec_static=1.0,
                mse_rec_dynamic=189.07672802272313,
                beta=0.00020732900159342376,
            )
            weight_decay = 7.942186445089656e-06
            latent_velocity_source = "image-space-finite-differences"
            num_mlp_layers = 4
            mlp_hidden_dim = 40
        elif dynamics_model_name == "node-mechanical-mlp":
            base_lr = 0.004515967701381585
            loss_weights = dict(
                mse_z=0.10726653464324469,
                mse_rec_static=1.0,
                mse_rec_dynamic=455.96650828264166,
                beta=0.01663158737179567,
            )
            weight_decay = 8.746966135967026e-06
            latent_velocity_source = "latent-space-finite-differences"
            num_mlp_layers = 4
            mlp_hidden_dim = 46
        elif dynamics_model_name == "node-cornn":
            base_lr = 0.00398427112108674
            loss_weights = dict(
                mse_z=0.2548882784224465,
                mse_rec_static=1.0,
                mse_rec_dynamic=8.493464866080236,
                beta=0.0009692016013541893,
            )
            weight_decay = 0.00010890363692419105
            latent_velocity_source = "image-space-finite-differences"
            cornn_gamma = 14.699222042132245
            cornn_epsilon = 1.122193753584045
        elif dynamics_model_name == "node-con":
            base_lr = 0.006720897010650257
            loss_weights = dict(
                mse_z=0.03020580710811323,
                mse_rec_static=1.0,
                mse_rec_dynamic=113.57201294093048,
                beta=0.00038589900089786266,
            )
            weight_decay = 0.00019609847803674207
            latent_velocity_source = "image-space-finite-differences"
        elif dynamics_model_name == "node-lnn":
            base_lr = 0.0015553597576502523
            loss_weights = dict(
                mse_z=0.24740396032120054,
                mse_rec_static=1.0,
                mse_rec_dynamic=19.997472562384235,
                beta=0.000148761017920335,
            )
            weight_decay = 1.367664507404463e-05
            latent_velocity_source = "image-space-finite-differences"
            lnn_learn_dissipation = True
            num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 13, "softplus"
            diag_shift, diag_eps = 1.3009374296641844e-06, 1.4901550009073945e-05
        elif dynamics_model_name in [
            "node-general-lss",
            "node-mechanical-lss",
            "node-hippo-lss",
        ]:
            base_lr = 0.009140398915788182
            loss_weights = dict(
                mse_z=0.3540013026659153,
                mse_rec_static=1.0,
                mse_rec_dynamic=3.8239959063309903,
                beta=0.0004775274363009053,
            )
            weight_decay = 5.409956968011885e-06
            latent_velocity_source = "image-space-finite-differences"
        elif dynamics_model_name == "discrete-mlp":
            base_lr = 0.006092601805515173
            loss_weights = dict(
                mse_z=0.06416478341882806,
                mse_rec_static=1.0,
                mse_rec_dynamic=190.29900573734943,
                beta=0.0003989687418962098,
            )
            weight_decay = 1.4942423494736107e-05
            latent_velocity_source = "image-space-finite-differences"
            num_mlp_layers = 4
            mlp_hidden_dim = 90
            mlp_nonlinearity_name = "leaky_relu"
        else:
            raise NotImplementedError(
                f"beta_vae with node_type '{dynamics_model_name}' not implemented yet."
            )
    else:
        raise NotImplementedError(f"ae_type '{ae_type}' not implemented yet.")

# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in [
    "node",
    "discrete",
    "ar",
], f"Unknown dynamics_type: {dynamics_type}"

now = datetime.now()
logdir = (
    Path("logs").resolve()
    / "single_pendulum_dynamics_autoencoder"
    / f"{now:%Y-%m-%d_%H-%M-%S}"
)
logdir.mkdir(parents=True, exist_ok=True)

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"pendulum_nl-1.dill"
)

if __name__ == "__main__":
    if long_horizon_dataset:
        dataset_name = "pendulum/single_pendulum_32x32px_h-101"
    else:
        dataset_name = "pendulum/single_pendulum_32x32px"
    datasets, dataset_info, dataset_metadata = load_dataset(
        dataset_name,
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
    # size of torques
    n_tau = train_ds.element_spec["tau"].shape[-1]  # dimension of the control input
    print(f"Control input dimension: {n_tau}")
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape

    # get the dynamics function
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    # initialize the neural networks
    if ae_type == "beta_vae":
        autoencoder_model = VAE(
            latent_dim=n_z, img_shape=img_shape, norm_layer=nn.LayerNorm
        )
    else:
        autoencoder_model = Autoencoder(
            latent_dim=n_z, img_shape=img_shape, norm_layer=nn.LayerNorm
        )
    if dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp"]:
        dynamics_model = MlpOde(
            latent_dim=n_z,
            input_dim=n_tau,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
            mechanical_system=True
            if dynamics_model_name == "node-mechanical-mlp"
            else False,
        )
    elif dynamics_model_name == "node-cornn":
        dynamics_model = CornnOde(
            latent_dim=n_z,
            input_dim=n_tau,
            gamma=cornn_gamma,
            epsilon=cornn_epsilon,
        )
    elif dynamics_model_name in ["node-con", "node-w-con"]:
        dynamics_model = ConOde(
            latent_dim=n_z,
            input_dim=n_tau,
            use_w_coordinates=dynamics_model_name == "node-w-con",
        )
    elif dynamics_model_name in ["node-con-iae"]:
        dynamics_model = ConIaeOde(
            latent_dim=n_z,
            input_dim=n_tau,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
        )
    elif dynamics_model_name == "node-lnn":
        dynamics_model = LnnOde(
            latent_dim=n_z,
            input_dim=n_tau,
            learn_dissipation=lnn_learn_dissipation,
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
        dynamics_model = LinearStateSpaceOde(
            latent_dim=n_z,
            input_dim=n_tau,
            transition_matrix_init=dynamics_model_name.split("-")[
                1
            ],  # "general", "mechanical", or "hippo"
        )
    elif dynamics_model_name == "discrete-mlp":
        dynamics_model = DiscreteMlpDynamics(
            state_dim=num_past_timesteps * n_z,
            input_dim=n_tau,
            output_dim=n_z,
            dt=dataset_metadata["dt"],
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
        )
    elif dynamics_model_name == "ar-con-iae-cfa":
        dynamics_model = DiscreteConIaeCfaDynamics(
            latent_dim=n_z,
            input_dim=n_tau,
            dt=dataset_metadata["sim_dt"],
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
        )
    else:
        raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")
    nn_model = DynamicsAutoencoder(
        autoencoder=autoencoder_model,
        dynamics=dynamics_model,
        dynamics_type=dynamics_type,
        num_past_timesteps=num_past_timesteps,
    )

    # import solver class from diffrax
    # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    solver_class = getattr(
        __import__("diffrax", fromlist=[dataset_metadata["solver_class"]]),
        dataset_metadata["solver_class"],
    )

    # call the factory function for the sensing task
    task_callables, metrics_collection_cls = dynamics_autoencoder.task_factory(
        "pendulum",
        nn_model,
        ts=dataset_metadata["ts"],
        sim_dt=dataset_metadata["sim_dt"],
        loss_weights=loss_weights,
        ae_type=ae_type,
        dynamics_type=dynamics_type,
        start_time_idx=start_time_idx,
        solver=solver_class(),
        latent_velocity_source=latent_velocity_source,
        num_past_timesteps=num_past_timesteps,
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
        init_fn=nn_model.forward_all_layers,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        logdir=logdir,
    )
    print("Final training metrics:\n", state.metrics.compute())
