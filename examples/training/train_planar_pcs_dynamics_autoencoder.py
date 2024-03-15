from datetime import datetime
import flax.linen as nn
import jax

# jax.config.update("jax_platform_name", "cpu")  # set default device to 'cpu'
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import jsrm
from pathlib import Path
import tensorflow as tf

from src.models.autoencoders import Autoencoder, VAE
from src.models.discrete_forward_dynamics import (
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

system_type = "pcc_ns-2"
long_horizon_dataset = True
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
""" dynamics_model_name in [
    "node-general-mlp", "node-mechanical-mlp", "node-mechanical-mlp-s", 
    "node-cornn", "node-con", "node-w-con", "node-con-iae", "node-dcon", "node-lnn", 
    "node-hippo-lss", "node-mamba",
    "discrete-mlp", "discrete-elman-rnn", "discrete-gru-rnn", "discrete-general-lss", "discrete-hippo-lss", "discrete-mamba",
]
"""
dynamics_model_name = "node-mechanical-mlp"
# size of latent space
n_z = 4
# simulation time step
sim_dt = 1e-2

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
            num_mlp_layers = 4
            mlp_hidden_dim = 40
            raise NotImplementedError
        elif dynamics_model_name == "node-mechanical-mlp":
            base_lr = 0.009549630971301099
            loss_weights = dict(
                mse_z=0.15036907451864656,
                mse_rec_static=1.0,
                mse_rec_dynamic=16.356448652349172,
                beta=0.00014574221959894125,
            )
            weight_decay = 5.1572222268612065e-05
            num_mlp_layers, mlp_hidden_dim = 4, 52
            mlp_nonlinearity_name = "elu"
        elif dynamics_model_name == "node-cornn":
            base_lr = 0.0032720052876344437
            loss_weights = dict(
                mse_z=0.44777585091731187,
                mse_rec_static=1.0,
                mse_rec_dynamic=2.511229022994574,
                beta=0.00011714626957666846,
            )
            weight_decay = 1.8390286588494643e-05
            cornn_gamma, cornn_epsilon = 35.60944428175452, 0.05125440449424828
        elif dynamics_model_name in ["node-con", "node-w-con"]:
            """ node-w-con with n_z > 4
            rmse_rec_dynamic_val = 0.14014695584774017
            n_z = 17
            base_lr = 0.009575159163417718
            loss_weights = dict(
                mse_z=0.10188200495675905,
                mse_rec_static=1.0,
                mse_rec_dynamic=3.3080609062995894,
                beta=0.0001718351163778155,
            )
            weight_decay = 9.534255318218664e-06
            """
            if n_z > 4:
                base_lr = 0.009575159163417718
                loss_weights = dict(
                    mse_z=0.10188200495675905,
                    mse_rec_static=1.0,
                    mse_rec_dynamic=3.3080609062995894,
                    beta=0.0001718351163778155,
                )
                weight_decay = 9.534255318218664e-06
            else:
                base_lr = 0.00948905454174363
                loss_weights = dict(
                    mse_z=0.25452411365868755,
                    mse_rec_static=1.0,
                    mse_rec_dynamic=10.716251946521883,
                    beta=0.0001000304459146119,
                )
                weight_decay = 2.9642914376858733e-05
        elif dynamics_model_name == "node-lnn":
            base_lr = 0.002922002372648181
            loss_weights = dict(
                mse_z=0.45534786191007814,
                mse_rec_static=1.0,
                mse_rec_dynamic=8.181311383245621,
                beta=0.012987943339071037,
            )
            weight_decay = 7.869361804893107e-06
            lnn_learn_dissipation = True
            num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 6, 9, "sigmoid"
            diag_shift, diag_eps = 2.3522236095556114e-06, 0.0019146626034900816
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
            raise NotImplementedError
        elif dynamics_model_name == "discrete-mlp":
            base_lr = 0.008868218513411644
            loss_weights = dict(
                mse_z=0.41624019460716366,
                mse_rec_static=1.0,
                mse_rec_dynamic=407.1895196862229,
                beta=0.03405228893154261,
            )
            weight_decay = 0.00018061847705335356
            num_mlp_layers, mlp_hidden_dim = 4, 95
            mlp_nonlinearity_name = "elu"
        elif dynamics_model_name == "discrete-elman-rnn":
            base_lr = 0.009562362872368196
            loss_weights = dict(
                mse_z=0.4515819661074938,
                mse_rec_static=1.0,
                mse_rec_dynamic=45.25873190730584,
                beta=0.001817925663163544,
            )
            weight_decay = 0.00015443793550364007
        elif dynamics_model_name == "discrete-gru-rnn":
            base_lr = 0.0061904901667741855
            loss_weights = dict(
                mse_z=0.0791729093402154,
                mse_rec_static=1.0,
                mse_rec_dynamic=34.12991695226881,
                beta=0.00022655846366566662,
            )
            weight_decay = 0.0001519957156945279
        elif dynamics_model_name == "discrete-mamba":
            base_lr = 0.00870873016301107
            loss_weights = dict(
                mse_z=0.28214113853521156,
                mse_rec_static=1.0,
                mse_rec_dynamic=33.60427838050405,
                beta=0.0007276292657337367,
            )
            weight_decay = 2.360420656597323e-05
        else:
            raise NotImplementedError(
                f"beta_vae with node_type '{dynamics_model_name}' not implemented yet."
            )
    else:
        raise NotImplementedError(f"ae_type '{ae_type}' not implemented yet.")

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
    / f"{system_type}_dynamics_autoencoder"
    / f"{now:%Y-%m-%d_%H-%M-%S}"
)
logdir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    if long_horizon_dataset:
        dataset_name = f"planar_pcs/{system_type}_32x32px_h-101"
    else:
        dataset_name = f"planar_pcs/{system_type}_32x32px"
    datasets, dataset_info, dataset_metadata = load_dataset(
        dataset_name,
        seed=seed,
        batch_size=batch_size,
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
            input_dim=num_past_timesteps * n_tau,
            output_dim=n_z,
            dt=dataset_metadata["dt"],
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
        )
    elif dynamics_model_name in ["discrete-elman-rnn", "discrete-gru-rnn"]:
        dynamics_model = DiscreteRnnDynamics(
            state_dim=num_past_timesteps * n_z,
            input_dim=num_past_timesteps * n_tau,
            output_dim=n_z,
            rnn_method=dynamics_model_name.split("-")[1],  # "elman" or "gru"
        )
    elif dynamics_model_name == "discrete-mamba":
        dynamics_model = DiscreteMambaDynamics(
            state_dim=num_past_timesteps * n_z,
            input_dim=num_past_timesteps * n_tau,
            output_dim=n_z,
            dt=dataset_metadata["dt"],
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

    # call the factory function for the dynamics autoencoder task
    print(
        "Dataset dt:",
        dataset_metadata["dt"],
        "dataset sim_dt:",
        dataset_metadata["sim_dt"],
        "actually using sim_dt",
        sim_dt,
    )
    task_callables, metrics_collection_cls = dynamics_autoencoder.task_factory(
        system_type,
        nn_model,
        ts=dataset_metadata["ts"],
        sim_dt=sim_dt,
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
