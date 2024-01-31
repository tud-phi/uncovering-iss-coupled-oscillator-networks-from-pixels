import flax.linen as nn
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
from jax import Array, random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import pendulum
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

from src.models.autoencoders import Autoencoder, VAE
from src.models.discrete_forward_dynamics import DiscreteMlpDynamics
from src.models.neural_odes import ConOde, CornnOde, LnnOde, MlpOde
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.training.dataset_utils import load_dataset, load_dummy_neural_network_input
from src.training.loops import run_eval
from src.tasks import dynamics_autoencoder
from src.training.train_state_utils import restore_train_state
from src.visualization.latent_space import (
    visualize_mapping_from_configuration_to_latent_space,
)

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

ae_type = "beta_vae"  # "None", "beta_vae", "wae"
# dynamics_model_name in ["node-general-mlp", "node-mechanical-mlp", "node-cornn", "node-con", "node-lnn", "discrete-mlp"]
dynamics_model_name = "node-lnn"
# latent space shape
n_z = 2

batch_size = 10
loss_weights = dict(mse_q=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0)
start_time_idx = 1
num_past_timesteps = 2

norm_layer = nn.LayerNorm
num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 20, "leaky_relu"
cornn_gamma, cornn_epsilon = 1.0, 1.0
lnn_learn_dissipation = True
diag_shift, diag_eps = 1e-6, 2e-6
if ae_type == "wae":
    raise NotImplementedError
elif ae_type == "beta_vae":
    if dynamics_model_name == "node-general-mlp":
        experiment_id = "2024-01-26_10-16-46"
        num_mlp_layers = 4
        mlp_hidden_dim = 40
    elif dynamics_model_name == "node-mechanical-mlp":
        experiment_id = "2024-01-30_16-14-20"
        num_mlp_layers = 2
        mlp_hidden_dim = 2
    elif dynamics_model_name == "node-cornn":
        experiment_id = "2024-01-27_10-30-52"
        cornn_gamma, cornn_epsilon = 14.699222042132245, 1.122193753584045
    elif dynamics_model_name == "node-con":
        experiment_id = "2024-01-28_22-19-27"
    elif dynamics_model_name == "node-lnn":
        experiment_id = "2024-01-31_10-39-39"
        lnn_learn_dissipation = True
        num_mlp_layers, mlp_hidden_dim, mlp_nonlinearity_name = 4, 13, "relu"
        diag_shift, diag_eps = 1.3009374296641844e-06, 1.4901550009073945e-05
    elif dynamics_model_name == "discrete-mlp":
        experiment_id = "2024-01-29_19-31-02"
        num_mlp_layers = 4
        mlp_hidden_dim = 90
        mlp_nonlinearity_name = "leaky_relu"
    else:
        raise NotImplementedError(
            f"beta_vae with node_type '{dynamics_model_name}' not implemented yet."
        )
else:
    raise NotImplementedError

# identify the dynamics_type
dynamics_type = dynamics_model_name.split("-")[0]
assert dynamics_type in ["node", "discrete"], f"Unknown dynamics_type: {dynamics_type}"

sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / f"pendulum_nl-1.dill"
)
ckpt_dir = (
    Path("logs").resolve() / "single_pendulum_dynamics_autoencoder" / experiment_id
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
    # size of torques
    n_tau = train_ds.element_spec["tau"].shape[-1]  # dimension of the control input
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape

    # get the dynamics function
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)

    # initialize the neural networks
    if ae_type == "beta_vae":
        autoencoder_model = VAE(
            latent_dim=n_z, img_shape=img_shape, norm_layer=norm_layer
        )
    else:
        autoencoder_model = Autoencoder(
            latent_dim=n_z, img_shape=img_shape, norm_layer=norm_layer
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
    elif dynamics_model_name == "node-con":
        dynamics_model = ConOde(
            latent_dim=n_z,
            input_dim=n_tau,
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
    elif dynamics_model_name == "discrete-mlp":
        dynamics_model = DiscreteMlpDynamics(
            latent_dim=n_z,
            input_dim=n_tau,
            output_dim=n_z,
            dt=dataset_metadata["dt"],
            num_past_timesteps=num_past_timesteps,
            num_layers=num_mlp_layers,
            hidden_dim=mlp_hidden_dim,
            nonlinearity=getattr(nn, mlp_nonlinearity_name),
        )
    else:
        raise ValueError(f"Unknown dynamics_model_name: {dynamics_model_name}")
    nn_model = DynamicsAutoencoder(
        autoencoder=autoencoder_model,
        dynamics=dynamics_model,
        dynamics_type=dynamics_type,
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
        latent_velocity_source="image-space-finite-differences",
        num_past_timesteps=num_past_timesteps,
    )

    # load the neural network dummy input
    nn_dummy_input = load_dummy_neural_network_input(test_ds, task_callables)
    # load the training state from the checkpoint directory
    state = restore_train_state(
        rng=rng,
        ckpt_dir=ckpt_dir,
        nn_model=nn_model,
        nn_dummy_input=nn_dummy_input,
        metrics_collection_cls=metrics_collection_cls,
        init_fn=nn_model.initialize_all_weights,
    )

    print("Run testing...")
    state, test_history = run_eval(test_ds, state, task_callables)
    test_metrics = state.metrics.compute()
    print(
        "\n"
        f"Final test metrics:\n"
        f"rmse_rec_static={test_metrics['rmse_rec_static']:.4f}, "
        f"rmse_rec_dynamic={test_metrics['rmse_rec_dynamic']:.4f}"
    )

    test_batch = next(test_ds.as_numpy_iterator())
    test_preds = task_callables.forward_fn(test_batch, state.params)

    for i in range(test_batch["rendering_ts"].shape[0]):
        print("Trajectory:", i)

        for t in range(start_time_idx, test_batch["rendering_ts"].shape[1]):
            print("Time step:", t)

            img_gt = (128 * (1.0 + test_batch["rendering_ts"][i, t])).astype(jnp.uint8)
            img_rec = (
                128 * (1.0 + test_preds["rendering_dynamic_ts"][i, t - start_time_idx])
            ).astype(jnp.uint8)

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            img_gt_plot = axes[0].imshow(img_gt, vmin=0, vmax=255)
            plt.colorbar(img_gt_plot, ax=axes[0])
            axes[0].set_title("Original")
            img_rec_plot = axes[1].imshow(img_rec, vmin=0, vmax=255)
            plt.colorbar(img_rec_plot, ax=axes[1])
            axes[1].set_title("Reconstruction")
            plt.show()
