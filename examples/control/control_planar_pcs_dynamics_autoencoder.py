from flax.core import FrozenDict
import flax.linen as nn

from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'
from jax import Array, jit, random
import jax.numpy as jnp
import jsrm
from jsrm.integration import ode_with_forcing_factory
from jsrm.systems import planar_pcs
import numpy as onp
from pathlib import Path
import tensorflow as tf
from typing import Dict, Tuple, Union

from src.models.autoencoders import Autoencoder, VAE
from src.models.neural_odes import (
    ConOde,
    CornnOde,
    LnnOde,
    LinearStateSpaceOde,
    MambaOde,
    MlpOde,
)
from src.models.neural_odes.utils import generate_positive_definite_matrix_from_params
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.rendering import render_planar_pcs
from src.rollout import rollout_ode, rollout_ode_with_latent_space_control
from src.training.dataset_utils import load_dataset, load_dummy_neural_network_input
from src.training.loops import run_eval
from src.tasks import dynamics_autoencoder
from src.training.train_state_utils import restore_train_state
from src.visualization.img_animation import (
    animate_pred_vs_target_image_pyplot,
)

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

seed = 0
rng = random.PRNGKey(seed=seed)
tf.random.set_seed(seed=seed)

system_type = "pcc_ns-2"
ae_type = "beta_vae"  # "None", "beta_vae", "wae"
dynamics_model_name = "node-w-con"
# latent space shape
n_z = 4

batch_size = 10
norm_layer = nn.LayerNorm
cornn_gamma, cornn_epsilon = 1.0, 1.0
lnn_learn_dissipation = True
diag_shift, diag_eps = 1e-6, 2e-6
if ae_type == "wae":
    raise NotImplementedError
elif ae_type == "beta_vae":
    if dynamics_model_name == "node-con":
        experiment_id = "2024-02-14_18-34-27"
    elif dynamics_model_name == "node-w-con":
        experiment_id = "2024-02-21_13-34-53"
    else:
        raise NotImplementedError(
            f"beta_vae with node_type '{dynamics_model_name}' not implemented yet."
        )
else:
    raise NotImplementedError

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
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_pcs_ns-{num_segments}.dill"
)
ckpt_dir = (
    Path("logs").resolve() / f"{system_type}_dynamics_autoencoder" / experiment_id
)


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
    # dimension of the configuration space
    n_q = train_ds.element_spec["x_ts"].shape[-1] // 2
    # size of torques
    n_tau = train_ds.element_spec["tau"].shape[-1]  # dimension of the control input
    # image shape
    img_shape = train_ds.element_spec["rendering_ts"].shape[-3:]  # image shape

    # get the dynamics function
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.factory(
        sym_exp_filepath, strain_selector=dataset_metadata["strain_selector"]
    )
    ode_fn = ode_with_forcing_factory(dynamical_matrices_fn, robot_params)

    # initialize the rendering function
    rendering_fn = partial(
        render_planar_pcs,
        forward_kinematics_fn,
        robot_params,
        width=img_shape[0],
        height=img_shape[0],
        origin_uv=dataset_metadata["rendering"]["origin_uv"],
        line_thickness=dataset_metadata["rendering"]["line_thickness"],
    )

    # initialize the neural networks
    if ae_type == "beta_vae":
        autoencoder_model = VAE(
            latent_dim=n_z, img_shape=img_shape, norm_layer=norm_layer
        )
    else:
        autoencoder_model = Autoencoder(
            latent_dim=n_z, img_shape=img_shape, norm_layer=norm_layer
        )
    if dynamics_model_name == "node-cornn":
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

    # define settings for the closed-loop simulation
    sim_duration = 3.0  # s
    control_dt = 1e-2  # control and time step of 1e-2 s
    sim_dt = 1e-3 * control_dt  # simulation time step of 1e-5 s
    ts = jnp.linspace(0.0, sim_duration, num=int(sim_duration / control_dt))
    ode_rollout_fn = partial(
        rollout_ode,
        ode_fn=ode_fn,
        ts=ts,
        sim_dt=sim_dt,
        rendering_fn=rendering_fn,
        solver=solver_class(),
        show_progress=True,
    )
    # define the task callables for the rollout
    (
        task_callables_rollout_learned,
        metrics_collection_cls,
    ) = dynamics_autoencoder.task_factory(
        system_type,
        nn_model,
        ts=ts,
        sim_dt=sim_dt,
        ae_type=ae_type,
        dynamics_type=dynamics_type,
        solver=solver_class(),
        latent_velocity_source="image-space-finite-differences",
    )
    # load the neural network dummy input
    nn_dummy_input = load_dummy_neural_network_input(
        test_ds, task_callables_rollout_learned
    )
    # load the training state from the checkpoint directory
    state = restore_train_state(
        rng=rng,
        ckpt_dir=ckpt_dir,
        nn_model=nn_model,
        nn_dummy_input=nn_dummy_input,
        metrics_collection_cls=metrics_collection_cls,
        init_fn=nn_model.initialize_all_weights,
    )
    forward_fn_learned = jit(task_callables_rollout_learned.forward_fn)


    def control_fn(t: Array, x: Array) -> Tuple[Array, Dict[str, Array]]:
        """
        Control function for the setpoint regulation.
        Args:
            t: current time
            x: current state of the system
        Returns:
            tau: control input
            control_info: dictionary with control information
        """
        # compute the control input
        tau, control_info = dynamics_model.apply(
            {"params": state.params["dynamics"]},
            x,
            z_des,
            kp=0.0,
            kd=0.0,
            method=dynamics_model.setpoint_regulation_control_fn,
        )
        return tau, control_info

    # render target image
    q_des = jnp.zeros((n_q,))
    target_img = rendering_fn(q_des)
    # normalize the target image
    # convert rendering image to grayscale
    target_img = tf.image.rgb_to_grayscale(target_img)
    # normalize rendering image to [0, 1]
    target_img = tf.cast(target_img, tf.float32) / 128.0 - 1.0
    # convert image to jax array
    target_img = jnp.array(target_img)
    # encode the target image
    target_img_bt = target_img[None, ...]
    z_des_bt = nn_model.apply(
        {"params": state.params}, target_img_bt, method=nn_model.encode
    )
    z_des = z_des_bt[0, :]

    # set initial condition for closed-loop simulation
    q0 = (
        0.1
        * jnp.tile(jnp.array([1.0, -1.0]), reps=int(jnp.ceil(n_q / 2)))[:n_q]
        * dataset_metadata["x0_max"][:n_q]
    )
    x0 = jnp.concatenate([q0, jnp.zeros((n_q,))])

    # start closed-loop simulation
    print("Simulating closed-loop dynamics...")
    sim_ts = rollout_ode_with_latent_space_control(
        ode_fn=ode_fn,
        rendering_fn=rendering_fn,
        encode_fn=jit(
            partial(
                nn_model.apply,
                {"params": state.params},
                method=nn_model.encode,
            )
        ),
        ts=ts,
        sim_dt=sim_dt,
        x0=x0,
        input_dim=n_tau,
        latent_dim=n_z,
        control_fn=jit(control_fn),
    )

    # extract both the ground-truth and the statically predicted images
    img_ts = onp.array(sim_ts["rendering_ts"])
    img_des_ts = onp.array(jnp.tile(target_img, reps=(img_ts.shape[0], 1, 1, 1)))

    # animate the rollout
    print("Animate the rollout...")
    animate_pred_vs_target_image_pyplot(
        onp.array(ts),
        img_pred_ts=img_ts,
        img_target_ts=img_des_ts,
        filepath=ckpt_dir / "controlled_rollout.mp4",
        step_skip=1,
        show=True,
        label_pred="Actual behavior",
        label_target="Desired behavior",
    )