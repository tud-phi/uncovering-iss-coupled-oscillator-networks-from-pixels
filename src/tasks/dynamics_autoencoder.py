from clu import metrics as clu_metrics
from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from flax.core import FrozenDict
from flax.struct import dataclass
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, debug, jit, jvp, lax, random, vmap
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple, Type

from src.losses.kld import kullback_leiber_divergence
from src.losses.psnr import peak_signal_to_noise_ratio
from src.losses.ssim import structural_similarity_index
from src.metrics import RootAverage
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.models.discrete_forward_dynamics import DiscreteConIaeCfaDynamics
from src.models.neural_odes import ConIaeOde
from src.structs import TaskCallables


def assemble_input(batch) -> Tuple[Array]:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return (img_bt,)


def task_factory(
    system_type: str,
    nn_model: DynamicsAutoencoder,
    ts: Array,
    sim_dt: float,
    encode_fn: Optional[Callable] = None,
    decode_fn: Optional[Callable] = None,
    encode_kwargs: Optional[Dict[str, Any]] = None,
    decode_kwargs: Optional[Dict[str, Any]] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    ae_type: str = "None",
    dynamics_type: str = "node",  # node or discrete
    dynamics_order: int = 2,
    start_time_idx: int = 1,
    solver: AbstractSolver = Dopri5(),
    latent_velocity_source: str = "latent-space-finite-differences",
    num_past_timesteps: int = 2,
    interpret_discrete_hidden_state_as_displacement: bool = False,
    compute_psnr: bool = False,
    compute_ssim: bool = False,
) -> Tuple[TaskCallables, Type[clu_metrics.Collection]]:
    """
    Factory function for the task of learning a representation using first-principle dynamics while using the
    ground-truth velocity.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics_fn functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        nn_model: the neural network model of the dynamics autoencoder. Should contain both the autoencoder and the neural ODE.
        ts: time steps of the samples
        sim_dt: Time step used for simulation [s].
        encode_fn: the function to use for encoding the input image to the latent space
        decode_fn: the function to use for decoding the latent space to the output image
        encode_kwargs: additional kwargs to pass to the encode_fn
        decode_kwargs: additional kwargs to pass to the decode_fn
        loss_weights: the weights for the different loss terms
        ae_type: Autoencoder type. If None, a normal autoencoder will be used.
            One of ["wae", "beta_vae", "None"]
        dynamics_type: Dynamics type. One of ["node", "discrete"]
            node: Neural ODE dynamics map a state consisting of latent and their velocity to the state derivative.
            discrete: Discrete forward dynamics map the latents at the num_past_timesteps to the next latent.
            ar: Discrete forward dynamics maps autoregressively the latent state (latent position and velocity) to the
                next latent state at time step sim_dt.
        dynamics_order: the order of the dynamics. Either 1 or 2.
            By default, the dynamics are of order 2 (i.e., the state consists of the latent position and velocity).
            If the dynamics are of order 1, the state consists only of the latent position.
        start_time_idx: the index of the time step to start the simulation at. Needs to be >=1 to enable the application
            of finite differences for the latent-space velocity.
        solver: Diffrax solver to use for the simulation. Only use for the neural ODE.
        latent_velocity_source: the source of the latent velocity. Only used for the neural ODE.
            Can be either "latent-space-finite-differences", or "image-space-finite-differences". Only active if
            dynamics_order=2.
        num_past_timesteps: the number of past timesteps to use for the discrete forward dynamics when dynamics_order=2.
            When dynamics_order=1, we set num_past_timesteps=1.
        interpret_discrete_hidden_state_as_displacement: whether to interpret the hidden state of the discrete forward dynamics
            as differences (i.e., deltas) between actual physical states. It has been shown that this can improve the
            performance of RNNs, see:
            J. Martinez, M. J. Black, and J. Romero, “On human motion prediction using recurrent neural networks,”
            in Proc. IEEE Conf. on Comput. Vis. Pattern Recognit., 2017.
        compute_psnr: whether to compute the Peak Signal-to-Noise Ratio (PSNR) as a metric
        compute_ssim: whether to compute the Structural Similarity Index (SSIM) as a metric
    Returns:
        task_callables: struct containing the functions for the learning task
        metrics_collection_cls: contains class for collecting metrics
    """
    # make sure that dynamics order is either 1 or 2
    assert dynamics_order in [1, 2], "The dynamics order needs to be either 1 or 2."
    if dynamics_order == 1:
        num_past_timesteps = 1
        print(f"Warning: Setting num_past_timesteps to 1 as dynamics_order=1.")

    # time step between samples
    sample_dt = (ts[1:] - ts[:-1]).mean()
    # compute the dynamic rollout of the latent representation
    t0 = ts[start_time_idx]  # start time
    tf = ts[-1]  # end time
    # maximum of integrator steps
    max_int_steps = int((tf - t0) / sim_dt) + 1
    # make sure that the simulation time steps are consistent with the sample time steps
    assert (
        sample_dt >= sim_dt
    ), "The simulation time step needs to be smaller or equal to the sample time step."
    """
    assert (
        sample_dt % sim_dt == 0
    ), "The sample time step needs to be a multiple of the simulation time step."
    """
    sample_sim_skip_step = int(sample_dt // sim_dt)
    # simulation time stamps
    ts_sim = jnp.arange(t0, tf + sim_dt, sim_dt)

    if encode_fn is None:
        encode_fn = nn_model.encode
    if decode_fn is None:
        decode_fn = nn_model.decode
    if encode_kwargs is None:
        encode_kwargs = {}
    if decode_kwargs is None:
        decode_kwargs = {}

    if loss_weights is None:
        loss_weights = {}
    loss_weights = (
        dict(mse_z=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0) | loss_weights
    )

    if ae_type == "wae":
        loss_weights = dict(mmd=1.0) | loss_weights
        uniform_distr_range = (-1.0, 1.0)

        from src.losses import wae

        wae_mmd_loss_fn = wae.make_wae_mdd_loss(
            distribution="uniform", uniform_distr_range=uniform_distr_range
        )
    elif ae_type == "beta_vae":
        loss_weights = dict(beta=1.0) | loss_weights

    if dynamics_type == "discrete":
        assert (
            start_time_idx >= num_past_timesteps - 1
        ), "The start time idx needs to be >= num_past_timesteps - 1 to enable the passing of a sufficient number of inputs to the model."

    def forward_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[Array] = None,
        training: bool = False,
    ) -> Dict[str, Array]:
        img_bt = batch["rendering_ts"]
        (img_flat_bt,) = assemble_input(batch)

        batch_size = batch["rendering_ts"].shape[0]
        n_z = nn_model.autoencoder.latent_dim

        # static predictions by passing the image through the encoder
        # output will be of shape batch_dim * time_dim x latent_dim
        if ae_type == "beta_vae":
            # output will be of shape batch_dim * time_dim x latent_dim
            mu_static_bt, logvar_static_bt = nn_model.apply(
                {"params": nn_params},
                img_flat_bt,
                method=nn_model.encode_vae,
                **encode_kwargs,
            )
            if training is True:
                # reparameterize
                z_static_pred_flat_bt = nn_model.reparameterize(
                    rng, mu_static_bt, logvar_static_bt
                )
            else:
                z_static_pred_flat_bt = mu_static_bt
        else:
            z_static_pred_flat_bt = nn_model.apply(
                {"params": nn_params}, img_flat_bt, method=encode_fn, **encode_kwargs
            )

        # reshape to batch_dim x time_dim x n_z
        z_static_pred_bt = z_static_pred_flat_bt.reshape(
            (batch_size, -1, *z_static_pred_flat_bt.shape[1:])
        )

        def estimate_initial_latent_velocity() -> Array:
            match latent_velocity_source:
                case "latent-space-finite-differences":
                    z_static_pred_4_dfd_bt = z_static_pred_bt
                    # apply finite differences to the static latent representation to get the static latent velocity
                    z_d_static_fd_bt = vmap(
                        lambda _z_ts: jnp.gradient(_z_ts, sample_dt, axis=0),
                        in_axes=(0,),
                        out_axes=0,
                    )(z_static_pred_4_dfd_bt)

                    # initial latent velocity as estimated by finite differences
                    z_d_init_bt = z_d_static_fd_bt[:, start_time_idx, ...]
                case "image-space-finite-differences":
                    # apply finite differences to the image space to get the image velocity
                    img_d_fd_bt = vmap(
                        lambda _img_ts: jnp.gradient(_img_ts, sample_dt, axis=0),
                        in_axes=(0,),
                        out_axes=0,
                    )(img_bt).astype(img_bt.dtype)

                    def encode_img_bt_to_latent_bt(_img_bt) -> Array:
                        _z = nn_model.apply(
                            {"params": nn_params},
                            _img_bt,
                            method=encode_fn,
                            **encode_kwargs,
                        )

                        return _z

                    # choose where we want to compute the latent velocity
                    img_init_fd_bt = img_bt[:, start_time_idx, ...]
                    img_d_init_fd_bt = img_d_fd_bt[:, start_time_idx, ...]

                    # computing the jacobian-vector product is more efficient
                    # than first computing the jacobian and then performing a matrix multiplication
                    _, z_d_init_bt = jvp(
                        encode_img_bt_to_latent_bt,
                        (img_init_fd_bt,),
                        (img_d_init_fd_bt,),
                    )
                case _:
                    raise ValueError(
                        f"latent_velocity_source must be either 'direct-finite-differences' "
                        f"or 'image-space-finite-differences', but is {latent_velocity_source}"
                    )

            return z_d_init_bt

        if dynamics_type == "node":
            # initial latent position as provided by the encoder
            z_init_bt = z_static_pred_bt[:, start_time_idx, ...]
            match dynamics_order:
                case 1:
                    x_init_bt = z_init_bt
                case 2:
                    # latent velocity at initial time provided by the encoder
                    z_d_init_bt = estimate_initial_latent_velocity()
                    # specify initial state for the dynamic rollout
                    x_init_bt = jnp.concatenate((z_init_bt, z_d_init_bt), axis=-1)
                case _:
                    raise ValueError(f"Dynamics order {dynamics_order} not supported.")

            # construct ode_fn and initiate ODE term
            def ode_fn(t: Array, x: Array, tau: Array) -> Array:
                x_d = nn_model.apply(
                    {"params": nn_params},
                    x,
                    tau,
                    method=nn_model.forward_dynamics,
                )
                return x_d

            ode_term = ODETerm(ode_fn)

            def ode_solve_fn(x0: Array, tau: Array):
                """
                Uses diffrax.diffeqsolve for solving the ode
                Arguments:
                    x0: initial state of shape (2*n_z, )
                    tau: external torques of shape (n_tau, )
                Returns:
                    sol: solution of the ode as a diffrax class
                """
                return diffeqsolve(
                    ode_term,
                    solver=solver,
                    t0=t0,  # initial time
                    t1=tf,  # final time
                    dt0=sim_dt,  # time step of integration
                    y0=x0,
                    args=tau,
                    saveat=SaveAt(ts=ts[start_time_idx:]),
                    max_steps=max_int_steps,
                )

            batched_ode_solve_fn = vmap(ode_solve_fn, in_axes=(0, 0))

            # simulate
            sol_bt = batched_ode_solve_fn(
                x_init_bt.astype(jnp.float64),  # initial state
                batch["tau"],  # external torques
            )

            # extract the rolled-out latent representations
            z_dynamic_pred_bt = sol_bt.ys[..., :n_z].astype(jnp.float32)
            z_d_dynamic_pred_bt = sol_bt.ys[..., n_z:].astype(jnp.float32)

            # extract the rolled-out states
            x_dynamic_pred_bt = sol_bt.ys.astype(jnp.float32)
        elif dynamics_type == "discrete":
            # construct the input for the discrete forward dynamics
            z_past_bt = z_static_pred_bt[:, start_time_idx - num_past_timesteps + 1 :]
            # construct batch of external torques of shape batch_dim x time_dim x n_tau
            tau_bt_discrete = jnp.expand_dims(batch["tau"], axis=1).repeat(
                ts.shape[0], axis=1
            )

            def autoregress_fn(_z_past_ts: Array, _tau_ts: Array) -> Array:
                """
                Autoregressive function for the discrete forward dynamics
                Arguments:
                    _z_past_ts: past latent representations of shape (num_past_timesteps, n_z)
                    _tau_ts: past & current external torques of shape (num_past_timesteps, n_tau)
                Returns:
                    _z_next: next latent representation of shape (batch_size, n_z)
                """

                _z_next = nn_model.apply(
                    {"params": nn_params},
                    _z_past_ts,
                    _tau_ts,
                    method=nn_model.forward_dynamics,
                )
                return _z_next

            def scan_fn(
                _carry: Dict[str, Array], _tau: Array
            ) -> Tuple[Dict[str, Array], Array]:
                """
                Function used as part of lax.scan rollout for the discrete forward dynamics
                Arguments:
                    _carry: carry state of the scan function
                    _tau: external torques of shape (n_tau, )
                """
                # append the current external torque to the past external torques
                _tau_ts = jnp.concatenate((_carry["tau_ts"], _tau[None, ...]), axis=0)

                if interpret_discrete_hidden_state_as_displacement:
                    _hidden_state = jnp.concatenate(
                        [
                            jnp.diff(
                                _carry["z_past_ts"], axis=0
                            ),  # this is the relative displacement of the previous latents from the current latent
                            _carry["z_past_ts"][
                                -2:-1, ...
                            ],  # this is the current latent
                        ],
                        axis=0,
                    ).flatten()
                else:
                    _hidden_state = _carry["z_past_ts"].flatten()

                # evaluate the autoregressive function
                _z_next = autoregress_fn(_hidden_state, _tau_ts.flatten())

                # update the carry state
                _carry = dict(
                    z_past_ts=jnp.concatenate(
                        (_carry["z_past_ts"][1:], _z_next[None, ...]), axis=0
                    ),
                    tau_ts=_tau_ts[1:],
                )

                return _carry, _z_next

            def rollout_discrete_dynamics_fn(_z_ts: Array, _tau_ts: Array) -> Array:
                """
                Rollout function for the discrete forward dynamics
                Arguments:
                    _z_ts: latent representations of shape (time_dim, n_z)
                    _tau_ts: external torques of shape (time_dim, n_tau)
                Returns:
                    _z_dynamic_ts: next latent representation of shape (time_dim - start_time_idx, n_z)
                """
                carry_init = dict(
                    z_past_ts=_z_ts[
                        start_time_idx - num_past_timesteps + 1 : start_time_idx + 1
                    ],  # shape (num_past_timesteps, n_z)
                    tau_ts=_tau_ts[
                        start_time_idx - num_past_timesteps + 1 : start_time_idx
                    ],  # shape (num_past_timesteps - 1, n_tau)
                )
                carry_final, _z_dynamic_ts = lax.scan(
                    scan_fn,
                    init=carry_init,
                    xs=_tau_ts[start_time_idx:],
                )

                return _z_dynamic_ts

            z_dynamic_pred_bt = vmap(
                rollout_discrete_dynamics_fn,
                in_axes=(0, 0),
                out_axes=0,
            )(z_past_bt, tau_bt_discrete.astype(jnp.float32))
            z_d_dynamic_pred_bt = jnp.zeros_like(z_dynamic_pred_bt)
        elif dynamics_type == "ar":
            # autoregresses the latent space positions and velocities

            # initial latent position as provided by the encoder
            z_init_bt = z_static_pred_bt[:, start_time_idx, ...]
            match dynamics_order:
                case 1:
                    x_init_bt = z_init_bt
                case 2:
                    # latent velocity at initial time provided by the encoder
                    z_d_init_bt = estimate_initial_latent_velocity()
                    # specify initial state for the dynamic rollout
                    x_init_bt = jnp.concatenate((z_init_bt, z_d_init_bt), axis=-1)
                case _:
                    raise ValueError(f"Dynamics order {dynamics_order} not supported.")

            # construct batch of external torques of shape batch_dim x time_dim x n_tau
            tau_bt_sim = jnp.expand_dims(batch["tau"], axis=1).repeat(
                ts_sim.shape[0], axis=1
            )

            def autoregress_fn(_x: Array, _tau: Array) -> Array:
                """
                Autoregressive function for the discrete forward dynamics
                Arguments:
                    _x: latent state of shape (2*n_z, )
                    _tau: external torques of shape (n_tau, )
                Returns:
                    _x: next hidden state of shape (2*n_z, )
                """

                _x_next = nn_model.apply(
                    {"params": nn_params},
                    _x,
                    _tau,
                    method=nn_model.forward_dynamics,
                )
                return _x_next

            def scan_fn(
                _carry: Dict[str, Array], _tau: Array
            ) -> Tuple[Dict[str, Array], Array]:
                """
                Function used as part of lax.scan rollout for the discrete forward dynamics
                Arguments:
                    _carry: carry state of the scan function
                    _tau: external torques of shape (n_tau, )
                """
                _x = _carry["x"]

                # evaluate the autoregressive function
                _x_next = autoregress_fn(_x, _tau.flatten())

                # update the carry state
                _carry = dict(x=_x_next)

                return _carry, _x_next

            def rollout_discrete_dynamics_fn(_x0: Array, _tau_ts: Array) -> Array:
                """
                Rollout function for the discrete forward dynamics
                Arguments:
                    _x0: initial latent state of shape (2*n_z, )
                    _tau_ts: external torques of shape (time_dim, n_tau)
                Returns:
                    _x_dynamic_ts: next latent representation of shape (time_dim, n_z)
                """
                carry_init = dict(x=_x0)
                carry_final, _x_dynamic_ts = lax.scan(
                    scan_fn,
                    init=carry_init,
                    xs=_tau_ts,
                )

                return _x_dynamic_ts

            x_dynamic_pred_sim_bt = vmap(
                rollout_discrete_dynamics_fn,
                in_axes=(0, 0),
                out_axes=0,
            )(x_init_bt.astype(jnp.float64), tau_bt_sim.astype(jnp.float64))

            # extract only the sampling steps (i.e., remove the remaining simulation steps)
            x_dynamic_pred_bt = x_dynamic_pred_sim_bt[
                :, ::sample_sim_skip_step, ...
            ].astype(jnp.float32)

            z_dynamic_pred_bt = x_dynamic_pred_bt[..., :n_z].astype(jnp.float32)
            if dynamics_order == 2:
                z_d_dynamic_pred_bt = x_dynamic_pred_bt[..., n_z:].astype(jnp.float32)
            else:
                z_d_dynamic_pred_bt = None
        else:
            raise ValueError(f"Unknown dynamics_type: {dynamics_type}")

        # flatten the dynamic latent predictions
        z_dynamic_pred_flat_bt = z_dynamic_pred_bt.reshape(
            (-1, *z_dynamic_pred_bt.shape[2:])
        )

        # send the rolled-out latent representations through the decoder
        # output will be of shape batch_dim * time_dim x width x height x channels
        img_static_pred_flat_bt = nn_model.apply(
            {"params": nn_params},
            z_static_pred_flat_bt,
            method=decode_fn,
            **decode_kwargs,
        )
        img_dynamic_pred_flat_bt = nn_model.apply(
            {"params": nn_params},
            z_dynamic_pred_flat_bt,
            method=decode_fn,
            **decode_kwargs,
        )

        # reshape to batch_dim x time_dim x width x height x channels
        img_static_pred_bt = img_static_pred_flat_bt.reshape(
            (batch_size, -1, *img_static_pred_flat_bt.shape[1:])
        )
        img_dynamic_pred_bt = img_dynamic_pred_flat_bt.reshape(
            (batch_size, -1, *img_dynamic_pred_flat_bt.shape[1:])
        )

        preds = dict(
            z_static_ts=z_static_pred_bt,
            img_static_ts=img_static_pred_bt,
            z_dynamic_ts=z_dynamic_pred_bt,
            img_dynamic_ts=img_dynamic_pred_bt,
        )
        if dynamics_order == 2:
            preds["xi_dynamic_ts"] = jnp.concatenate(
                (z_dynamic_pred_bt, z_d_dynamic_pred_bt), axis=-1
            ),
        else:
            preds["xi_dynamic_ts"] = z_dynamic_pred_bt

        if ae_type == "beta_vae":
            # reshape to batch_dim x time_dim x n_z
            preds["mu_static_ts"] = mu_static_bt.reshape(
                (batch_size, -1, *mu_static_bt.shape[1:])
            )
            preds["logvar_static_ts"] = logvar_static_bt.reshape(
                (batch_size, -1, *logvar_static_bt.shape[1:])
            )

        if dynamics_type != "discrete":
            preds["x_dynamic_ts"] = x_dynamic_pred_bt

        if type(nn_model.dynamics) in [ConIaeOde, DiscreteConIaeCfaDynamics]:
            # autoencoder the torque

            def autoencode_input(tau: Array) -> Array:
                return nn_model.dynamics.apply(
                    {"params": nn_params["dynamics"]},
                    tau,
                    method=nn_model.dynamics.autoencode_input,
                )

            tau_bt = batch["tau"]
            preds["tau_pred"] = vmap(autoencode_input)(tau_bt)

        return preds

    def loss_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[Array] = None,
        training: bool = False,
    ) -> Tuple[Array, Dict[str, Array]]:
        preds = forward_fn(batch, nn_params, rng=rng, training=training)

        z_static_pred_bt = preds["z_static_ts"]
        z_dynamic_pred_bt = preds["z_dynamic_ts"]

        # compute the latent space error
        error_z = z_dynamic_pred_bt - z_static_pred_bt[:, start_time_idx:]
        # compute the mean squared error
        mse_z = jnp.mean(jnp.square(error_z))

        # supervised MSE loss on the reconstructed image of the static predictions
        mse_rec_static = jnp.mean(
            jnp.square(preds["img_static_ts"] - batch["rendering_ts"])
        )
        # supervised MSE loss on the reconstructed image of the dynamic predictions
        mse_rec_dynamic = jnp.mean(
            jnp.square(
                preds["img_dynamic_ts"] - batch["rendering_ts"][:, start_time_idx:]
            )
        )

        # total loss
        loss = (
            loss_weights["mse_z"] * mse_z
            + loss_weights["mse_rec_static"] * mse_rec_static
            + loss_weights["mse_rec_dynamic"] * mse_rec_dynamic
        )

        if ae_type == "wae":
            latent_dim = preds["z_static_ts"].shape[-1]

            (img_target_bt,) = assemble_input(batch)
            img_pred_bt = preds["img_static_ts"].reshape(
                (-1, *preds["img_static_ts"].shape[2:])
            )
            z_pred_bt = preds["z_static_ts"].reshape((-1, latent_dim))

            # Wasserstein Autoencoder MMD loss
            mmd_loss = wae_mmd_loss_fn(
                x_rec=img_pred_bt, x_target=img_target_bt, z=z_pred_bt, rng=rng
            )

            loss = loss + loss_weights["mmd"] * mmd_loss
        elif ae_type == "beta_vae":
            # KLD loss
            # https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/beta_vae/beta_vae_model.py#L101
            kld_loss = kullback_leiber_divergence(
                preds["mu_static_ts"], preds["logvar_static_ts"]
            )

            loss = loss + loss_weights["beta"] * kld_loss

        if type(nn_model.dynamics) in [ConIaeOde, DiscreteConIaeCfaDynamics]:
            # autoencoder the torque
            mse_tau_rec = jnp.mean(jnp.square(preds["tau_pred"] - batch["tau"]))
            loss = loss + loss_weights.get("mse_tau_rec", 0.0) * mse_tau_rec

        return loss, preds

    def compute_metrics_fn(
        batch: Dict[str, Array], preds: Dict[str, Array]
    ) -> Dict[str, Array]:
        batch_loss_dict = {
            "mse_rec_static": jnp.mean(
                jnp.square(preds["img_static_ts"] - batch["rendering_ts"])
            ),
            "mse_rec_dynamic": jnp.mean(
                jnp.square(
                    preds["img_dynamic_ts"] - batch["rendering_ts"][:, start_time_idx:]
                )
            ),
        }

        if compute_psnr:
            batch_loss_dict["psnr_rec_static"] = peak_signal_to_noise_ratio(
                preds["img_static_ts"],
                batch["rendering_ts"],
                data_min=-1.0,
                data_max=1.0,
            )
            batch_loss_dict["psnr_rec_dynamic"] = peak_signal_to_noise_ratio(
                preds["img_dynamic_ts"],
                batch["rendering_ts"][:, start_time_idx:],
                data_min=-1.0,
                data_max=1.0,
            )

        if compute_ssim:
            batch_loss_dict["ssim_rec_static"] = structural_similarity_index(
                preds["img_static_ts"].reshape(-1, *preds["img_static_ts"].shape[2:]),
                batch["rendering_ts"].reshape(-1, *batch["rendering_ts"].shape[2:]),
            )
            batch_loss_dict["ssim_rec_dynamic"] = structural_similarity_index(
                preds["img_dynamic_ts"].reshape(-1, *preds["img_dynamic_ts"].shape[2:]),
                batch["rendering_ts"][:, start_time_idx:].reshape(
                    -1, *batch["rendering_ts"].shape[2:]
                ),
            )

        if type(nn_model.dynamics) in [ConIaeOde, DiscreteConIaeCfaDynamics]:
            batch_loss_dict["mse_tau_rec"] = jnp.mean(
                jnp.square(preds["tau_pred"] - batch["tau"])
            )

        return batch_loss_dict

    task_callables = TaskCallables(
        system_type, assemble_input, forward_fn, loss_fn, compute_metrics_fn
    )

    @dataclass  # <-- required for JAX transformations
    class MetricsCollection(clu_metrics.Collection):
        loss: clu_metrics.Average.from_output("loss")
        lr: clu_metrics.LastValue.from_output("lr")
        rmse_rec_static: RootAverage.from_output("mse_rec_static")
        rmse_rec_dynamic: RootAverage.from_output("mse_rec_dynamic")

        if compute_psnr:
            psnr_rec_static: RootAverage.from_output("psnr_rec_static")
            psnr_rec_dynamic: RootAverage.from_output("psnr_rec_dynamic")

        if compute_ssim:
            ssim_rec_static: RootAverage.from_output("ssim_rec_static")
            ssim_rec_dynamic: RootAverage.from_output("ssim_rec_dynamic")

        if type(nn_model.dynamics) in [ConIaeOde, DiscreteConIaeCfaDynamics]:
            rmse_tau_rec: RootAverage.from_output("mse_tau_rec")

    metrics_collection_cls = MetricsCollection
    return task_callables, metrics_collection_cls
