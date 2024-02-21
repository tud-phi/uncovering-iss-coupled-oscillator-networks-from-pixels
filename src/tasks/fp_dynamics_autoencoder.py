from clu import metrics as clu_metrics
from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from flax.core import FrozenDict
from flax.struct import dataclass
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, debug, jit, jvp, random, vmap
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Any, Callable, Dict, Optional, Tuple, Type

from src.metrics import RootAverage
from src.losses.kld import kullback_leiber_divergence
from src.structs import TaskCallables


def assemble_input(batch) -> Tuple[Array]:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return (img_bt,)


def task_factory(
    system_type: str,
    nn_model: nn.Module,
    ode_fn: Callable,
    ts: Array,
    sim_dt: float,
    encode_fn: Optional[Callable] = None,
    decode_fn: Optional[Callable] = None,
    encode_kwargs: Optional[Dict[str, Any]] = None,
    decode_kwargs: Optional[Dict[str, Any]] = None,
    x0_min: Optional[Array] = None,
    x0_max: Optional[Array] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    ae_type: str = "None",
    normalize_configuration_loss=False,
    solver: AbstractSolver = Dopri5(),
    start_time_idx: int = 1,
    configuration_velocity_source: str = "direct-finite-differences",
) -> Tuple[TaskCallables, Type[clu_metrics.Collection]]:
    """
    Factory function for the task of learning a representation using first-principle dynamics while using the
    ground-truth velocity.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics_fn functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        nn_model: the neural network model to use
        ode_fn: ODE function. It should have the following signature:
            ode_fn(t, x, tau) -> x_dot
        ts: time steps of the samples
        sim_dt: Time step used for simulation [s].
        encode_fn: the function to use for encoding the input image to the latent space
        decode_fn: the function to use for decoding the latent space to the output image
        encode_kwargs: additional kwargs to pass to the encode_fn
        decode_kwargs: additional kwargs to pass to the decode_fn
        x0_min: the minimal value for the initial state of the simulation
        x0_max: the maximal value for the initial state of the simulation
        loss_weights: the weights for the different loss terms
        ae_type: Autoencoder type. If None, a normal autoencoder will be used.
            One of ["wae", "beta_vae", "None"]
        normalize_configuration_loss: whether to normalize the configuration loss dividing by (q0_max - q0_min)
        solver: Diffrax solver to use for the simulation.
        start_time_idx: the index of the time step to start the simulation at. Needs to be >=1 to enable the application
            of finite differences for the latent-space velocity.
        configuration_velocity_source: the source of the configuration velocity.
            Can be either "direct-finite-differences", "image-space-finite-differences", or "ground-truth".
    Returns:
        task_callables: struct containing the functions for the learning task
        metrics_collection_cls: contains class for collecting metrics
    """
    # time step between samples
    sample_dt = (ts[1:] - ts[:-1]).mean()
    # compute the dynamic rollout of the latent representation
    t0 = ts[start_time_idx]  # start time
    tf = ts[-1]  # end time
    # maximum of integrator steps
    max_int_steps = int((tf - t0) / sim_dt) + 1

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
        dict(mse_q=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0) | loss_weights
    )

    if ae_type == "wae":
        loss_weights = dict(mmd=1.0) | loss_weights

        assert system_type == "pendulum", "WAE only implemented for pendulum system"
        if system_type == "pendulum":
            uniform_distr_range = (-jnp.pi, jnp.pi)
        else:
            uniform_distr_range = (-1.0, 1.0)

        from src.losses import wae

        wae_mmd_loss_fn = wae.make_wae_mdd_loss(
            distribution="uniform", uniform_distr_range=uniform_distr_range
        )
    elif ae_type == "beta_vae":
        loss_weights = dict(beta=1.0) | loss_weights

    if normalize_configuration_loss is True:
        assert (
            x0_min is not None and x0_max is not None
        ), "x0_min and x0_max must be provided for normalizing the configuration loss"

    # initiate ODE term from `ode_fn`
    ode_term = ODETerm(ode_fn)

    def ode_solve_fn(x0: Array, tau: Array):
        """
        Uses diffrax.diffeqsolve for solving the ode
        Arguments:
            x0: initial state of shape (2*n_q, )
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

    batched_ode_solve_fn = jit(vmap(ode_solve_fn, in_axes=(0, 0)))

    def forward_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[Array] = None,
        training: bool = False,
    ) -> Dict[str, Array]:
        img_bt = batch["rendering_ts"]
        (img_flat_bt,) = assemble_input(batch)

        batch_size = batch["rendering_ts"].shape[0]
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        # static predictions by passing the image through the encoder
        # output will be of shape batch_dim * time_dim x latent_dim
        # if the system is a pendulum, the latent dim should be 2*n_q
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
                encoder_output = nn_model.reparameterize(
                    rng, mu_static_bt, logvar_static_bt
                )
            else:
                encoder_output = mu_static_bt
        else:
            encoder_output = nn_model.apply(
                {"params": nn_params}, img_flat_bt, method=encode_fn, **encode_kwargs
            )

        if system_type == "pendulum":
            # if the system is a pendulum, we interpret the encoder output as sin(theta) and cos(theta) for each joint
            # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
            # output of arctan2 will be in the range [-pi, pi]
            q_static_pred_flat_bt = jnp.arctan2(
                encoder_output[..., :n_q], encoder_output[..., n_q:]
            )
        else:
            q_static_pred_flat_bt = encoder_output

        # reshape to batch_dim x time_dim x n_q
        q_static_pred_bt = q_static_pred_flat_bt.reshape(
            (batch_size, -1, *q_static_pred_flat_bt.shape[1:])
        )

        # initial configuration at initial time provided by the encoder
        q_init_bt = q_static_pred_bt[:, start_time_idx, ...]

        match configuration_velocity_source:
            case "direct-finite-differences":
                q_static_pred_4_dfd_bt = q_static_pred_bt
                if system_type == "pendulum":
                    # for penduli, we might have a transition issue at the boundary of the [-pi, pi] range
                    # so we unwrap the angles
                    q_static_pred_4_dfd_bt = jnp.unwrap(q_static_pred_4_dfd_bt, axis=1)

                # apply finite differences to the static latent representation to get the static latent velocity
                q_d_static_fd_bt = vmap(
                    lambda _q_ts: jnp.gradient(_q_ts, sample_dt, axis=0),
                    in_axes=(0,),
                    out_axes=0,
                )(q_static_pred_4_dfd_bt)

                # initial configuration velocity as estimated by finite differences
                q_d_init_bt = q_d_static_fd_bt[:, start_time_idx, ...]
            case "image-space-finite-differences":
                # apply finite differences to the image space to get the image velocity
                img_d_fd_bt = vmap(
                    lambda _img_ts: jnp.gradient(_img_ts, sample_dt, axis=0),
                    in_axes=(0,),
                    out_axes=0,
                )(img_bt).astype(img_bt.dtype)

                def encode_img_bt_to_configuration_bt(_img_bt) -> Array:
                    _encoder_output = nn_model.apply(
                        {"params": nn_params},
                        _img_bt,
                        method=encode_fn,
                        **encode_kwargs,
                    )

                    if system_type == "pendulum":
                        # if the system is a pendulum, we interpret the encoder output as sin(theta) and cos(theta) for each joint
                        # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
                        # output of arctan2 will be in the range [-pi, pi]
                        _q = jnp.arctan2(
                            _encoder_output[..., :n_q], _encoder_output[..., n_q:]
                        )
                    else:
                        _q = _encoder_output

                    return _q

                # choose where we want to compute the configuration velocity
                img_init_fd_bt = img_bt[:, start_time_idx, ...]
                img_d_init_fd_bt = img_d_fd_bt[:, start_time_idx, ...]

                # computing the jacobian-vector product is more efficient
                # than first computing the jacobian and then performing a matrix multiplication
                _, q_d_init_bt = jvp(
                    encode_img_bt_to_configuration_bt,
                    (img_init_fd_bt,),
                    (img_d_init_fd_bt,),
                )
            case "ground-truth":
                # use the ground-truth velocity
                q_d_init_bt = batch["x_ts"][:, start_time_idx, n_q:]
            case _:
                raise ValueError(
                    f"configuration_velocity_source must be either 'direct-finite-differences' "
                    f"or 'image-space-finite-differences', but is {configuration_velocity_source}"
                )

        # specify initial state for the dynamic rollout
        x_init_bt = jnp.concatenate((q_init_bt, q_d_init_bt), axis=-1)

        # simulate
        sol_bt = batched_ode_solve_fn(
            x_init_bt.astype(jnp.float64),  # initial state
            batch["tau"],  # external torques
        )

        # extract the rolled-out latent representations
        q_dynamic_pred_bt = sol_bt.ys[..., :n_q].astype(jnp.float32)

        # flatten the dynamic configuration predictions
        q_dynamic_pred_flat_bt = q_dynamic_pred_bt.reshape(
            (-1, *q_dynamic_pred_bt.shape[2:])
        )

        if system_type == "pendulum":
            # if the system is a pendulum, the input into the decoder should be sin(theta) and cos(theta) for each joint
            # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
            decoder_static_input = jnp.concatenate(
                [jnp.sin(q_static_pred_flat_bt), jnp.cos(q_static_pred_flat_bt)],
                axis=-1,
            )
            decoder_dynamic_input = jnp.concatenate(
                [
                    jnp.sin(q_dynamic_pred_flat_bt),
                    jnp.cos(q_dynamic_pred_flat_bt),
                ],
                axis=-1,
            )
        else:
            decoder_static_input = q_static_pred_flat_bt
            decoder_dynamic_input = q_dynamic_pred_flat_bt

        # send the rolled-out latent representations through the decoder
        # output will be of shape batch_dim * time_dim x width x height x channels
        img_static_pred_flat_bt = nn_model.apply(
            {"params": nn_params},
            decoder_static_input,
            method=decode_fn,
            **decode_kwargs,
        )
        img_dynamic_pred_flat_bt = nn_model.apply(
            {"params": nn_params},
            decoder_dynamic_input,
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
            q_static_ts=q_static_pred_bt,
            img_static_ts=img_static_pred_bt,
            q_dynamic_ts=q_dynamic_pred_bt,
            x_dynamic_ts=sol_bt.ys.astype(jnp.float32),  # the full state
            img_dynamic_ts=img_dynamic_pred_bt,
        )

        if ae_type == "beta_vae":
            # reshape to batch_dim x time_dim x n_q
            preds["mu_static_ts"] = mu_static_bt.reshape(
                (batch_size, -1, *mu_static_bt.shape[1:])
            )
            preds["logvar_static_ts"] = logvar_static_bt.reshape(
                (batch_size, -1, *logvar_static_bt.shape[1:])
            )

        return preds

    def loss_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[Array] = None,
        training: bool = False,
    ) -> Tuple[Array, Dict[str, Array]]:
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates
        preds = forward_fn(batch, nn_params, rng=rng, training=training)

        q_static_pred_bt = preds["q_static_ts"]
        q_dynamic_pred_bt = preds["q_dynamic_ts"]

        # compute the configuration error
        error_q = q_dynamic_pred_bt - q_static_pred_bt[:, start_time_idx:]
        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)
        # if requested, normalize the configuration loss by dividing by (q0_max - q0_min)
        if normalize_configuration_loss is True:
            error_q = error_q / (x0_max[:n_q] - x0_min[:n_q])

        # compute the mean squared error
        mse_q = jnp.mean(jnp.square(error_q))

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
            loss_weights["mse_q"] * mse_q
            + loss_weights["mse_rec_static"] * mse_rec_static
            + loss_weights["mse_rec_dynamic"] * mse_rec_dynamic
        )

        if ae_type == "wae":
            latent_dim = preds["q_static_ts"].shape[-1]

            (img_target_bt,) = assemble_input(batch)
            img_pred_bt = preds["img_static_ts"].reshape(
                (-1, *preds["img_static_ts"].shape[2:])
            )
            q_pred_bt = preds["q_static_ts"].reshape((-1, latent_dim))

            # Wasserstein Autoencoder MMD loss
            mmd_loss = wae_mmd_loss_fn(
                x_rec=img_pred_bt, x_target=img_target_bt, z=q_pred_bt, rng=rng
            )

            loss = loss + loss_weights["mmd"] * mmd_loss
        elif ae_type == "beta_vae":
            # KLD loss
            # https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/beta_vae/beta_vae_model.py#L101
            kld_loss = kullback_leiber_divergence(
                preds["mu_static_ts"], preds["logvar_static_ts"]
            )

            loss = loss + loss_weights["beta"] * kld_loss

        return loss, preds

    def compute_metrics_fn(
        batch: Dict[str, Array], preds: Dict[str, Array]
    ) -> Dict[str, Array]:
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        q_static_pred_bt = preds["q_static_ts"]
        q_dynamic_pred_bt = preds["q_dynamic_ts"]
        q_target_bt = batch["x_ts"][..., :n_q]

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

        # compute the configuration error
        error_q_static = q_static_pred_bt - q_target_bt
        error_q_dynamic = q_dynamic_pred_bt - q_target_bt[:, start_time_idx:]

        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q_static = normalize_joint_angles(error_q_static)
            error_q_dynamic = normalize_joint_angles(error_q_dynamic)

        # compute the configuration velocity error
        error_q_d_dynamic = (
            preds["x_dynamic_ts"][..., n_q:] - batch["x_ts"][:, start_time_idx:, n_q:]
        )

        if normalize_configuration_loss:
            error_q_static_norm = error_q_static / (x0_max[:n_q] - x0_min[:n_q])
            error_q_dynamic_norm = error_q_dynamic / (x0_max[:n_q] - x0_min[:n_q])
            error_q_d_dynamic_norm = error_q_d_dynamic / (x0_max[n_q:] - x0_min[n_q:])
            batch_loss_dict.update(
                {
                    "mse_q_static_norm": jnp.mean(jnp.square(error_q_static_norm)),
                    "mse_q_dynamic_norm": jnp.mean(jnp.square(error_q_dynamic_norm)),
                    "mse_q_d_dynamic_norm": jnp.mean(
                        jnp.square(error_q_d_dynamic_norm)
                    ),
                }
            )
        else:
            batch_loss_dict.update(
                {
                    "mse_q_static": jnp.mean(jnp.square(error_q_static)),
                    "mse_q_dynamic": jnp.mean(jnp.square(error_q_dynamic)),
                    "mse_q_d_dynamic": jnp.mean(jnp.square(error_q_d_dynamic)),
                }
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

        if normalize_configuration_loss is True:
            rmse_q_static_norm: RootAverage.from_output("mse_q_static_norm")
            rmse_q_dynamic_norm: RootAverage.from_output("mse_q_dynamic_norm")
            rmse_q_d_dynamic_norm: RootAverage.from_output("mse_q_d_dynamic_norm")
        else:
            rmse_q_static: RootAverage.from_output("mse_q_static")
            rmse_q_dynamic: RootAverage.from_output("mse_q_dynamic")
            rmse_q_d_dynamic: RootAverage.from_output("mse_q_d_dynamic")

    metrics_collection_cls = MetricsCollection
    return task_callables, metrics_collection_cls
