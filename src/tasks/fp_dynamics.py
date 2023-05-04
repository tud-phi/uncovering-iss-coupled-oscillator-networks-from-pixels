from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from flax.core import FrozenDict
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, debug, jacrev, jit, random, vmap
import jax.numpy as jnp
import jax_metrics as jm
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Callable, Dict, Optional, Tuple

from src.metrics import NoReduce
from src.structs import TaskCallables


@jit
def assemble_input(batch) -> Array:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return img_bt


def task_factory(
        system_type: str,
        nn_model: nn.Module,
        ode_fn: Callable,
        loss_weights: Optional[Dict[str, float]] = None,
        solver: AbstractSolver = Dopri5(),
        start_time_idx: int = 1,
        configuration_velocity_source: str = "direct-finite-differences",
) -> Tuple[TaskCallables, jm.Metrics]:
    """
    Factory function for the task of learning a representation using first-principle dynamics while using the
    ground-truth velocity.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        nn_model: the neural network model to use
        ode_fn: ODE function. It should have the following signature:
            ode_fn(t, x) -> x_dot
        loss_weights: the weights for the different loss terms
        solver: Diffrax solver to use for the simulation.
        start_time_idx: the index of the time step to start the simulation at. Needs to be >=1 to enable the application
            of finite differences for the latent-space velocity.
        configuration_velocity_source: the source of the configuration velocity.
            Can be either "direct-finite-differences", or "image-space-finite-differences"
    Returns:
        task_callables: struct containing the functions for the learning task
        metrics: struct containing the metrics for the learning task
    """
    if loss_weights is None:
        loss_weights = dict(mse_q=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0)

    # initiate ODE term from `ode_fn`
    ode_term = ODETerm(ode_fn)

    @jit
    def forward_fn(batch: Dict[str, Array], nn_params: FrozenDict) -> Dict[str, Array]:
        img_bt = batch["rendering_ts"]
        img_flat_bt = assemble_input(batch)
        t_ts = batch["t_ts"][
            0
        ]  # we just assume that the time steps are the same for all batch items
        dt = (t_ts[1:] - t_ts[:-1]).mean()

        batch_size = batch["rendering_ts"].shape[0]
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        # static predictions by passing the image through the encoder
        # output will be of shape batch_dim * time_dim x latent_dim
        # if the system is a pendulum, the latent dim should be 2*n_q
        encoder_output = nn_model.apply(
            {"params": nn_params}, img_flat_bt, method=nn_model.encode
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
                # apply finite differences to the static latent representation to get the static latent velocity
                q_d_static_fd_bt = vmap(
                    lambda _q_ts: jnp.gradient(_q_ts, dt, axis=0), in_axes=(0,), out_axes=0
                )(q_static_pred_bt)

                # initial configuration velocity as estimated by finite differences
                q_d_init_bt = q_d_static_fd_bt[:, start_time_idx, ...]
            case "image-space-finite-differences":
                # apply finite differences to the image space to get the image velocity
                img_d_fd_bt = vmap(
                    lambda _img_ts: jnp.gradient(_img_ts, dt, axis=0), in_axes=(0,), out_axes=0
                )(img_bt)

                def encode_img_to_configuration(_img) -> Array:
                    _encoder_output = nn_model.apply(
                        {"params": nn_params}, jnp.expand_dims(_img, axis=0), method=nn_model.encode
                    ).squeeze(axis=0)

                    if system_type == "pendulum":
                        # if the system is a pendulum, we interpret the encoder output as sin(theta) and cos(theta) for each joint
                        # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
                        # output of arctan2 will be in the range [-pi, pi]
                        _q = jnp.arctan2(_encoder_output[..., :n_q], _encoder_output[..., n_q:])
                    else:
                        _q = _encoder_output

                    return _q

                # choose where we want to compute the configuration velocity
                img_init_fd_bt = img_bt[:, start_time_idx, ...]
                img_d_init_fd_bt = img_d_fd_bt[:, start_time_idx, ...]

                dq_dimg_init_bt = vmap(
                    lambda _img: jacrev(encode_img_to_configuration)(_img),
                    in_axes=(0,),
                    out_axes=0,
                )(img_init_fd_bt)

                # flatten so we can do matrix multiplication
                # establish shape (batch_dim, n_q, -1)
                dq_dimg_init_bt_flat = dq_dimg_init_bt.reshape((*dq_dimg_init_bt.shape[0:2], -1))
                # establish shape (batch_dim, -1)
                img_d_init_fd_bt_flat = img_d_init_fd_bt.reshape((img_d_init_fd_bt.shape[0], -1))

                # apply the chain rule to compute the velocity in latent space
                q_d_init_hat_bt_flat = vmap(
                    jnp.matmul,
                    in_axes=0,
                    out_axes=0
                )(dq_dimg_init_bt_flat, img_d_init_fd_bt_flat)

                # reshape the result to (batch_dim, n_q)
                q_d_init_bt = q_d_init_hat_bt_flat.reshape(q_init_bt.shape)
            case _:
                raise ValueError(f"configuration_velocity_source must be either 'direct-finite-differences' "
                                 f"or 'image-space-finite-differences', but is {configuration_velocity_source}")

        # specify initial state for the dynamic rollout
        x_init_bt = jnp.concatenate((q_init_bt, q_d_init_bt), axis=-1)

        # compute the dynamic rollout of the latent representation
        ode_solve_fn = partial(
            diffeqsolve,
            ode_term,
            solver,
            saveat=SaveAt(ts=t_ts[start_time_idx:]),
            max_steps=t_ts[start_time_idx:].shape[-1],
        )
        # simulate
        sol_bt = vmap(ode_solve_fn, in_axes=(None, None, None, 0))(
            t_ts[start_time_idx],  # initial time
            t_ts[-1],  # final time
            dt,  # time step
            x_init_bt.astype(jnp.float64),  # initial state
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
            {"params": nn_params}, decoder_static_input, method=nn_model.decode
        )
        img_dynamic_pred_flat_bt = nn_model.apply(
            {"params": nn_params}, decoder_dynamic_input, method=nn_model.decode
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
            rendering_static_ts=img_static_pred_bt,
            q_dynamic_ts=q_dynamic_pred_bt,
            rendering_dynamic_ts=img_dynamic_pred_bt,
        )

        return preds

    @jit
    def loss_fn(
            batch: Dict[str, Array],
            nn_params: FrozenDict,
            rng: Optional[random.PRNGKey] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        preds = forward_fn(batch, nn_params)

        q_static_pred_bt = preds["q_static_ts"]
        q_dynamic_pred_bt = preds["q_dynamic_ts"]

        # compute the configuration error
        error_q = q_dynamic_pred_bt - q_static_pred_bt[:, start_time_idx:]

        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)

        # compute the mean squared error
        mse_q = jnp.mean(jnp.square(error_q))

        # supervised MSE loss on the reconstructed image of the static predictions
        mse_rec_static = jnp.mean(
            jnp.square(preds["rendering_static_ts"] - batch["rendering_ts"])
        )
        # supervised MSE loss on the reconstructed image of the dynamic predictions
        mse_rec_dynamic = jnp.mean(
            jnp.square(
                preds["rendering_dynamic_ts"]
                - batch["rendering_ts"][:, start_time_idx:]
            )
        )

        # total loss
        loss = (
                loss_weights["mse_q"] * mse_q
                + loss_weights["mse_rec_static"] * mse_rec_static
                + loss_weights["mse_rec_dynamic"] * mse_rec_dynamic
        )

        return loss, preds

    @jit
    def compute_metrics(
            batch: Dict[str, Array], preds: Dict[str, Array]
    ) -> Dict[str, Array]:
        q_static_pred_bt = preds["q_static_ts"]
        q_dynamic_pred_bt = preds["q_dynamic_ts"]
        q_target_bt = batch["x_ts"][..., : batch["x_ts"].shape[-1] // 2]

        # compute the configuration error
        error_q_static = q_static_pred_bt - q_target_bt
        error_q_dynamic = q_dynamic_pred_bt - q_target_bt[:, start_time_idx:]

        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q_static = normalize_joint_angles(error_q_static)
            error_q_dynamic = normalize_joint_angles(error_q_dynamic)

        metrics = {
            "rmse_q_static": jnp.sqrt(jnp.mean(jnp.square(error_q_static))),
            "rmse_rec_static": jnp.sqrt(
                jnp.mean(
                    jnp.square(preds["rendering_static_ts"] - batch["rendering_ts"])
                )
            ),
            "rmse_q_dynamic": jnp.sqrt(jnp.mean(jnp.square(error_q_dynamic))),
            "rmse_rec_dynamic": jnp.sqrt(
                jnp.mean(
                    jnp.square(
                        preds["rendering_dynamic_ts"]
                        - batch["rendering_ts"][:, start_time_idx:]
                    )
                )
            ),
        }
        return metrics

    task_callables = TaskCallables(assemble_input, forward_fn, loss_fn, compute_metrics)

    metrics = jm.Metrics(
        {
            "loss": jm.metrics.Mean().from_argument("loss"),
            "lr": NoReduce().from_argument("lr"),
            "rmse_q_static": jm.metrics.Mean().from_argument("rmse_q_static"),
            "rmse_rec_static": jm.metrics.Mean().from_argument("rmse_rec_static"),
            "rmse_q_dynamic": jm.metrics.Mean().from_argument("rmse_q_dynamic"),
            "rmse_rec_dynamic": jm.metrics.Mean().from_argument("rmse_rec_dynamic"),
        }
    )

    return task_callables, metrics
