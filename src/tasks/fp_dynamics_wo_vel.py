from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from flax.core import FrozenDict
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, debug, jit, random, vmap
import jax.numpy as jnp
import jax_metrics as jm
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Any, Callable, Dict, Optional, Tuple

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
    encode_fn: Callable = None,
    decode_fn: Callable = None,
    encode_kwargs: Dict[str, Any] = None,
    decode_kwargs: Dict[str, Any] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    solver: AbstractSolver = Dopri5(),
) -> Tuple[TaskCallables, jm.Metrics]:
    """
    Factory function for the task of learning a representation using first-principle dynamics while using the
    ground-truth velocity.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        nn_model: the neural network model to use
        encode_fn: the function to use for encoding the input image to the latent space
        decode_fn: the function to use for decoding the latent space to the output image
        encode_kwargs: additional kwargs to pass to the encode_fn
        decode_kwargs: additional kwargs to pass to the decode_fn
        ode_fn: ODE function. It should have the following signature:
            ode_fn(t, x) -> x_dot
        loss_weights: the weights for the different loss terms
        solver: Diffrax solver to use for the simulation.
    Returns:
        task_callables: struct containing the functions for the learning task
        metrics: struct containing the metrics for the learning task
    """
    if encode_fn is None:
        encode_fn = nn_model.encode
    if decode_fn is None:
        decode_fn = nn_model.decode
    if encode_kwargs is None:
        encode_kwargs = {}
    if decode_kwargs is None:
        decode_kwargs = {}

    if loss_weights is None:
        loss_weights = dict(mse_q=1.0, mse_rec_static=1.0, mse_rec_dynamic=1.0)

    # initiate ODE term from `ode_fn`
    ode_term = ODETerm(ode_fn)

    @partial(jit, static_argnames="training")
    def forward_fn(
        batch: Dict[str, Array], nn_params: FrozenDict, training: bool = False
    ) -> Dict[str, Array]:
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

        # specify initial state for the dynamic rollout
        q_0_bt = q_static_pred_bt[
            :, 0, ...
        ]  # initial configuration at time t=0 as provided by the encoder
        q_d_0_bt = batch["x_ts"][
            :, 0, n_q:
        ]  # initial ground-truth velocity at time t=0
        x_0_bt = jnp.concatenate(
            (q_0_bt, q_d_0_bt), axis=-1
        )  # initial state at time t=0

        # compute the dynamic rollout of the latent representation
        ode_solve_fn = partial(
            diffeqsolve,
            ode_term,
            solver,
            saveat=SaveAt(ts=t_ts),
            max_steps=t_ts.shape[-1],
        )
        # simulate
        sol_bt = vmap(ode_solve_fn, in_axes=(None, None, None, 0))(
            t_ts[0],  # initial time
            t_ts[-1],  # final time
            dt,  # time step
            x_0_bt.astype(jnp.float64),  # initial state
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
            **decode_kwargs
        )
        img_dynamic_pred_flat_bt = nn_model.apply(
            {"params": nn_params},
            decoder_dynamic_input,
            method=decode_fn,
            **decode_kwargs
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

    @partial(jit, static_argnames="training")
    def loss_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[KeyArray] = None,
        training: bool = False,
    ) -> Tuple[Array, Dict[str, Array]]:
        preds = forward_fn(batch, nn_params, training=training)

        q_static_pred_bt = preds["q_static_ts"]
        q_dynamic_pred_bt = preds["q_dynamic_ts"]

        # compute the configuration error
        error_q = q_dynamic_pred_bt - q_static_pred_bt

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
            jnp.square(preds["rendering_dynamic_ts"] - batch["rendering_ts"])
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
        error_q_dynamic = q_dynamic_pred_bt - q_target_bt

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
                    jnp.square(preds["rendering_dynamic_ts"] - batch["rendering_ts"])
                )
            ),
        }
        return metrics

    task_callables = TaskCallables(
        system_type, assemble_input, forward_fn, loss_fn, compute_metrics
    )

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
