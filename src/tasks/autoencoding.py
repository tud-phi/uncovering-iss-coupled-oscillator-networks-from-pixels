from flax.core import FrozenDict
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, jit
import jax.numpy as jnp
import jax_metrics as jm
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Callable, Dict, Tuple

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
    system_type: str, nn_model: nn.Module, loss_weights: Dict[str, float] = None
) -> Tuple[TaskCallables, jm.Metrics]:
    """
    Factory function for the autoencoding task.
    I.e. the task of reconstructing the input image with the latent space supervised by the configuration.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        nn_model: the neural network model to use
        loss_weights: the weights for the different loss terms
    Returns:
        task_callables: struct containing the functions for the learning task
        metrics: struct containing the metrics for the learning task
    """
    if loss_weights is None:
        loss_weights = dict(mse_q=1.0, mse_rec=1.0)

    @jit
    def forward_fn(batch: Dict[str, Array], nn_params: FrozenDict) -> Dict[str, Array]:
        img_bt = assemble_input(batch)
        batch_size = batch["rendering_ts"].shape[0]
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        # output will be of shape batch_dim * time_dim x latent_dim
        q_pred_bt = nn_model.apply(
            {"params": nn_params}, img_bt, method=nn_model.encode
        )

        if system_type == "pendulum":
            # if the system is a pendulum, we interpret the encoder output as sin(theta) and cos(theta) for each joint
            # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
            # output of arctan2 will be in the range [-pi, pi]
            q_pred_bt = jnp.arctan2(q_pred_bt[..., :n_q], q_pred_bt[..., n_q:])

            # if the system is a pendulum, the input into the decoder should be sin(theta) and cos(theta) for each joint
            # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
            input_decoder = jnp.concatenate(
                [jnp.sin(q_pred_bt), jnp.cos(q_pred_bt)], axis=-1
            )
        else:
            input_decoder = q_pred_bt

        # output will be of shape batch_dim * time_dim x width x height x channels
        img_pred_bt = nn_model.apply(
            {"params": nn_params}, input_decoder, method=nn_model.decode
        )

        # reshape to batch_dim x time_dim x ...
        q_pred_bt = q_pred_bt.reshape((batch_size, -1, *q_pred_bt.shape[1:]))
        img_pred_bt = img_pred_bt.reshape((batch_size, -1, *img_pred_bt.shape[1:]))

        preds = dict(q_ts=q_pred_bt, rendering_ts=img_pred_bt)

        return preds

    @jit
    def loss_fn(
        batch: Dict[str, Array], nn_params: FrozenDict
    ) -> Tuple[Array, Dict[str, Array]]:
        preds = forward_fn(batch, nn_params)

        q_pred_bt = preds["q_ts"]
        q_target_bt = batch["x_ts"][..., : batch["x_ts"].shape[-1] // 2]

        # compute the configuration error
        error_q = q_pred_bt - q_target_bt

        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)

        # compute the mean squared error
        mse_q = jnp.mean(jnp.square(error_q))

        # supervised MSE loss on the reconstructed image
        mse_rec = jnp.mean(jnp.square(preds["rendering_ts"] - batch["rendering_ts"]))

        # total loss
        loss = loss_weights["mse_q"] * mse_q + loss_weights["mse_rec"] * mse_rec

        return loss, preds

    @jit
    def compute_metrics(
        batch: Dict[str, Array], preds: Dict[str, Array]
    ) -> Dict[str, Array]:
        q_pred_bt = preds["q_ts"]
        q_target_bt = batch["x_ts"][..., : batch["x_ts"].shape[-1] // 2]

        # compute the configuration error
        error_q = q_pred_bt - q_target_bt

        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)

        metrics = {
            "rmse_q": jnp.sqrt(jnp.mean(jnp.square(error_q))),
            "rmse_rec": jnp.sqrt(
                jnp.mean(jnp.square(preds["rendering_ts"] - batch["rendering_ts"]))
            ),
        }
        return metrics

    task_callables = TaskCallables(assemble_input, forward_fn, loss_fn, compute_metrics)

    metrics = jm.Metrics(
        {
            "loss": jm.metrics.Mean().from_argument("loss"),
            "lr": NoReduce().from_argument("lr"),
            "rmse_q": jm.metrics.Mean().from_argument("rmse_q"),
            "rmse_rec": jm.metrics.Mean().from_argument("rmse_rec"),
        }
    )

    return task_callables, metrics
