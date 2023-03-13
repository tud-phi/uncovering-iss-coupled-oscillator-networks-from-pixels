from flax.core import FrozenDict
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, jit
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Callable, Dict, Tuple

from src.structs import TaskCallables


@jit
def assemble_input(batch) -> Array:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return img_bt


def task_factory(
    nn_model: nn.Module, loss_weights: Dict[str, float] = None
) -> TaskCallables:
    """
    Factory function for the autoencoding task.
    I.e. the task of reconstructing the input image with the latent space supervised by the joint angles.
    Will return a TaskCallables object with the predict_fn, loss_fn, and compute_metrics functions.
    nn_model: the neural network model to use
    loss_weights: the weights for the different loss terms
    """
    if loss_weights is None:
        loss_weights = dict(mse_q=1.0, mse_rec=1.0)

    @jit
    def predict_fn(batch: Dict[str, Array], nn_params: FrozenDict) -> Dict[str, Array]:
        img_bt = assemble_input(batch)
        batch_size = batch["rendering_ts"].shape[0]

        # output will be of shape batch_dim * time_dim x latent_dim
        q_pred_bt = nn_model.apply(
            {"params": nn_params}, img_bt, method=nn_model.encode
        )
        # output will be of shape batch_dim * time_dim x width x height x channels
        img_pred_bt = nn_model.apply(
            {"params": nn_params}, q_pred_bt, method=nn_model.decode
        )

        # reshape to batch_dim x time_dim x ...
        q_pred_bt = q_pred_bt.reshape((batch_size, -1, *q_pred_bt.shape[1:]))
        img_pred_bt = img_pred_bt.reshape((batch_size, -1, *img_pred_bt.shape[1:]))

        # normalize the joint angles
        # TODO: only normalize for pendulums, but not for soft robots or other systems
        q_pred_bt = normalize_joint_angles(q_pred_bt)

        preds = dict(q_ts=q_pred_bt, rendering_ts=img_pred_bt)

        return preds

    @jit
    def loss_fn(
        batch: Dict[str, Array], nn_params: FrozenDict
    ) -> Tuple[Array, Dict[str, Array]]:
        preds = predict_fn(batch, nn_params)

        q_pred_bt = preds["q_ts"]
        q_target_bt = batch["x_ts"][..., : batch["x_ts"].shape[-1] // 2]

        # supervised MSE loss on the configuration space
        # TODO: only normalize for pendulums, but not for soft robots or other systems
        mse_q = jnp.mean(jnp.square(normalize_joint_angles(q_pred_bt - q_target_bt)))

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

        # compute the normalized joint angle error
        error_q = normalize_joint_angles(q_pred_bt - q_target_bt)

        metrics = {
            "rmse_q": jnp.sqrt(jnp.mean(jnp.square(error_q))),
            "mse_rec": jnp.sqrt(
                jnp.mean(jnp.square(preds["rendering_ts"] - batch["rendering_ts"]))
            ),
        }
        return metrics

    task_callables = TaskCallables(assemble_input, predict_fn, loss_fn, compute_metrics)
    return task_callables
