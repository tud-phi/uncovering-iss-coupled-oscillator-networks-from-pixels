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


def task_factory(nn_model: nn.Module) -> TaskCallables:
    @jit
    def predict_fn(batch: Dict[str, Array], nn_params: FrozenDict) -> Dict[str, Array]:
        img_bt = assemble_input(batch)

        # TODO: only normalize for pendulums, but not for soft robots or other systems
        # output will be of shape batch_dim * time_dim x latent_dim
        q_pred_bt = normalize_joint_angles(nn_model.apply({"params": nn_params}, img_bt))

        # reshape to batch_dim x time_dim x latent_dim
        q_pred_bt = q_pred_bt.reshape((batch["rendering_ts"].shape[0], -1, q_pred_bt.shape[-1]))

        preds = dict(
            q_ts=q_pred_bt
        )

        return preds

    @jit
    def loss_fn(batch: Dict[str, Array], nn_params: FrozenDict) -> Tuple[Array, Dict[str, Array]]:
        preds = predict_fn(batch, nn_params)

        q_pred_bt = preds["q_ts"]
        q_target_bt = batch["x_ts"][..., :batch["x_ts"].shape[-1] // 2]

        # TODO: only normalize for pendulums, but not for soft robots or other systems
        mse = jnp.mean(jnp.square(
            normalize_joint_angles(q_pred_bt - q_target_bt)
        ))
        return mse, preds

    @jit
    def compute_metrics(batch: Dict[str, Array], preds: Dict[str, Array]) -> Dict[str, Array]:
        q_pred_bt = preds["q_ts"]
        q_target_bt = batch["x_ts"][..., :batch["x_ts"].shape[-1] // 2]

        # compute the normalized joint angle error
        error_q = normalize_joint_angles(q_pred_bt - q_target_bt)

        metrics = {
            "rmse_q": jnp.sqrt(jnp.mean(jnp.square(error_q))),
        }
        return metrics

    task_callables = TaskCallables(assemble_input, predict_fn, loss_fn, compute_metrics)
    return task_callables
