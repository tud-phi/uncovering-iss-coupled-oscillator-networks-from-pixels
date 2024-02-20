from clu import metrics as clu_metrics
from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from flax.core import FrozenDict
from flax.struct import dataclass
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, random
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Callable, Dict, Optional, Tuple, Type

from src.metrics import RootAverage
from src.structs import TaskCallables


def assemble_input(batch) -> Tuple[Array]:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return (img_bt,)


def task_factory(
    system_type: str, nn_model: nn.Module
) -> Tuple[TaskCallables, Type[clu_metrics.Collection]]:
    def forward_fn(
        batch: Dict[str, Array], nn_params: FrozenDict, training: bool = False
    ) -> Dict[str, Array]:
        (img_bt,) = assemble_input(batch)

        # output will be of shape batch_dim * time_dim x latent_dim
        q_pred_bt = nn_model.apply({"params": nn_params}, img_bt)

        # if necessary, normalize the joint angles
        if system_type == "pendulum":
            q_pred_bt = normalize_joint_angles(q_pred_bt)

        # reshape to batch_dim x time_dim x latent_dim
        q_pred_bt = q_pred_bt.reshape(
            (batch["rendering_ts"].shape[0], -1, q_pred_bt.shape[-1])
        )

        preds = dict(q_ts=q_pred_bt)

        return preds

    def loss_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[Array] = None,
        training: bool = False,
    ) -> Tuple[Array, Dict[str, Array]]:
        preds = forward_fn(batch, nn_params, training=training)

        q_pred_bt = preds["q_ts"]
        q_target_bt = batch["x_ts"][..., : batch["x_ts"].shape[-1] // 2]

        # compute the configuration error
        error_q = q_pred_bt - q_target_bt

        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)

        # compute the mean squared error
        mse = jnp.mean(jnp.square(error_q))

        return mse, preds

    def compute_metrics_fn(
        batch: Dict[str, Array], preds: Dict[str, Array]
    ) -> Dict[str, Array]:
        q_pred_bt = preds["q_ts"]
        q_target_bt = batch["x_ts"][..., : batch["x_ts"].shape[-1] // 2]

        # compute the configuration error
        error_q = q_pred_bt - q_target_bt

        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)

        return {
            "mse_q": jnp.mean(jnp.square(error_q)),
        }

    task_callables = TaskCallables(
        system_type, assemble_input, forward_fn, loss_fn, compute_metrics_fn
    )

    @dataclass  # <-- required for JAX transformations
    class MetricsCollection(clu_metrics.Collection):
        loss: clu_metrics.Average.from_output("loss")
        lr: clu_metrics.LastValue.from_output("lr")
        rmse_q: RootAverage.from_output("mse_q")

    metrics_collection_cls = MetricsCollection
    return task_callables, metrics_collection_cls
