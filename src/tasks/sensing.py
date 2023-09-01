from flax.core import FrozenDict
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, jit, random
import jax.numpy as jnp
import jax_metrics as jm
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Callable, Dict, Optional, Tuple

from src.metrics import NoReduce, RootMean
from src.structs import TaskCallables


@jit
def assemble_input(batch) -> Array:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return img_bt


def task_factory(
    system_type: str, nn_model: nn.Module
) -> Tuple[TaskCallables, jm.Metrics]:
    @partial(jit, static_argnames="training")
    def forward_fn(
        batch: Dict[str, Array], nn_params: FrozenDict, training: bool = False
    ) -> Dict[str, Array]:
        img_bt = assemble_input(batch)

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

    @partial(jit, static_argnames="training")
    def loss_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[random.KeyArray] = None,
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
            "mse_q": jnp.mean(jnp.square(error_q)),
        }
        return metrics

    metrics = jm.Metrics(
        {
            "loss": jm.metrics.Mean().from_argument("loss"),
            "lr": NoReduce().from_argument("lr"),
            "rmse_q": RootMean().from_argument("mse_q"),
        }
    )

    task_callables = TaskCallables(
        system_type, assemble_input, forward_fn, loss_fn, compute_metrics
    )
    return task_callables, metrics
