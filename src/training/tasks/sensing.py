from flax.core import FrozenDict
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, jit
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Callable, Dict, Tuple

from src.structs import TaskCallables


@jit
def preprocess_batch(batch) -> Array:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return img_bt


def task_factory(nn_model: nn.Module) -> TaskCallables:
    @jit
    def model_forward_fn(batch: Dict[str, Array], nn_params: FrozenDict) -> Array:
        img_bt = preprocess_batch(batch)

        # output will be of shape batch_dim * time_dim x latent_dim
        q_pred_bt = nn_model.apply({"params": nn_params}, img_bt)

        # reshape to batch_dim x time_dim x latent_dim
        q_pred_bt = q_pred_bt.reshape((batch["rendering_ts"].shape[0], -1, q_pred_bt.shape[-1]))

        return q_pred_bt

    @jit
    def loss_fn(batch: Dict[str, Array], nn_params: FrozenDict) -> Tuple[Array, Array]:
        q_pred_bt = model_forward_fn(batch, nn_params)
        q_target_bt = batch["x_ts"][..., :batch["x_ts"].shape[-1] // 2]
        mse = jnp.mean(jnp.square(
            normalize_joint_angles(q_pred_bt - q_target_bt)
        ))
        return mse, q_target_bt

    task_callables = TaskCallables(preprocess_batch, model_forward_fn, loss_fn)
    return task_callables
