from flax.core import FrozenDict
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, debug, jit, random
import jax.numpy as jnp
import jax_metrics as jm
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Callable, Dict, Optional, Tuple

from src.losses.masked_mse import masked_mse_loss
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
    loss_weights: Dict[str, float] = None,
    normalize_latent_space: bool = True,
    weight_on_foreground: float = None,
    use_wae: bool = False,
) -> Tuple[TaskCallables, jm.Metrics]:
    """
    Factory function for the autoencoding task.
    I.e. the task of reconstructing the input image with the latent space supervised by the configuration.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        nn_model: the neural network model to use
        loss_weights: the weights for the different loss terms
        normalize_latent_space: whether to normalize the latent space by for example projecting angles to [-pi, pi]
        weight_on_foreground: if None, a normal MSE loss will be used. Otherwise, a masked MSE loss will be used
            with the given weight for the masked area (usually the foreground).
        use_wae: whether to apply the Wasserstein Autoencoder regularization
    Returns:
        task_callables: struct containing the functions for the learning task
        metrics: struct containing the metrics for the learning task
    """
    if loss_weights is None:
        loss_weights = dict(mse_q=1.0, mse_rec=1.0)

    if use_wae:
        from src.losses import wae

        if system_type == "pendulum":
            uniform_distr_range = (-jnp.pi, jnp.pi)
        else:
            uniform_distr_range = (-1.0, 1.0)
        wae_mmd_loss_fn = wae.make_wae_mdd_loss(
            distribution="uniform", uniform_distr_range=uniform_distr_range
        )

    @jit
    def forward_fn(batch: Dict[str, Array], nn_params: FrozenDict) -> Dict[str, Array]:
        img_bt = assemble_input(batch)
        batch_size = batch["rendering_ts"].shape[0]
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        # output will be of shape batch_dim * time_dim x latent_dim
        q_pred_bt = nn_model.apply(
            {"params": nn_params}, img_bt, method=nn_model.encode
        )

        if normalize_latent_space and system_type == "pendulum":
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
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[random.PRNGKey] = None,
        step: Optional[int] = 0
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
        if weight_on_foreground is None:
            mse_rec = jnp.mean(
                jnp.square(preds["rendering_ts"] - batch["rendering_ts"])
            )
        else:
            # allows to equally weigh the importance of correctly reconstructing the foreground and background
            mse_rec = masked_mse_loss(
                preds["rendering_ts"],
                batch["rendering_ts"],
                threshold_cond_sign=-1,
                weight_loss_masked_area=weight_on_foreground,
            )

        # total loss
        loss = loss_weights["mse_q"] * mse_q + loss_weights["mse_rec"] * mse_rec

        if use_wae:
            latent_dim = preds["q_ts"].shape[-1]

            img_target_bt = assemble_input(batch)
            img_pred_bt = preds["rendering_ts"].reshape(
                (-1, *preds["rendering_ts"].shape[2:])
            )
            q_pred_bt = preds["q_ts"].reshape((-1, latent_dim))

            # Wasserstein Autoencoder MMD loss
            mmd_loss = wae_mmd_loss_fn(
                x_rec=img_pred_bt, x_target=img_target_bt, z=q_pred_bt, rng=rng
            )

            loss = loss + loss_weights["mmd"] * mmd_loss

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

        if weight_on_foreground is not None:
            # allows to equally weigh the importance of correctly reconstructing the foreground and background
            metrics["masked_rmse_rec"] = masked_mse_loss(
                preds["rendering_ts"],
                batch["rendering_ts"],
                threshold_cond_sign=-1,
                weight_loss_masked_area=weight_on_foreground,
            )
        return metrics

    task_callables = TaskCallables(assemble_input, forward_fn, loss_fn, compute_metrics)

    accumulated_metrics_dict = {
        "loss": jm.metrics.Mean().from_argument("loss"),
        "lr": NoReduce().from_argument("lr"),
        "rmse_q": jm.metrics.Mean().from_argument("rmse_q"),
        "rmse_rec": jm.metrics.Mean().from_argument("rmse_rec"),
    }

    if weight_on_foreground is not None:
        accumulated_metrics_dict["masked_rmse_rec"] = jm.metrics.Mean().from_argument(
            "masked_rmse_rec"
        )

    accumulated_metrics = jm.Metrics(accumulated_metrics_dict)

    return task_callables, accumulated_metrics
