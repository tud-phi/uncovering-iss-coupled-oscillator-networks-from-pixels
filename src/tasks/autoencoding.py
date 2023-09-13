from clu import metrics as clu_metrics
from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from flax.core import FrozenDict
from flax.struct import dataclass
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, debug, random
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Any, Callable, Dict, Optional, Tuple, Type

from src.losses.kld import kullback_leiber_divergence
from src.losses.masked_mse import masked_mse_loss
from src.metrics import RootAverage
from src.structs import TaskCallables


def assemble_input(batch) -> Array:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return img_bt


def task_factory(
    system_type: str,
    nn_model: nn.Module,
    encode_fn: Callable = None,
    decode_fn: Callable = None,
    encode_kwargs: Dict[str, Any] = None,
    decode_kwargs: Dict[str, Any] = None,
    loss_weights: Dict[str, float] = None,
    normalize_latent_space: bool = True,
    rec_loss_type: str = "mse",
    weight_on_foreground: float = None,
    ae_type: str = "None",
) -> Tuple[TaskCallables, Type[clu_metrics.Collection]]:
    """
    Factory function for the autoencoding task.
    I.e. the task of reconstructing the input image with the latent space supervised by the configuration.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        nn_model: the neural network model to use
        encode_fn: the function to use for encoding the input image to the latent space
        decode_fn: the function to use for decoding the latent space to the output image
        encode_kwargs: additional kwargs to pass to the encode_fn
        decode_kwargs: additional kwargs to pass to the decode_fn
        loss_weights: the weights for the different loss terms
        normalize_latent_space: whether to normalize the latent space by for example projecting angles to [-pi, pi]
        rec_loss_type: the type of reconstruction loss to use. One of ["mse", "bce"]
        weight_on_foreground: if None, a normal MSE loss will be used. Otherwise, a masked MSE loss will be used
            with the given weight for the masked area (usually the foreground).
        ae_type: Autoencoder type. If None, a normal autoencoder will be used.
            One of ["wae", "beta_vae", "None"]
    Returns:
        task_callables: struct containing the functions for the learning task
        metrics_collection_cls: contains class for collecting metrics
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
        if rec_loss_type == "mse":
            loss_weights = dict(mse_q=0.0, mse_rec=1.0)
        elif rec_loss_type == "bce":
            loss_weights = dict(mse_q=0.0, bce_rec=1.0)
        else:
            raise ValueError(f"Unknown reconstruction loss type: {rec_loss_type}")

    if weight_on_foreground is not None:
        assert rec_loss_type == "mse", "Only MSE loss is supported for masked MSE loss"

    if ae_type == "wae":
        from src.losses import wae

        if system_type == "pendulum":
            uniform_distr_range = (-jnp.pi, jnp.pi)
        else:
            uniform_distr_range = (-1.0, 1.0)
        wae_mmd_loss_fn = wae.make_wae_mdd_loss(
            distribution="uniform", uniform_distr_range=uniform_distr_range
        )

    def forward_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[random.KeyArray] = None,
        training: bool = False,
    ) -> Dict[str, Array]:
        img_bt = assemble_input(batch)
        batch_size = batch["rendering_ts"].shape[0]
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        if ae_type == "beta_vae":
            # output will be of shape batch_dim * time_dim x latent_dim
            mu_bt, logvar_bt = nn_model.apply(
                {"params": nn_params},
                img_bt,
                method=nn_model.encode_vae,
                **encode_kwargs,
            )
            if training is True:
                # reparameterize
                z_pred_bt = nn_model.reparameterize(rng, mu_bt, logvar_bt)
            else:
                z_pred_bt = mu_bt
        else:
            # output will be of shape batch_dim * time_dim x latent_dim
            z_pred_bt = nn_model.apply(
                {"params": nn_params}, img_bt, method=encode_fn, **encode_kwargs
            )

        if normalize_latent_space and system_type == "pendulum":
            latent_dim = z_pred_bt.shape[-1] // 2
            # if the system is a pendulum, we interpret the encoder output as sin(theta) and cos(theta) for each joint
            # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
            # output of arctan2 will be in the range [-pi, pi]
            z_pred_bt = jnp.arctan2(
                z_pred_bt[..., :latent_dim], z_pred_bt[..., latent_dim:]
            )

            # if the system is a pendulum, the input into the decoder should be sin(theta) and cos(theta) for each joint
            # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
            input_decoder = jnp.concatenate(
                [jnp.sin(z_pred_bt), jnp.cos(z_pred_bt)], axis=-1
            )
        else:
            input_decoder = z_pred_bt

        # output will be of shape batch_dim * time_dim x width x height x channels
        img_pred_bt = nn_model.apply(
            {"params": nn_params}, input_decoder, method=decode_fn, **decode_kwargs
        )

        # reshape to batch_dim x time_dim x ...
        z_pred_bt = z_pred_bt.reshape((batch_size, -1, *z_pred_bt.shape[1:]))
        img_pred_bt = img_pred_bt.reshape((batch_size, -1, *img_pred_bt.shape[1:]))
        preds = dict(q_ts=z_pred_bt, rendering_ts=img_pred_bt)

        if ae_type == "beta_vae":
            preds["mu_ts"] = mu_bt.reshape((batch_size, -1, *mu_bt.shape[1:]))
            preds["logvar_ts"] = logvar_bt.reshape(
                (batch_size, -1, *logvar_bt.shape[1:])
            )

        return preds

    def loss_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[random.KeyArray] = None,
        training: bool = False,
    ) -> Tuple[Array, Dict[str, Array]]:
        preds = forward_fn(batch, nn_params, rng=rng, training=training)

        q_pred_bt = preds["q_ts"]
        q_target_bt = batch["x_ts"][..., : batch["x_ts"].shape[-1] // 2]

        # compute the configuration error
        error_q = q_pred_bt - q_target_bt

        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)

        # compute the mean squared error
        mse_q = jnp.mean(jnp.square(error_q))

        if rec_loss_type == "bce":
            from optax import sigmoid_binary_cross_entropy

            # supervised binary cross entropy loss on the reconstructed image
            # first, we need to bring predictions and targets from the range [-1.0, 1.0] into the range [0, 1]
            img_label_bt = jnp.round(batch["rendering_ts"] / 2 + 0.5, decimals=0)
            img_pred_bt = preds["rendering_ts"] / 2 + 0.5
            # then, we can compute the binary cross entropy loss
            rec_loss = jnp.mean(sigmoid_binary_cross_entropy(img_pred_bt, img_label_bt))
            # debug.print("rec_loss = {rec_loss}", rec_loss=rec_loss)
            # multiply with loss weight
            rec_loss = loss_weights["bce_rec"] * rec_loss

        elif rec_loss_type == "mse":
            # supervised MSE loss on the reconstructed image
            if weight_on_foreground is None:
                rec_loss = jnp.mean(
                    jnp.square(preds["rendering_ts"] - batch["rendering_ts"])
                )
            else:
                # allows to equally weigh the importance of correctly reconstructing the foreground and background
                rec_loss = masked_mse_loss(
                    preds["rendering_ts"],
                    batch["rendering_ts"],
                    threshold_cond_sign=-1,
                    weight_loss_masked_area=weight_on_foreground,
                )
            rec_loss = loss_weights["mse_rec"] * rec_loss

        # total loss
        loss = loss_weights["mse_q"] * mse_q + rec_loss

        if ae_type == "wae":
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
        elif ae_type == "beta_vae":
            # KLD loss
            # https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/beta_vae/beta_vae_model.py#L101
            kld_loss = kullback_leiber_divergence(preds["mu_ts"], preds["logvar_ts"])

            loss = loss + loss_weights["beta"] * kld_loss

        return loss, preds

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
            "mse_rec": jnp.mean(
                jnp.square(preds["rendering_ts"] - batch["rendering_ts"])
            ),
        }

        if weight_on_foreground is not None:
            # allows to equally weigh the importance of correctly reconstructing the foreground and background
            metrics["masked_mse_rec"] = masked_mse_loss(
                preds["rendering_ts"],
                batch["rendering_ts"],
                threshold_cond_sign=-1,
                weight_loss_masked_area=weight_on_foreground,
            )

        if ae_type == "beta_vae":
            # KLD loss
            metrics["kld"] = kullback_leiber_divergence(
                preds["mu_ts"], preds["logvar_ts"]
            )

        return metrics

    task_callables = TaskCallables(
        system_type, assemble_input, forward_fn, loss_fn, compute_metrics
    )

    @dataclass  # <-- required for JAX transformations
    class MetricsCollection(clu_metrics.Collection):
        loss: clu_metrics.Average.from_output("loss")
        lr: clu_metrics.LastValue.from_output("lr")
        rmse_q: RootAverage.from_output("mse_q")
        rmse_rec: RootAverage.from_output("mse_rec")

        if weight_on_foreground is not None:
            rmse_rec: RootAverage.from_output("masked_mse_rec")

        if ae_type == "beta_vae":
            kld: clu_metrics.Average.from_output("kld")

    metrics_collection_cls = MetricsCollection

    return task_callables, metrics_collection_cls
