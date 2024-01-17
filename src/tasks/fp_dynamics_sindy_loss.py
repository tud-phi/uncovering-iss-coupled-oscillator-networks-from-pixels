from clu import metrics as clu_metrics
from flax.core import FrozenDict
from flax.struct import dataclass
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, debug, jacfwd, jacrev, jit, jvp, random, vmap
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Any, Callable, Dict, Optional, Tuple, Type

from src.metrics import RootAverage
from src.losses.kld import kullback_leiber_divergence
from src.structs import TaskCallables


def assemble_input(batch) -> Array:
    # batch of images
    rendering_ts = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    rendering_bt = rendering_ts.reshape((-1, *rendering_ts.shape[2:]))

    return rendering_bt


def task_factory(
    system_type: str,
    nn_model: nn.Module,
    ode_fn: Callable,
    ts: Array,
    encode_fn: Optional[Callable] = None,
    decode_fn: Optional[Callable] = None,
    encode_kwargs: Optional[Dict[str, Any]] = None,
    decode_kwargs: Optional[Dict[str, Any]] = None,
    x0_min: Optional[Array] = None,
    x0_max: Optional[Array] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    ae_type: str = "None",
    normalize_configuration_loss=False,
) -> Tuple[TaskCallables, Type[clu_metrics.Collection]]:
    """
    Factory function for the task of learning a representation using first-principle dynamics while using the
    ground-truth velocity.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        nn_model: the neural network model to use
        ode_fn: ODE function. It should have the following signature:
            ode_fn(t, x) -> x_d
        ts: time steps of the samples
        encode_fn: the function to use for encoding the input image to the latent space
        decode_fn: the function to use for decoding the latent space to the output image
        encode_kwargs: additional kwargs to pass to the encode_fn
        decode_kwargs: additional kwargs to pass to the decode_fn
        x0_min: the minimal value for the initial state of the simulation
        x0_max: the maximal value for the initial state of the simulation
        loss_weights: the weights for the different loss terms
        ae_type: Autoencoder type. If None, a normal autoencoder will be used.
            One of ["wae", "beta_vae", "None"]
        normalize_configuration_loss: whether to normalize the configuration loss dividing by (q0_max - q0_min)
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
        loss_weights = {}
    loss_weights = (
        dict(mse_rec=1.0, mse_sindy_q_dd=1.0, mse_sindy_rendering_dd=1.0) | loss_weights
    )

    if ae_type == "wae":
        loss_weights = dict(mmd=1.0) | loss_weights

        assert system_type == "pendulum", "WAE only implemented for pendulum system"
        if system_type == "pendulum":
            uniform_distr_range = (-jnp.pi, jnp.pi)
        else:
            uniform_distr_range = (-1.0, 1.0)

        from src.losses import wae

        wae_mmd_loss_fn = wae.make_wae_mdd_loss(
            distribution="uniform", uniform_distr_range=uniform_distr_range
        )
    elif ae_type == "beta_vae":
        loss_weights = dict(beta=1.0) | loss_weights

    if normalize_configuration_loss is True:
        assert (
            x0_min is not None and x0_max is not None
        ), "x0_min and x0_max must be provided for normalizing the configuration loss"

    def forward_fn(
        batch: Dict[str, Array],
        nn_params: FrozenDict,
        rng: Optional[random.KeyArray] = None,
        training: bool = False,
    ) -> Dict[str, Array]:
        rendering_bt = assemble_input(batch)
        rendering_d_ts = batch["rendering_d_ts"]
        rendering_dd_ts = batch["rendering_dd_ts"]
        rendering_d_bt = rendering_d_ts.reshape((-1, *rendering_d_ts.shape[2:]))
        rendering_dd_bt = rendering_dd_ts.reshape((-1, *rendering_dd_ts.shape[2:]))

        batch_size = batch["rendering_ts"].shape[0]
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        def _encode(_rendering_bt: Array) -> Tuple[Array, Dict[str, Array]]:
            # static predictions by passing the image through the encoder
            # output will be of shape batch_dim * time_dim x latent_dim
            # if the system is a pendulum, the latent dim should be 2*n_q
            _model_output = nn_model.apply(
                {"params": nn_params}, _rendering_bt, method=encode_fn, **encode_kwargs
            )

            if ae_type == "beta_vae":
                _mu_bt, _logvar_bt = _model_output
                if training is True:
                    # reparameterize
                    _q_bt = nn_model.reparameterize(rng, _mu_bt, _logvar_bt)
                else:
                    _q_bt = _mu_bt
            else:
                _q_bt = _model_output
                _mu_bt = jnp.zeros_like(_q_bt)
                _logvar_bt = jnp.zeros_like(_mu_bt)

            if system_type == "pendulum":
                # if the system is a pendulum, we interpret the encoder output as sin(theta) and cos(theta) for each joint
                # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
                # output of arctan2 will be in the range [-pi, pi]
                _q_bt = jnp.arctan2(_q_bt[..., :n_q], _q_bt[..., n_q:])

            _aux_bt = dict(
                mu_bt=_mu_bt,
                logvar_bt=_logvar_bt,
            )

            return _q_bt, _aux_bt

        # compute the configuration and its velocity (using the Jacobian-vector product)
        q_pred_bt, q_d_pred_bt, aux_bt = jvp(
            _encode, (rendering_bt,), (rendering_d_bt,), has_aux=True
        )
        mu_bt, logvar_bt = aux_bt["mu_bt"], aux_bt["logvar_bt"]

        # compute jacobian of the encoder and its derivative
        jac_encoder_bt, jac_d_encoder_bt, _ = jvp(
            jacrev(_encode, has_aux=True),
            (rendering_bt,),
            (rendering_d_bt,),
            has_aux=True,
        )
        # compute the configuration acceleration (using the product rule)
        # q_dd = J @ x_dd + J_d @ x_d
        q_dd_pred_bt = jnp.tensordot(
            jac_encoder_bt, rendering_dd_bt, axes=4
        ) + jnp.tensordot(jac_d_encoder_bt, rendering_d_bt, axes=4)

        # construct ts_bt
        ts_bt = jnp.repeat(ts[None, ...], repeats=batch_size, axis=0).flatten()
        # stack the q and q_d to the state vector
        x_bt = jnp.concatenate((q_pred_bt, q_d_pred_bt), axis=-1)
        # construct tau_ts
        tau_ts = jnp.repeat(batch["tau"][:, None, ...], repeats=ts.shape[0], axis=1)
        tau_bt = tau_ts.reshape((-1, *tau_ts.shape[2:]))
        # evaluate the dynamics function at the state vector
        x_d_ode_bt = vmap(ode_fn)(ts_bt, x_bt, tau_bt)
        # acceleration of the generalized/latent coordinates
        q_dd_ode_pred_bt = x_d_ode_bt[..., n_q:]

        def _decode(_q_bt: Array) -> Array:
            if system_type == "pendulum":
                # if the system is a pendulum, the input into the decoder should be sin(theta) and cos(theta) for each joint
                # e.g. for two joints: z = [sin(q_1), sin(q_2), cos(q_1), cos(q_2)]
                _decoder_input = jnp.concatenate(
                    [jnp.sin(_q_bt), jnp.cos(_q_bt)],
                    axis=-1,
                )
            else:
                _decoder_input = _q_bt

            # send the latent representations through the decoder
            _rendering_bt = nn_model.apply(
                {"params": nn_params},
                _decoder_input,
                method=decode_fn,
                **decode_kwargs,
            )

            return _rendering_bt

        # compute the rendering and its velocity (using the Jacobian-vector product)
        rendering_pred_bt, rendering_d_pred_bt = jvp(
            _decode, (q_pred_bt,), (q_d_pred_bt,)
        )

        # compute jacobian of the decoder and its derivative
        jac_decoder_bt, jac_d_decoder_bt = jvp(
            jacfwd(_decode), (q_pred_bt,), (q_d_pred_bt,)
        )
        # compute the rendering acceleration (using the product rule)
        rendering_dd_pred_bt = jnp.tensordot(
            jac_decoder_bt, q_dd_pred_bt, axes=2
        ) + jnp.tensordot(jac_d_decoder_bt, q_d_pred_bt, axes=2)

        # reshape to batch_dim x time_dim x n_q
        q_pred_ts = q_pred_bt.reshape((batch_size, -1, *q_pred_bt.shape[1:]))
        q_d_pred_ts = q_d_pred_bt.reshape((batch_size, -1, *q_d_pred_bt.shape[1:]))
        q_dd_pred_ts = q_dd_pred_bt.reshape((batch_size, -1, *q_dd_pred_bt.shape[1:]))
        q_dd_ode_pred_ts = q_dd_ode_pred_bt.reshape(
            (batch_size, -1, *q_dd_ode_pred_bt.shape[1:])
        )

        # reshape to batch_dim x time_dim x width x height x channels
        rendering_pred_ts = rendering_pred_bt.reshape(
            (batch_size, -1, *rendering_pred_bt.shape[1:])
        )
        rendering_d_pred_ts = rendering_d_pred_bt.reshape(
            (batch_size, -1, *rendering_d_pred_bt.shape[1:])
        )
        rendering_dd_pred_ts = rendering_dd_pred_bt.reshape(
            (batch_size, -1, *rendering_dd_pred_bt.shape[1:])
        )

        preds = dict(
            q_ts=q_pred_ts,
            q_d_ts=q_d_pred_ts,
            q_dd_ts=q_dd_pred_ts,
            q_dd_ode_ts=q_dd_ode_pred_ts,  # acceleration in latent space computed using the ODE
            rendering_ts=rendering_pred_ts,
            rendering_d_ts=rendering_d_pred_ts,
            rendering_dd_ts=rendering_dd_pred_ts,
        )

        if ae_type == "beta_vae":
            # reshape to batch_dim x time_dim x n_q
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
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates
        preds = forward_fn(batch, nn_params, rng=rng, training=training)

        # supervised MSE loss on the reconstructed image of the static predictions
        mse_rec = jnp.mean(jnp.square(preds["rendering_ts"] - batch["rendering_ts"]))

        # compute the SINDy q_dd/z_dd loss
        mse_sindy_q_dd = jnp.mean(jnp.square(preds["q_dd_ode_ts"] - preds["q_dd_ts"]))

        # compute the SINDy rendering_dd loss
        mse_sindy_rendering_dd = jnp.mean(
            jnp.square(preds["rendering_dd_ts"] - batch["rendering_dd_ts"])
        )

        # total loss
        loss = (
            loss_weights["mse_rec"] * mse_rec
            + loss_weights["mse_sindy_q_dd"] * mse_sindy_q_dd
            + loss_weights["mse_sindy_rendering_dd"] * mse_sindy_rendering_dd
        )

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
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        batch_loss_dict = {
            "mse_rec": jnp.mean(
                jnp.square(preds["rendering_ts"] - batch["rendering_ts"])
            ),
        }

        # compute the configuration error
        error_q_ts = preds["q_ts"] - batch["x_ts"][..., :n_q]
        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q_ts = normalize_joint_angles(error_q_ts)

        # compute the configuration velocity error
        error_q_d_ts = preds["q_d_ts"] - batch["x_ts"][..., n_q:]

        if normalize_configuration_loss:
            error_q_norm_ts = error_q_ts / (x0_max[:n_q] - x0_min[:n_q])
            error_q_d_norm_ts = error_q_d_ts / (x0_max[n_q:] - x0_min[n_q:])
            batch_loss_dict.update(
                {
                    "mse_q_norm": jnp.mean(jnp.square(error_q_norm_ts)),
                    "mse_q_d_norm": jnp.mean(jnp.square(error_q_d_norm_ts)),
                }
            )
        else:
            batch_loss_dict.update(
                {
                    "mse_q": jnp.mean(jnp.square(error_q_ts)),
                    "msq_q_d": jnp.mean(jnp.square(error_q_d_ts)),
                }
            )

        return batch_loss_dict

    task_callables = TaskCallables(
        system_type, assemble_input, forward_fn, loss_fn, compute_metrics
    )

    @dataclass  # <-- required for JAX transformations
    class MetricsCollection(clu_metrics.Collection):
        loss: clu_metrics.Average.from_output("loss")
        lr: clu_metrics.LastValue.from_output("lr")
        rmse_rec: RootAverage.from_output("mse_rec")

        if normalize_configuration_loss is True:
            rmse_q_norm: RootAverage.from_output("mse_q_norm")
            rmse_q_d_norm: RootAverage.from_output("mse_q_d_norm")
        else:
            rmse_q: RootAverage.from_output("mse_q")
            rmse_q_d: RootAverage.from_output("msq_q_d")

    metrics_collection_cls = MetricsCollection
    return task_callables, metrics_collection_cls
