from flax.core import FrozenDict
from functools import partial
from jax import Array, jit


@jit
def preprocess_batch(batch) -> Array:
    # batch of images
    img_bt = batch["rendering_ts"]

    # flatten to the shape batch_dim * time_dim x img_width x img_height x img_channels
    img_bt = img_bt.reshape((-1, *img_bt.shape[2:]))

    return img_bt


@partial(
    jit,
    static_argnums=(0,),
    static_argnames=("nn_model", ),
)
def model_forward(nn_model, nn_params: FrozenDict, batch) -> Array:
    img_bt = preprocess_batch(batch)

    # output will be of shape batch_dim * time_dim x latent_dim
    q_hat_bt = nn_model.apply({"params": nn_params}, img_bt)

    # reshape to batch_dim x time_dim x latent_dim
    q_hat_bt = q_hat_bt.reshape((batch["rendering_ts"].shape[0], -1, q_hat_bt.shape[-1]))

    return q_hat_bt
