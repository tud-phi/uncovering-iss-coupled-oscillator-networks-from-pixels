from flax import linen as nn  # Linen API
from jax import Array, random
import jax.numpy as jnp
import math
from typing import Callable, Optional, Sequence, Tuple, Type

from .simple_cnn import Decoder


class Encoder(nn.Module):
    """A simple CNN encoder."""

    latent_dim: int
    strides: Tuple[int, int] = (1, 1)
    nonlinearity: Callable = nn.leaky_relu
    norm_layer: Optional[Type] = None

    @nn.compact
    def __call__(self, x) -> Tuple[Array, Array]:
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=self.strides)(x)
        if self.norm_layer is not None:
            x = self.norm_layer()(x)
        x = self.nonlinearity(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=self.strides)(x)
        if self.norm_layer is not None:
            x = self.norm_layer()(x)
        x = self.nonlinearity(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=256)(x)
        if self.norm_layer is not None:
            x = self.norm_layer()(x)
        x = self.nonlinearity(x)

        mu = nn.Dense(features=self.latent_dim)(x)
        logvar = nn.Dense(features=self.latent_dim)(x)

        return mu, logvar


class VAE(nn.Module):
    """A Variational Autoencoder."""

    img_shape: Tuple[int, int, int]
    latent_dim: int
    strides: Tuple[int, int] = (1, 1)
    nonlinearity: Callable = nn.leaky_relu
    norm_layer: Optional[Type] = None
    clip_decoder_output: bool = True

    def setup(self):
        self.encoder = Encoder(
            latent_dim=self.latent_dim,
            strides=self.strides,
            nonlinearity=self.nonlinearity,
            norm_layer=self.norm_layer,
        )

        # the size of the image after the convolutional encoder, but before the dense layers
        # currently, we are using 2 convolutional layers
        downsampled_img_dim = (
            int(self.img_shape[0] / (self.strides[0] ** 2)),
            int(self.img_shape[1] / (self.strides[0] ** 2)),
            32,  # number of channels of the encoded image
        )
        # print("Computed downsampled image dimension:", downsampled_img_dim)

        self.decoder = Decoder(
            img_shape=self.img_shape,
            downsampled_img_dim=downsampled_img_dim,
            strides=self.strides,
            nonlinearity=self.nonlinearity,
            norm_layer=self.norm_layer,
            clip_output=self.clip_decoder_output,
        )

    def __call__(self, x):
        mu, logvar = self.encoder(x)
        x_rec = self.decoder(mu)

        return x_rec

    def encode(self, x: Array) -> Array:
        mu, logvar = self.encoder(x)
        return mu

    def encode_vae(self, x: Array) -> Tuple[Array, Array]:
        return self.encoder(x)

    def decode(self, z: Array) -> Array:
        return self.decoder(z)

    def generate(self, z: Array) -> Array:
        return nn.sigmoid(self.decoder(z))

    def reparameterize(self, rng: Array, mu: Array, logvar: Array) -> Array:
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape, logvar.dtype)
        return mu + eps * std
