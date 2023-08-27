from flax import linen as nn  # Linen API
from jax import Array, random
import jax.numpy as jnp
import math
from typing import Callable, Sequence, Tuple


class Encoder(nn.Module):
    """A simple CNN encoder."""

    latent_dim: int
    strides: Tuple[int, int] = (1, 1)
    nonlinearity: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x) -> Tuple[Array, Array]:
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=self.strides)(x)
        x = self.nonlinearity(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=self.strides)(x)
        x = self.nonlinearity(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=256)(x)
        x = self.nonlinearity(x)

        mu = nn.Dense(features=self.latent_dim)(x)
        logvar = nn.Dense(features=self.latent_dim)(x)

        return mu, logvar


class Decoder(nn.Module):
    """A simple CNN decoder."""

    img_shape: Tuple[int, int, int] = (64, 64, 3)
    downsampled_img_dim: Sequence = (2, 2, 768)
    strides: Tuple[int, int] = (1, 1)
    nonlinearity: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x):
        x = self.nonlinearity(x)
        x = nn.Dense(features=256)(x)
        x = self.nonlinearity(x)

        x = nn.Dense(features=math.prod(self.downsampled_img_dim))(x)
        x = x.reshape(
            (x.shape[0], *self.downsampled_img_dim)  # batch size
        )  # unflatten

        x = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=self.strides)(x)
        x = self.nonlinearity(x)
        x = nn.ConvTranspose(
            features=self.img_shape[-1], kernel_size=(3, 3), strides=self.strides
        )(x)

        # clip to [-1, 1]
        x = -1.0 + 2 * nn.sigmoid(x)

        return x


class VAE(nn.Module):
    """A Variational Autoencoder."""

    img_shape: Tuple[int, int, int]
    latent_dim: int
    strides: Tuple[int, int] = (1, 1)
    nonlinearity: Callable = nn.leaky_relu

    def setup(self):
        self.encoder = Encoder(
            latent_dim=self.latent_dim,
            strides=self.strides,
            nonlinearity=self.nonlinearity,
        )

        # the size of the image after the convolutional encoder, but before the dense layers
        # currently, we are using 2 convolutional layers
        downsampled_img_dim = (
            int(self.img_shape[0] / (self.strides[0] ** 2)),
            int(self.img_shape[1] / (self.strides[0] ** 2)),
            32,  # number of channels of the encoded image
        )
        print("Computed downsampled image dimension:", downsampled_img_dim)

        self.decoder = Decoder(
            img_shape=self.img_shape,
            downsampled_img_dim=downsampled_img_dim,
            strides=self.strides,
            nonlinearity=self.nonlinearity,
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
        eps = random.normal(rng, logvar.shape)
        return mu + eps * std
