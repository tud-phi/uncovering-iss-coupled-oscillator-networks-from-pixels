from flax import linen as nn  # Linen API
from typing import Callable, Tuple


class Encoder(nn.Module):
    """A simple CNN encoder."""

    img_shape: Tuple[int, int, int]
    latent_dim: int
    nonlinearity: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = self.nonlinearity(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = self.nonlinearity(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=256)(x)
        x = self.nonlinearity(x)
        x = nn.Dense(features=self.latent_dim)(x)

        return x


class Decoder(nn.Module):
    """A simple CNN decoder."""

    img_shape: Tuple[int, int, int]
    latent_dim: int
    nonlinearity: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x):
        x = self.nonlinearity(x)
        x = nn.Dense(features=256)(x)
        x = self.nonlinearity(x)
        # allow later reshaping to batch_dim x width x height x 32
        x = nn.Dense(features=self.img_shape[0] * self.img_shape[1] * 32)(x)

        x = x.reshape(
            (
                x.shape[0],  # batch size
                self.img_shape[0],  # width
                self.img_shape[1],  # height
                32,  # channels
            )
        )  # unflatten

        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = self.nonlinearity(x)
        x = nn.Conv(features=self.img_shape[-1], kernel_size=(3, 3))(x)

        # clip to [-1, 1]
        x = -1.0 + 2 * nn.sigmoid(x)

        return x


class Autoencoder(nn.Module):
    """A simple CNN autoencoder."""

    img_shape: Tuple[int, int, int]
    latent_dim: int
    nonlinearity: Callable = nn.leaky_relu

    def setup(self):
        self.encoder = Encoder(
            img_shape=self.img_shape, latent_dim=self.latent_dim, nonlinearity=self.nonlinearity
        )
        self.decoder = Decoder(
            img_shape=self.img_shape, latent_dim=self.latent_dim, nonlinearity=self.nonlinearity
        )

    def __call__(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        return x_rec

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
