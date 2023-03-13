from flax import linen as nn  # Linen API
from typing import Callable, Tuple


class Encoder(nn.Module):
    """A simple CNN encoder."""
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
    latent_dim: int
    img_shape: Tuple[int, int, int] = (32, 32, 1)
    nonlinearity: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x):
        x = self.nonlinearity(x)
        x = nn.Dense(features=256)(x)
        x = self.nonlinearity(x)
        x = nn.Dense(features=32768)(x)

        x = x.reshape((
            x.shape[0],  # batch size
            self.img_shape[0],  # width
            self.img_shape[1],  # height
            32  # channels
        ))  # unflatten

        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = self.nonlinearity(x)
        x = nn.Conv(features=self.img_shape[-1], kernel_size=(3, 3))(x)

        return x


class Autoencoder(nn.Module):
    """A simple CNN autoencoder."""
    latent_dim: int
    nonlinearity: Callable = nn.leaky_relu

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, nonlinearity=self.nonlinearity)
        self.decoder = Decoder(latent_dim=self.latent_dim, nonlinearity=self.nonlinearity)

    def __call__(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        return x_rec

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
