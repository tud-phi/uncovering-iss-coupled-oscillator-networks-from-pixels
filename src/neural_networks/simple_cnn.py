from flax import linen as nn  # Linen API
from typing import Callable


class Encoder(nn.Module):
    """A simple CNN encoder."""
    latent_dim: int
    nonlinearity: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = self.nonlinearity(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = self.nonlinearity(x)

        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=256)(x)
        x = self.nonlinearity(x)
        x = nn.Dense(features=self.latent_dim)(x)

        return x


class Decoder(nn.Module):
    """A simple CNN decoder."""
    latent_dim: int
    nonlinearity: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.latent_dim)(x)
        x = self.nonlinearity(x)
        x = nn.Dense(features=256)(x)

        x = x.reshape((x.shape[0], 16, 16, 1))  # flatten

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = self.nonlinearity(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = self.nonlinearity(x)

        return x


class Autoencoder(nn.Module):
    """A simple CNN autoencoder."""
    latent_dim: int
    nonlinearity: Callable = nn.leaky_relu

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, nonlinearity=self.nonlinearity)
        self.decoder = Decoder(latent_dim=self.latent_dim, nonlinearity=self.nonlinearity)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
