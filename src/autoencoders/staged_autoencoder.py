from flax import linen as nn  # Linen API
import jax.numpy as jnp
from typing import Callable, Sequence, Tuple


class StagedAutoencoder(nn.Module):
    backbone: nn.Module
    config_dim: int  # dimensionality of the configuration space

    def setup(self):
        self.encoder_head = nn.Dense(features=self.config_dim)
        self.decoder_head = nn.Dense(features=self.backbone.latent_dim)

    def __call__(self, x, use_head: bool = True):
        x = self.encode(x, use_head=use_head)
        x = self.decode(x, use_head=use_head)
        return x

    def encode(self, x, use_head: bool = True):
        x = self.backbone.encode(x)
        if use_head:
            x = self.encoder_head(x)
        return x

    def decode(self, x, use_head: bool = True):
        if use_head:
            x = self.decoder_head(x)
        x = self.backbone.decode(x)
        return x
