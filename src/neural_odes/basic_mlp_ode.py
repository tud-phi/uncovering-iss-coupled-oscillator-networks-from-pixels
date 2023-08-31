from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable


class MLPOde(nn.Module):
    """A simple MLP ODE."""

    state_dim: int
    num_layers: int = 5
    hidden_dim: int = 20
    nonlinearity: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for _ in range(self.num_layers - 1):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = self.nonlinearity(x)
        x = nn.Dense(features=self.state_dim)(x)

        return x
