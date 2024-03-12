from flax import linen as nn  # Linen API
from jax import debug, lax, vmap
import jax.numpy as jnp
from typing import Any, Callable, Optional, Sequence, Tuple, Union


class DynamicsAutoencoder(nn.Module):
    autoencoder: nn.Module
    dynamics: nn.Module
    dynamics_type: str = "node"  # "node" or "discrete"
    num_past_timesteps: int = 2  # only used if dynamics_type == "discrete"

    def setup(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.autoencoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.autoencoder.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.autoencoder.decode(*args, **kwargs)

    def forward_all_layers(self, *args, **kwargs):
        z_bt = self.encode(*args, **kwargs)
        self.decode(z_bt)

        if self.dynamics_type == "node":
            x_bt = jnp.concatenate([z_bt, jnp.zeros_like(z_bt)], axis=-1)
            tau_bt = jnp.zeros_like(
                x_bt, shape=(z_bt.shape[0], self.dynamics.input_dim)
            )
            _ = vmap(
                self.dynamics.forward_all_layers,
            )(x_bt, tau_bt)
        elif self.dynamics_type == "discrete":
            z_ts = z_bt[: self.num_past_timesteps]
            tau = jnp.zeros_like(z_ts, shape=(self.dynamics.input_dim,))
            _ = self.dynamics.forward_all_layers(z_ts.flatten(), tau)

    def encode_vae(self, *args, **kwargs):
        return self.autoencoder.encode_vae(*args, **kwargs)

    def reparameterize(self, *args, **kwargs):
        return self.autoencoder.reparameterize(*args, **kwargs)

    def forward_dynamics(self, *args, **kwargs):
        return self.dynamics.forward_dynamics(*args, **kwargs)
