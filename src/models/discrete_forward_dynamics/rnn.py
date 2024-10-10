__all__ = ["DiscreteRnnDynamics"]
from flax import linen as nn  # Linen API
from jax import Array
import jax.numpy as jnp
from typing import Callable

from .discrete_forward_dynamics_base import DiscreteForwardDynamicsBase


class DiscreteRnnDynamics(DiscreteForwardDynamicsBase):
    """
    Implement a few standard RNNs in discrete time.
    """

    state_dim: int
    input_dim: int
    output_dim: int

    rnn_method: str = "elman"  # in ["elman", "gru"]

    @nn.compact
    def __call__(self, x: Array, tau: Array) -> Array:
        """
        Args:
            x: state of shape (state_dim, ). In RNN literature, this is often called the "hidden state" or "carry" h.
            tau: control input of shape (input_dim, ). In RNN literature, this is often called the "input x".
        Returns:
            x_next: state of shape (output_dim, ). In RNN literature, this is often called the next "hidden state" or "carry" h.
        """
        assert (
            self.output_dim <= self.state_dim
        ), "Output dim must be less than or equal to state dim"

        if self.rnn_method == "elman":
            # Elman RNN
            # https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks
            # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            x_next = nn.tanh(
                nn.Dense(features=self.state_dim)(x)
                + nn.Dense(features=self.state_dim)(tau[:self.input_dim])
            )
        elif self.rnn_method == "gru":
            # GRU
            # https://en.wikipedia.org/wiki/Gated_recurrent_unit
            # https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.GRUCell.html
            x_next, _ = nn.GRUCell(features=self.state_dim)(x, tau[: self.input_dim])
        else:
            raise NotImplementedError(f"RNN method {self.rnn_method} not implemented")

        # the state dim might be larger than the output dim
        # in which case we need to slice the output
        x_next = x_next[..., -self.output_dim :]

        return x_next
