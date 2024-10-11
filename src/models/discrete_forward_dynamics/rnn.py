__all__ = ["DiscreteRnnDynamics"]
from flax import linen as nn  # Linen API
from functools import partial
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
            tmp = nn.Dense(features=self.state_dim)(x)
            if self.input_dim > 0:
                tmp = tmp + nn.Dense(features=self.state_dim)(tau[: self.input_dim])
            x_next = nn.tanh(tmp)
        elif self.rnn_method == "gru":
            # GRU
            # https://en.wikipedia.org/wiki/Gated_recurrent_unit
            # https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.GRUCell.html
            x_next, _ = GRUCell(features=self.state_dim, input_dim=self.input_dim)(x, tau[: self.input_dim])
        else:
            raise NotImplementedError(f"RNN method {self.rnn_method} not implemented")

        # the state dim might be larger than the output dim
        # in which case we need to slice the output
        x_next = x_next[..., -self.output_dim :]

        return x_next


class GRUCell(nn.GRUCell):
    input_dim: int = 0

    @nn.compact
    def __call__(self, carry, inputs):
        """Gated recurrent unit (GRU) cell.

        Args:
        carry: the hidden state of the LSTM cell,
            initialized using `GRUCell.initialize_carry`.
        inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
        A tuple with the new carry and the output.
        """
        h = carry
        hidden_features = h.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(
            nn.Dense,
            features=hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init
        )
        dense_i = partial(
            nn.Dense,
            features=hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )

        r = dense_h(name='hr')(h)
        z = dense_h(name='hz')(h)
        if self.input_dim > 0:
            r = r + dense_i(name='ir')(inputs)
            z = z + dense_i(name='iz')(inputs)
        r = self.gate_fn(r)
        z = self.gate_fn(z)

        # add bias because the linear transformations aren't directly summed.
        n = r * dense_h(name='hn', use_bias=True)(h)
        if self.input_dim > 0:
            n = n + dense_i(name='in')(inputs)
        n = self.activation_fn(n)

        new_h = (1. - z) * n + z * h
        return new_h, new_h
