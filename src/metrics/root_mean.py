import typing as tp

import jax
import jax.numpy as jnp

from jax_metrics import types
from jax_metrics.metrics.mean import Mean
from jax_metrics.metrics.reduce import Reduction

M = tp.TypeVar("M", bound="RootMean")


class RootMean(Mean):
    """
    Computes the (weighted) root mean of the given values.
    """

    def compute(self) -> jax.Array:
        return jnp.sqrt(self.total / self.count)
