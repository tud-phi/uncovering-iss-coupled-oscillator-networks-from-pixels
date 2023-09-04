from clu.metrics import Average
import flax
import jax.numpy as jnp
from typing import Any


@flax.struct.dataclass
class RootAverage(Average):
    def compute(self) -> Any:
        return jnp.sqrt(self.total / self.count)
