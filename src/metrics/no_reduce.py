import jax
from jax import Array
import jax.numpy as jnp
import jax_metrics as jm
from jax_metrics.metrics.metric import Metric, RenameArguments
from simple_pytree import field, static_field
import typing as tp


class NoReduce(Metric):
    """Leave the step-time metric as it is and do not reduce it in any way over the steps of an epoch."""

    values: Array
    dtype: jnp.dtype = static_field()

    def __init__(
        self,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        """
        Creates a `NoReduce` instance.
        Arguments:
            dtype: (Optional) data type of the metric result. Defaults to `float32`.
        """
        self.dtype = dtype or jnp.float32
        self.__dict__.update(self._initial_values())

    def _initial_values(self) -> tp.Dict[str, tp.Any]:
        # initialize states
        values = jnp.array(0.0, dtype=self.dtype)

        return dict(values=values)

    def reset(self) -> jm.Metric:
        """
        Resets all of the metric state variables.
        Returns:
            An instance of `Reduce`.
        """
        return self.replace(**self._initial_values())

    def update(self, values: Array, **_) -> jm.Metric:
        """
        Updates the metric state variables with the given values.
        Arguments:
            values: The values to update the metric state variables with.
        Returns:
            An instance of `NoReduce`.
        """
        return self.replace(values=values)

    def compute(self) -> Array:
        return self.values

    def merge(self, other: jm.Metric) -> jm.Metric:
        return jax.tree_map(lambda x, y: x + y, self, other)

    def reduce(self) -> jm.Metric:
        return self

    def from_argument(self: jm.Metric, argument: str) -> RenameArguments[jm.Metric]:
        return self.rename_arguments(values=argument)
