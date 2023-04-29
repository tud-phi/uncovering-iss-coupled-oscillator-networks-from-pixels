from flax import struct
from flax.training import train_state
import jax_metrics as jm
from typing import Any, Callable


@struct.dataclass
class TaskCallables:
    assemble_input_fn: Callable
    forward_fn: Callable
    loss_fn: Callable
    compute_metrics_fn: Callable


class TrainState(train_state.TrainState):
    metrics: jm.Metrics
    batch_stats: Any
