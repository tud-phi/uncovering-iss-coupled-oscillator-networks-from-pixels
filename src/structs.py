from clu import metrics as clu_metrics
from flax import struct
from flax.training import train_state
from jax import Array, random
from typing import Callable


@struct.dataclass
class TaskCallables:
    system_type: str
    assemble_input_fn: Callable
    forward_fn: Callable
    loss_fn: Callable
    compute_metrics_fn: Callable


class TrainState(train_state.TrainState):
    rng: Array
    metrics: clu_metrics.Collection
