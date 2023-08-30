from flax import struct
from flax.training import train_state
from jax import random
import jax_metrics as jm
from typing import Callable


@struct.dataclass
class TaskCallables:
    system_type: str
    assemble_input_fn: Callable
    forward_fn: Callable
    loss_fn: Callable
    compute_metrics_fn: Callable


class TrainState(train_state.TrainState):
    rngKeyArray
    metrics: jm.Metrics
