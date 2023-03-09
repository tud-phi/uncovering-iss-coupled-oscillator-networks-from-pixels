from flax import struct
from typing import Callable


@struct.dataclass
class TaskCallables:
    assemble_input_fn: Callable
    predict_fn: Callable
    loss_fn: Callable
