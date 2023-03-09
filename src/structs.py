from flax import struct
from typing import Callable


@struct.dataclass
class TaskCallables:
    preprocess_batch_fn: Callable
    model_forward_fn: Callable
    loss_fn: Callable
