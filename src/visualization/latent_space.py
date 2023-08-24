from functools import partial
import jax
from jax import Array, debug, jit, random
import jax.numpy as jnp
from pathlib import Path
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.structs import TaskCallables, TrainState


def visualize_latent_space(
    eval_ds: tf.data.Dataset,
    state: TrainState,
    task_callables: TaskCallables,
):
    for batch_idx, batch in enumerate(eval_ds.as_numpy_iterator()):
        print(batch_idx)
