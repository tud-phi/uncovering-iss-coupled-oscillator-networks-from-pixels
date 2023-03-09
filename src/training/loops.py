from flax.training.train_state import TrainState
from functools import partial
import jax
from jax import Array, jit, random
import jax.numpy as jnp
import numpy as onp
from typing import Callable, Dict, List, Tuple

from src.structs import TaskCallables


@partial(
    jit,
    static_argnums=(2, 3),
    static_argnames=("learning_rate_fn", "task_callables"),
)
def train_step(
    train_state: TrainState,
    batch: Dict[str, Array],
    task_callables: TaskCallables,
    learning_rate_fn: Callable,
) -> Tuple[Dict[str, TrainState], Dict[str, Array]]:
    """
    Trains the neural network for one step.
    Args:
        states: Dictionary containing the current states of the training of the two neural networks^
        batch: dictionary of batch data
        learning_rate_fn: A function that takes the current step and returns the current learning rate.
            It has the signature learning_rate_fn(step: int) -> lr.
    Returns:
        states: Dictionary of updated training states
            Entries of dictionary:
                - MassMatrixNN: TrainState of the mass matrix neural network
                - PotentialEnergyNN: TrainState of the potential energy neural network
        metrics: Dictionary of training metrics
    """
    loss_fn = partial(task_callables.loss_fn, batch)
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, preds), grad_nn_params = grad_fn(train_state.params)

    # optimize the neural network parameters with gradient descent
    train_state = train_state.apply_gradients(grads=grad_nn_params)

    # # compute metrics
    # metrics = compute_metrics(batch, preds)
    # metrics["loss"] = loss
    # # save the currently active learning rates to the `metrics` dictionary
    # metrics["lr_mass_matrix_nn"] = learning_rate_fn(states["MassMatrixNN"].step)
    # metrics["lr_potential_energy_nn"] = learning_rate_fn(
    #     states["PotentialEnergyNN"].step
    # )

    return train_state

