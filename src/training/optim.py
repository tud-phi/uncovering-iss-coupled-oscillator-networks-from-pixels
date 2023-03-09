import optax
from typing import Callable


def create_learning_rate_fn(
    num_epochs: int, steps_per_epoch: int, base_lr: float, warmup_epochs: int = 0
) -> Callable:
    """
    Creates a learning rate schedule function. THe learning rate scheduler implements the following procedure:
        1. A linear increase of the learning rate for a specified number of warmup epochs up to the base lr
        2. A cosine decay of the learning rate throughout the remaining epochs
    Args:
        num_epochs: Number of epochs to train for.
        steps_per_epoch: Number of steps per epoch.
        base_lr: Base learning rate.
        warmup_epochs: Number of epochs for warmup.
    Returns:
        learning_rate_fn: A function that takes the current step and returns the current learning rate.
            It has the signature learning_rate_fn(step: int) -> lr.
    """
    # Create the learning rate function implementing the procedure documented in the docstring
    # Hint: use the following optax functions:
    # optax.linear_schedule, optax.cosine_decay_schedule, optax.join_schedules
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=cosine_epochs * steps_per_epoch
    )
    learning_rate_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )

    return learning_rate_fn
