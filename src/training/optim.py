import optax
from typing import Callable


def create_learning_rate_fn(
    num_epochs: int,
    steps_per_epoch: int,
    base_lr: float,
    warmup_epochs: int = 0,
    cosine_decay_epochs: int = None,
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
        cosine_decay_epochs: Number of epochs for cosine decay. If None, will use num_epochs - warmup_epochs.
    Returns:
        learning_rate_fn: A function that takes the current step and returns the current learning rate.
            It has the signature learning_rate_fn(step: int) -> lr.
    """
    schedules = []
    boundaries = []
    epoch_idx = 0

    if warmup_epochs > 0:
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_lr,
            transition_steps=warmup_epochs * steps_per_epoch,
        )
        epoch_idx += warmup_epochs
        schedules.append(warmup_fn)
        boundaries.append(epoch_idx * steps_per_epoch)

    if cosine_decay_epochs is None:
        cosine_decay_epochs = max(num_epochs - warmup_epochs, 0)

    assert (
        warmup_epochs + cosine_decay_epochs <= num_epochs
    ), "The sum of warmup_epochs and cosine_decay_epochs should be less than or equal to num_epochs"
    if warmup_epochs + cosine_decay_epochs < num_epochs:
        # have a period of constant learning rate
        constant_lr_fn = optax.constant_schedule(base_lr)
        schedules.append(constant_lr_fn)
        epoch_idx += num_epochs - warmup_epochs - cosine_decay_epochs
        boundaries.append(epoch_idx * steps_per_epoch)

    cosine_decay_fn = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=cosine_decay_epochs * steps_per_epoch
    )
    schedules.append(cosine_decay_fn)

    learning_rate_fn = optax.join_schedules(
        schedules=schedules,
        boundaries=boundaries,
    )

    return learning_rate_fn
