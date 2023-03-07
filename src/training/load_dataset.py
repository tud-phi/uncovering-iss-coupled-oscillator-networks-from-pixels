from jax import random
import jax.numpy as jnp
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Tuple


def load_datasets(
    name: str,
    val_perc: int = 20,
    test_perc: int = 20,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the dataset and splits it into a training, validation and test dataset.
    Args:
        name: Name of the dataset. Can also be name/config_name:version.
            https://www.tensorflow.org/datasets/api_docs/python/tfds/load
            Example: "mechanical_system/single_pendulum"
        val_perc: Percentage of validation dataset with respect to the entire dataset size. Needs to be in interval [0, 100].
        test_perc: Percentage of single_pendulum dataset with respect to the entire dataset size. Needs to be in interval [0, 100].
    Returns:
        train_ds: trainings set.
        val_ds: validation set.
        test_ds: test set.
    """
    assert 0 <= val_perc <= 100, "Validation ratio needs to be in interval [0, 100]."
    assert 0 <= test_perc <= 100, "Test ratio needs to be in interval [0, 100]."
    assert val_perc + test_perc <= 100, "The sum of validation and test ratio need to be equal or smaller than 100."

    train_ds, val_ds, test_ds = tfds.load(
        name,
        data_dir=Path("data/tensorflow_datasets"),
        split=[
            f"train[:{100 - val_perc - test_perc}%]",  # use the first part for training
            f"train[{100 - val_perc - test_perc}%:{100 - test_perc}%]",  # use the second part for validation
            f"train[{100 - test_perc}%:]"  # use the third part for testing
        ]
    )

    return train_ds, val_ds, test_ds
