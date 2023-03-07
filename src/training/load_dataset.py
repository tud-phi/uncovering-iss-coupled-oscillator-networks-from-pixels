from jax import random
import jax.numpy as jnp
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Tuple


def load_dataset(
    name: str,
    batch_size: int,
    val_perc: int = 20,
    test_perc: int = 20,
    normalize: bool = True,
    grayscale: bool = False,
) -> Dict[str, tf.data.Dataset]:
    """
    Loads the dataset and splits it into a training, validation and test set.
    Args:
        name: Name of the dataset. Can also be name/config_name:version.
            https://www.tensorflow.org/datasets/api_docs/python/tfds/load
            Example: "mechanical_system/single_pendulum"
        batch_size: Batch size of the dataset.
        val_perc: Percentage of validation dataset with respect to the entire dataset size. Needs to be in interval [0, 100].
        test_perc: Percentage of single_pendulum dataset with respect to the entire dataset size. Needs to be in interval [0, 100].
        normalize: Whether to normalize the rendering image to [0, 1].
        grayscale: Whether to convert the rendering image to grayscale.
    Returns:
        train_ds: trainings set.
        val_ds: validation set.
        test_ds: test set.
    """
    assert 0 <= val_perc <= 100, "Validation ratio needs to be in interval [0, 100]."
    assert 0 <= test_perc <= 100, "Test ratio needs to be in interval [0, 100]."
    assert val_perc + test_perc <= 100, "The sum of validation and test ratio need to be equal or smaller than 100."

    datasets = {}
    datasets["train"], datasets["val"], datasets["test"] = tfds.load(
        name,
        data_dir=Path("data/tensorflow_datasets"),
        split=[
            f"train[:{100 - val_perc - test_perc}%]",  # use the first part for training
            f"train[{100 - val_perc - test_perc}%:{100 - test_perc}%]",  # use the second part for validation
            f"train[{100 - test_perc}%:]"  # use the third part for testing
        ]
    )

    for split_name in datasets.keys():
        ds = datasets[split_name]

        if grayscale:
            # convert rendering image to grayscale
            ds = ds.map(lambda sample: sample | {
                "rendering_ts": tf.image.rgb_to_grayscale(sample["rendering_ts"]),
            })

        if normalize:
            # normalize rendering image to [0, 1]
            ds = ds.map(lambda sample: sample | {
                "rendering_ts": tf.cast(sample["rendering_ts"], tf.float32) / 255.0,
            })

        # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
        datasets[split_name] = ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return datasets
