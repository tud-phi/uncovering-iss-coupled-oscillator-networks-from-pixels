import dill
from jax import random
import jax.numpy as jnp
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Optional, Tuple


def load_dataset(
    name: str,
    seed: int,
    batch_size: int,
    val_perc: int = 20,
    test_perc: int = 20,
    num_threads: Optional[int] = None,
    prefetch: int = 2,
    normalize: bool = True,
    grayscale: bool = False
) -> Tuple[Dict[str, tf.data.Dataset], tfds.core.DatasetInfo, Dict]:
    """
    Loads the dataset and splits it into a training, validation and test set.
    Args:
        name: Name of the dataset. Can also be name/config_name:version.
            https://www.tensorflow.org/datasets/api_docs/python/tfds/load
            Example: "pendulum/single_pendulum"
        seed: Seed for the shuffling of the training dataset.
        batch_size: Batch size of the dataset.
        val_perc: Percentage of validation dataset with respect to the entire dataset size. Needs to be in interval [0, 100].
        test_perc: Percentage of single_pendulum dataset with respect to the entire dataset size. Needs to be in interval [0, 100].
        num_threads: Number of threads to use for parallel processing.
        prefetch: Number of batches to prefetch.
        normalize: Whether to normalize the rendering image to [0, 1].
        grayscale: Whether to convert the rendering image to grayscale.
    Returns:
        datasets: A dictionary with the following keys:
            train: trainings set.
            val: validation set.
            test: test set.
        dataset_info: Object with information about the dataset.
    """
    assert 0 <= val_perc <= 100, "Validation ratio needs to be in interval [0, 100]."
    assert 0 <= test_perc <= 100, "Test ratio needs to be in interval [0, 100]."
    assert (
        val_perc + test_perc <= 100
    ), "The sum of validation and test ratio need to be equal or smaller than 100."

    datasets = {}
    # percentage of the dataset that is used for training
    train_perc = 100 - val_perc - test_perc
    (datasets["train"], datasets["val"], datasets["test"]), dataset_info = tfds.load(
        name,
        data_dir=Path("data/tensorflow_datasets"),
        split=[
            f"train[:{train_perc}%]",  # use the first part for training
            f"train[{train_perc}%:{train_perc + val_perc}%]",  # use the second part for validation
            f"train[{train_perc + val_perc}%:]",  # use the third part for testing
        ],
        with_info=True,
    )

    # load metadata
    metadata_path = Path(dataset_info.data_dir) / "metadata.pkl"
    print(f"Loading metadata from {metadata_path.resolve()}")
    with open(str(metadata_path), "rb") as f:
        metadata = dill.load(f)

    options = tf.data.Options()
    # activate deterministic behaviour for the dataset
    options.deterministic = True
    if num_threads is not None:
        options.threading.private_threadpool_size = num_threads
    else:
        # if we don't set the threading options, we use auto-tuning to find the configuration (also prefetch)
        # which is the best for the current system
        options.autotune.enabled = True

    for split_name in datasets.keys():
        ds = datasets[split_name]

        # apply options to dataset
        ds = ds.with_options(options)

        if grayscale:
            # convert rendering image to grayscale
            ds = ds.map(
                lambda sample: sample
                | {
                    "rendering_ts": tf.image.rgb_to_grayscale(sample["rendering_ts"]),
                    "rendering_d_ts": tf.image.rgb_to_grayscale(sample["rendering_d_ts"]),
                    "rendering_dd_ts": tf.image.rgb_to_grayscale(sample["rendering_dd_ts"]),
                }
            )

        if normalize:
            # normalize rendering image to [0, 1]
            ds = ds.map(
                lambda sample: sample
                | {
                    "rendering_ts": tf.cast(sample["rendering_ts"], tf.float32) / 128.0
                    - 1.0,
                    "rendering_d_ts": tf.cast(sample["rendering_d_ts"], tf.float32) / 128.0,
                    "rendering_dd_ts": tf.cast(sample["rendering_dd_ts"], tf.float32) / 128.0,
                }
            )

        # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
        datasets[split_name] = ds.batch(batch_size, drop_remainder=True).prefetch(
            prefetch
        )

    # randomly shuffle the training dataset
    datasets["train"] = datasets["train"].shuffle(
        buffer_size=len(datasets["train"]),
        seed=seed,
        reshuffle_each_iteration=True,
    )

    return datasets, dataset_info, metadata
