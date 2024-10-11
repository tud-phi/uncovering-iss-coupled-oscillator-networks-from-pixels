import dill
from jax import Array
from jax import random
import jax.numpy as jnp
import numpy as onp
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
import types
from typing import Dict, Optional, Tuple

from src.structs import TaskCallables


def load_dataset(
    name: str,
    seed: int,
    batch_size: int,
    num_epochs: int = 1,
    val_perc: int = 20,
    test_perc: int = 20,
    num_threads: Optional[int] = None,
    prefetch: Optional[int] = 2,
    normalize: bool = True,
    grayscale: bool = False,
    dataset_type: str = "jsrm",
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
        normalize: Whether to normalize the rendering image to [-1, 1].
        grayscale: Whether to convert the rendering image to grayscale.
        dataset_type: Type of the dataset. Can be "jsrm", or "dm_hamiltonian_dynamics_suite"
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

    data_dir = Path("data/tensorflow_datasets")

    datasets = {}
    # percentage of the dataset that is used for training
    train_perc = 100 - val_perc - test_perc
    if dataset_type == "dm_hamiltonian_dynamics_suite":
        # example dataset name: "toy_physics/mass_spring_colors_friction
        dataset_path = (data_dir / f"{name}").resolve()
        print(f"Loading dataset from {dataset_path.resolve()}")

        # load metadata
        metadata_path = dataset_path / "metadata.pkl"
        metadata = {}
        if metadata_path.is_file():
            print(f"Loading metadata from {metadata_path.resolve()}")
            with open(str(metadata_path), "rb") as f:
                metadata = dill.load(f)

        datasets["train"], num_train_samples = load_dmhds_dataset(
            dataset_path,
            metadata=metadata,
            num_epochs=num_epochs,
            split_name="train",
        )
        datasets["val"], num_val_samples = load_dmhds_dataset(
            dataset_path,
            metadata=metadata,
            num_epochs=1,
            split_name="val",
        )
        datasets["test"], num_test_samples = load_dmhds_dataset(
            dataset_path,
            metadata=metadata,
            num_epochs=1,
            split_name="test",
        )
        split_sizes = {
            "train": num_train_samples,
            "val": num_val_samples,
            "test": num_test_samples,
        }
        dataset_info = {}
    else:
        (datasets["train"], datasets["val"], datasets["test"]), dataset_info = (
            tfds.load(
                name,
                data_dir=data_dir,
                split=[
                    f"train[:{train_perc}%]",  # use the first part for training
                    f"train[{train_perc}%:{train_perc + val_perc}%]",  # use the second part for validation
                    f"train[{train_perc + val_perc}%:]",  # use the third part for testing
                ],
                with_info=True,
            )
        )
        split_sizes = {
            "train": len(datasets["train"]),
            "val": len(datasets["val"]),
            "test": len(datasets["test"]),
        }

        # load metadata
        metadata_path = Path(dataset_info.data_dir) / "metadata.pkl"
        metadata = {}
        if metadata_path.is_file():
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
                }
            )

        datasets[split_name] = ds

    img_min_val, img_max_val = 0, 255
    if dataset_type in ["dm_hamiltonian_dynamics_suite", "reaction_diffusion"]:
        # determine the min and max values of the rendering image
        img_min_val, img_max_val = onp.inf, -onp.inf
        for sample_idx, sample in enumerate(datasets["test"].as_numpy_iterator()):
            img_min_val = min(onp.min(sample["rendering_ts"]), img_min_val)
            img_max_val = max(onp.max(sample["rendering_ts"]), img_max_val)
        print(f"Identified img_min_val: {img_min_val}, img_max_val: {img_max_val}")

    metadata["rendering"] = metadata.get("rendering", {})
    metadata["rendering"]["img_min_val"] = img_min_val
    metadata["rendering"]["img_max_val"] = img_max_val

    for split_name in datasets.keys():
        ds = datasets[split_name]

        if normalize:
            # normalize rendering image to [-1, 1]
            ds = ds.map(
                lambda sample: sample
                | {
                    "rendering_ts": (sample["rendering_ts"] - img_min_val)
                    / (img_max_val - img_min_val)
                    * 2.0
                    - 1.0,
                }
            )
            if "rendering_d_ts" in ds.element_spec:
                ds = ds.map(
                    lambda sample: sample
                    | {
                       "rendering_d_ts": sample["rendering_d_ts"] / (img_max_val - img_min_val) * 2.0,
                    }
                )


        # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
        datasets[split_name] = ds.batch(batch_size, drop_remainder=True)
        if prefetch is not None:
            datasets[split_name] = datasets[split_name].prefetch(prefetch)

    # randomly shuffle the training dataset
    datasets["train"] = datasets["train"].shuffle(
        buffer_size=split_sizes["train"],
        seed=seed,
        reshuffle_each_iteration=True,
    )

    # set the cardinality of the dataset
    if dataset_type == "dm_hamiltonian_dynamics_suite":
        for split_name, ds in datasets.items():
            cardinality_fn = lambda self: tf.constant(
                split_sizes[split_name] // batch_size
            )
            ds.cardinality = types.MethodType(cardinality_fn, ds)

    return datasets, dataset_info, metadata


def load_dummy_neural_network_input(
    ds: tf.data.Dataset,
    task_callables: TaskCallables,
) -> Array:
    # extract dummy batch from dataset
    nn_dummy_batch = next(ds.as_numpy_iterator())

    # assemble input for dummy batch
    nn_dummy_input = task_callables.assemble_input_fn(nn_dummy_batch)

    return nn_dummy_input


def load_dmhds_dataset(
    dataset_path: Path,
    metadata: Dict,
    num_epochs: int = 1,
    split_name: str = "train",
) -> Tuple[tf.data.Dataset, int]:
    from dm_hamiltonian_dynamics_suite.load_datasets import (
        load_dataset as dmhds_load_dataset,
    )

    match split_name:
        case "train":
            tf_record_prefix = "train"
        case "val" | "test":
            tf_record_prefix = "test"
        case _:
            raise ValueError(f"Unknown split: {split_name}")

    ds = dmhds_load_dataset(
        path=str(dataset_path),
        tfrecord_prefix=tf_record_prefix,
        sub_sample_length=None,
        threads=None,
        prefetch=False,
        shuffle=False,
        cache=False,
    )

    split_sizes = metadata["split_sizes"]
    num_val_samples = split_sizes["test"] // 2
    num_test_samples = split_sizes["test"] - num_val_samples
    match split_name:
        case "train":
            num_samples = split_sizes["train"]
        case "val":
            num_samples = num_val_samples
            ds = ds.take(num_val_samples)
        case "test":
            num_samples = num_test_samples
            ds = ds.skip(num_val_samples)
        case _:
            raise ValueError(f"Unknown split: {split_name}")

    # repeat dataset for multiple epochs
    ds = ds.repeat(num_epochs)

    ts = metadata["ts"]

    @tf.function
    def rename_keys_fn(x):
        if x.get("other", {}).get("tau", None) is not None:
            tau = tf.cast(x["other"]["tau"], tf.float32)
        else:
            tau = tf.zeros((x["x"].shape[-1],), dtype=tf.float32)
        y = dict(
            t_ts=ts,
            rendering_ts=x["image"],
            x_ts=x["x"],
            x_d_ts=x["dx_dt"],
            tau=tau,
        )
        return y

    ds = ds.map(rename_keys_fn)

    return ds, num_samples
