import functools
from matplotlib import pyplot as plt
from matplotlib import animation as plt_animation
from matplotlib import rc
import numpy as np
import os
from pathlib import Path
import requests
from subprocess import getstatusoutput
import tensorflow as tf

import jax

jax.config.update("jax_enable_x64", True)
tf.random.set_seed(seed=0)
rc('animation', html='jshtml')

from dm_hamiltonian_dynamics_suite import load_datasets
from dm_hamiltonian_dynamics_suite import datasets

# ["toy_physics/mass_spring", "toy_physics/mass_spring_friction", "toy_physics/mass_spring_colors",
# "toy_physics/mass_spring_colors_friction", "toy_physics/mass_spring_long_trajectory",
# "toy_physics/mass_spring_colors_long_trajectory", "toy_physics/pendulum", "toy_physics/pendulum_friction",
# "toy_physics/pendulum_colors", "toy_physics/pendulum_colors_friction", "toy_physics/pendulum_long_trajectory",
# "toy_physics/pendulum_colors_long_trajectory", "toy_physics/double_pendulum", "toy_physics/double_pendulum_friction",
# "toy_physics/double_pendulum_colors",
# "toy_physics/double_pendulum_colors_friction", "toy_physics/two_body", "toy_physics/two_body_colors",
# "multi_agent/matching_pennies", "multi_agent/matching_pennies_long_trajectory", "multi_agent/rock_paper_scissors",
# "multi_agent/rock_paper_scissors_long_trajectory", "mujoco_room/circle", "mujoco_room/spiral"]
dataset_name = "toy_physics/mass_spring_colors_friction"

DATASETS_URL = "gs://dm-hamiltonian-dynamics-suite"
DATASETS_FOLDER = (Path("data") / "tensorflow_datasets").resolve()
print("Datasets_folder", DATASETS_FOLDER)
DATASETS_FOLDER.mkdir(exist_ok=True)


def download_file(file_url, destination_file):
    print("Downloading", file_url, "to", destination_file)
    command = f"gsutil cp {file_url} {destination_file}"
    status_code, output = getstatusoutput(command)
    if status_code != 0:
        raise ValueError(output)


def download_dataset(dataset_name: str):
    """Downloads the provided dataset from the DM Hamiltonian Dataset Suite"""
    destination_folder = os.path.join(DATASETS_FOLDER, dataset_name)
    dataset_url = os.path.join(DATASETS_URL, dataset_name)
    os.makedirs(destination_folder, exist_ok=True)
    if "long_trajectory" in dataset_name:
        files = ("features.txt", "test.tfrecord")
    else:
        files = ("features.txt", "train.tfrecord", "test.tfrecord")
    for file_name in files:
        file_url = os.path.join(dataset_url, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        if os.path.exists(destination_file):
            print("File", file_url, "already present.")
            continue
        download_file(file_url, destination_file)


def unstack(value: np.ndarray, axis: int = 0):
    """Unstacks an array along an axis into a list"""
    split = np.split(value, value.shape[axis], axis=axis)
    return [np.squeeze(v, axis=axis) for v in split]


def make_batch_grid(
    batch: np.ndarray, grid_height: int, grid_width: int, with_padding: bool = True
):
    """Makes a single grid image from a batch of multiple images."""
    assert batch.ndim == 5
    assert grid_height * grid_width >= batch.shape[0]
    batch = batch[: grid_height * grid_width]
    batch = batch.reshape((grid_height, grid_width) + batch.shape[1:])
    if with_padding:
        batch = np.pad(
            batch,
            pad_width=[[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [0, 0]],
            mode="constant",
            constant_values=1.0,
        )
    batch = np.concatenate(unstack(batch), axis=-3)
    batch = np.concatenate(unstack(batch), axis=-2)
    if with_padding:
        batch = batch[:, 1:, 1:]
    return batch


def plot_animation_from_batch(
    batch: np.ndarray, grid_height, grid_width, with_padding=True, figsize=None
):
    """Plots an animation of the batch of sequences."""
    if figsize is None:
        figsize = (grid_width, grid_height)
    batch = make_batch_grid(batch, grid_height, grid_width, with_padding)
    batch = batch[:, ::-1]
    fig = plt.figure(figsize=figsize)
    plt.close()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    img = ax.imshow(batch[0])

    def frame_update(i):
        i = int(np.floor(i).astype("int64"))
        img.set_data(batch[i])
        return [img]

    anim = plt_animation.FuncAnimation(
        fig=fig,
        func=frame_update,
        frames=np.linspace(0.0, len(batch), len(batch) * 5 + 1)[:-1],
        save_count=len(batch),
        interval=10,
        blit=True,
    )
    return anim


def plot_sequence_from_batch(
    batch: np.ndarray, t_start: int = 0, with_padding: bool = True, fontsize: int = 20
):
    """Plots all of the sequences in the batch."""
    n, t, dx, dy = batch.shape[:-1]
    xticks = np.linspace(dx // 2, t * (dx + 1) - 1 - dx // 2, t)
    xtick_labels = np.arange(t) + t_start
    yticks = np.linspace(dy // 2, n * (dy + 1) - 1 - dy // 2, n)
    ytick_labels = np.arange(n)
    batch = batch.reshape((n * t, 1) + batch.shape[2:])
    batch = make_batch_grid(batch, n, t, with_padding)[0]
    plt.imshow(batch.squeeze())
    plt.xticks(ticks=xticks, labels=xtick_labels, fontsize=fontsize)
    plt.yticks(ticks=yticks, labels=ytick_labels, fontsize=fontsize)


def visualize_dataset(
    dataset_path: str,
    sequence_lengths: int = 60,  # 60
    grid_height: int = 2,
    grid_width: int = 5,
):
    """Visualizes a dataset loaded from the path provided."""
    batch_size = grid_height * grid_width
    ds = load_datasets.load_dataset(
        path=dataset_path,
        tfrecord_prefix="train",
        sub_sample_length=sequence_lengths,
        per_device_batch_size=batch_size,
        num_epochs=None,
        drop_remainder=True,
        threads=None,
        prefetch=False,
        shuffle=False,
        cache=False
    )

    print("Loaded dataset:\n", ds)
    # count_ds = ds.batch(1)
    # c = 0
    # for batch in ds.as_numpy_iterator():
    #     c += 1
    # print(f"Counted {c} samples")

    sample = next(iter(ds))
    batch_x = sample["x"].numpy()
    batch_image = sample["image"].numpy()
    # Plot real system dimensions
    plt.figure(figsize=(24, 8))
    for i in range(batch_x.shape[-1]):
        plt.subplot(1, batch_x.shape[-1], i + 1)
        plt.title(f"Samples from dimension {i + 1}")
        plt.plot(batch_x[:, :, i].T)
    plt.show()
    # Plot a sequence of 50 images
    plt.figure(figsize=(30, 10))
    plt.title("Samples from 50 steps sub sequences.")
    plot_sequence_from_batch(batch_image[:, :50])
    plt.show()
    # Plot animation
    # return plot_animation_from_batch(batch_image, grid_height, grid_width)


def main():
    # Download the dataset
    download_dataset(dataset_name)

    # Visualize the dataset
    visualize_dataset(str(DATASETS_FOLDER / dataset_name))


if __name__ == "__main__":
    main()
