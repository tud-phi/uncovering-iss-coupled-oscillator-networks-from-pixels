from functools import partial
import jax
from jax import Array, debug, jit, random
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

from src.structs import TaskCallables, TrainState


def visualize_mapping_from_configuration_to_latent_space(
    eval_ds: tf.data.Dataset,
    state: TrainState,
    task_callables: TaskCallables,
    rng: Optional[random.PRNGKey] = None,
):
    q_ss = None
    z_pred_ss = None
    num_batches = len(eval_ds)  # number of dataset samples
    for batch_idx, batch in (pbar := tqdm(enumerate(eval_ds.as_numpy_iterator()))):
        pbar.set_description(
            f"Plotting latent space: processing batch {batch_idx + 1} / {num_batches}"
        )
        preds = task_callables.forward_fn(batch, state.params, rng=rng)
        q_bt = batch["x_ts"][..., : batch["x_ts"].shape[-1] // 2]
        z_pred_bt = preds["q_ts"]

        if task_callables.system_type == "pendulum":
            # normalize configuration variables
            q_bt = normalize_joint_angles(q_bt)

        if batch_idx == 0:
            q_ss = jnp.zeros((num_batches,) + q_bt.shape)
            z_pred_ss = jnp.zeros((num_batches,) + z_pred_bt.shape)

        q_ss = q_ss.at[batch_idx].set(q_bt)
        z_pred_ss = z_pred_ss.at[batch_idx].set(z_pred_bt)

    if q_ss.shape[-1] > 1:
        warnings.warn(
            "Configuration-space has dimension > 1. Plotting only the first dimension."
        )

    q_ss = q_ss.reshape((-1, q_bt.shape[-1]))
    z_pred_ss = z_pred_ss.reshape((-1, z_pred_bt.shape[-1]))

    sort_indices = jnp.argsort(q_ss[:, 0])
    q_ss_sorted = q_ss[sort_indices]
    q_pred_ss_sorted = z_pred_ss[sort_indices]

    plt.figure()
    for latent_idx in range(q_pred_ss_sorted.shape[-1]):
        plt.plot(
            q_ss_sorted[:, 0],
            q_pred_ss_sorted[:, latent_idx],
            linestyle="None",
            marker=".",
        )
    plt.xlabel("q")
    plt.ylabel("z")
    plt.tight_layout()
    plt.show()
