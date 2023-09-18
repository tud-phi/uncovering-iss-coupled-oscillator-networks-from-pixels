from functools import partial
import jax
from jax import Array, debug, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple

from .utils import extract_states_from_dataset


def plot_basic_distribution(
    ds: tf.data.Dataset,
):
    x_ss, tau_ss = extract_states_from_dataset(ds)
    # number of configuration variables
    n_q = x_ss.shape[-1] // 2

    plt.figure(num="Histogram of configuration space")
    for q_idx in range(n_q):
        counts, bins = jnp.histogram(x_ss[:, q_idx], bins=50)
        plt.stairs(counts, bins, label=rf"$q_{q_idx}$")
    plt.xlabel("$q$")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # create a scatter plot of q vs q_d
    for q_idx in range(n_q):
        plt.figure(num=r"Scatter plot of $q$ vs. $q_d$")
        plt.scatter(x_ss[:, q_idx], x_ss[:, n_q + q_idx], label=rf"$q_{q_idx}$")
        plt.xlabel(r"$q$")
        plt.ylabel(r"$q_d$")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_acting_forces_distribution(
    ds: tf.data.Dataset,
    system_type: str,
    robot_params: Dict[str, Any],
    dynamical_matrices_fn: Callable,
):
    x_ss, tau_ss = extract_states_from_dataset(ds)
    # number of configuration variables
    n_q = x_ss.shape[-1] // 2

    def compute_acting_forces(x: Array):
        q, q_d = x[..., :n_q], x[..., n_q:]
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        return dict(
            tau_coriolis=C @ q_d,
            tau_g=G,
            tau_el=K,
            tau_d=D @ q_d,
        )

    tau_dict_ss = vmap(compute_acting_forces)(x_ss)

    plt.figure(num="Distribution of acting forces")
    sns.violinplot(x=jnp.linalg.norm(tau_dict_ss["tau_coriolis"], axis=-1), color="red", label=r"$\tau_{coriolis}$")
    sns.violinplot(x=jnp.linalg.norm(tau_dict_ss["tau_g"], axis=-1), color="blue", label=r"$\tau_{g}$")
    sns.violinplot(x=jnp.linalg.norm(tau_dict_ss["tau_el"], axis=-1), color="green", label=r"$\tau_{el}$")
    sns.violinplot(x=jnp.linalg.norm(tau_dict_ss["tau_d"], axis=-1), color="orange", label=r"$\tau_{d}$")
    plt.xlabel("force")
    plt.legend()
    plt.tight_layout()
    plt.show()
