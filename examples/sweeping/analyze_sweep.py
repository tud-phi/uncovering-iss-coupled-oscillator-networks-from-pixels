import dill
import jax

jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Dict, List, Tuple

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)


"""
# Constant strain
node-con-iae: sweep_id = "2024-05-19_17-07-08"
node-con-iae-s: sweep_id = "2024-05-19_17-09-42"
dsim-con-iae-cfa: sweep_id = "2024-05-19_17-08-05"
node-general-mlp: sweep_id = "2024-05-19_17-08-51"
# Planar PCS with two segments
Short horizon dataset (one seed): 
node-mechanical-mlp: sweep_id = "2024-02-19_10-42-33"
node-mechanical-mlp-s: sweep_id = "2024-02-21_09-03-52"
node-w-con: sweep_id = "2024-02-19_00-38-28"
Long horizon dataset (one seed):
node-w-con: sweep_id = "2024-03-12_12-53-29"
node-con-iae: sweep_id = "2024-03-15_21-44-34"
node-con-iae-s: sweep_id = "2024-03-17_22-26-44"
Long horizon dataset (three seeds):
dsim-con-iae-cfa: "2024-05-08_00-21-02"
node-con-iae-s: "2024-05-15_23-41-01"
# 2-body problem
node-mechanical-mlp: sweep_id = "2024-03-20_21-13-37"
"""

# sweep settings for planar PCS with two segments
sweep_ids = ["2024-05-15_23-41-01", "2024-05-08_00-21-02"]
system_types = ["pcc_ns-2", "pcc_ns-2"]  #  "cs", "pcc_ns-2" or "nb-2"
model_names = ["CON-S", "CFA-CON"]

# analysis settings
plot_sweep = True
verbose = False

# plotting settings
figsize = (4.0, 2.5)
lw = 2.3
elinewidth = 0.7 * lw
capsize = 2.5
capthick = 1.5
ecolor = None
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_sweep_results(sweep_id: str, system_type: str) -> Tuple[Dict, Path]:
    sweep_folder = Path("logs") / f"{system_type}_dynamics_autoencoder" / sweep_id
    with open(sweep_folder / "sweep_results.dill", "rb") as file:
        sweep_results = dill.load(file)

    return sweep_results, sweep_folder


def generate_sweep_stats(sweep_results, verbose: bool = False) -> Dict:
    n_z_range = jnp.unique(sweep_results["n_z"])
    num_n_z = n_z_range.shape[0]

    # remove entries with None values
    train_results = {}
    for key, value in sweep_results["train"].items():
        if value is not None:
            train_results[key] = jnp.array(value)
    test_results = {}
    for key, value in sweep_results["test"].items():
        if value is not None:
            test_results[key] = jnp.array(value)
    
    sweep_results_stats = {
        "n_z": n_z_range,
        "num_trainable_params": {key: jnp.zeros_like(value, shape=(num_n_z, )) for key, value in sweep_results["num_trainable_params"].items()},
        "train_mean": {key: jnp.zeros_like(value, shape=(num_n_z, )) for key, value in train_results.items()},
        "train_std": {key: jnp.zeros_like(value, shape=(num_n_z, )) for key, value in train_results.items()},
        "test_mean": {key: jnp.zeros_like(value, shape=(num_n_z, )) for key, value in test_results.items()},
        "test_std": {key: jnp.zeros_like(value, shape=(num_n_z, )) for key, value in test_results.items()},
    }
    for i, n_z in enumerate(n_z_range):
        # filter for current n_z
        selector = sweep_results["n_z"] == n_z

        filtered_seeds = sweep_results["seed"][selector]
        filtered_num_trainable_params = {key: value[selector] for key, value in sweep_results["num_trainable_params"].items()}
        for key, value in filtered_num_trainable_params.items():
            sweep_results_stats["num_trainable_params"][key] = sweep_results_stats["num_trainable_params"][key].at[i].set(jnp.mean(value).astype(jnp.int32).item())

        if verbose:
            print(f"Number of trainable parameters for n_z={n_z}: {filtered_num_trainable_params}")

        filtered_train_results = {}
        for key, value in  sweep_results["train"].items():
            if value is not None:
                filtered_train_results[key] = value[selector]
                sweep_results_stats["train_mean"][key] = sweep_results_stats["train_mean"][key].at[i].set(jnp.mean(filtered_train_results[key]))
                sweep_results_stats["train_std"][key] = sweep_results_stats["train_std"][key].at[i].set(jnp.std(filtered_train_results[key]))

        filtered_test_results = {}
        for key, value in  sweep_results["test"].items():
            if value is not None:
                filtered_test_results[key] = value[selector]
                sweep_results_stats["test_mean"][key] = sweep_results_stats["test_mean"][key].at[i].set(jnp.mean(filtered_test_results[key]))
                sweep_results_stats["test_std"][key] = sweep_results_stats["test_std"][key].at[i].set(jnp.std(filtered_test_results[key]))

            if verbose:
                print(
                    f"Test results for n_z={n_z} {key}: "
                    f"{jnp.mean(filtered_test_results[key]).item():4f} \u00B1 {jnp.std(filtered_test_results[key]).item():4f}"
                )

    return sweep_results_stats


def analyze_single_sweep(sweep_id: str, system_type: str, verbose: bool = True, plot_sweep: bool = True):
    sweep_results, sweep_folder = load_sweep_results(sweep_id, system_type)
    sweep_results_stats = generate_sweep_stats(sweep_results, verbose=verbose)

    if plot_sweep:
        # Plot the RMSE of the reconstruction vs the number of latent variables
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="RMSE of reconstruction vs. number of latent variables",
        )
        ax.errorbar(
            sweep_results_stats["n_z"],
            sweep_results_stats["test_mean"]["rmse_rec_static"],
            yerr=sweep_results_stats["test_std"]["rmse_rec_static"],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
            label="RMSE rec static"
        )
        ax.errorbar(
            sweep_results_stats["n_z"], 
            sweep_results_stats["test_mean"]["rmse_rec_dynamic"], 
            yerr=sweep_results_stats["test_std"]["rmse_rec_dynamic"],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
            label="RMSE rec dynamic"
        )
        ax.set_xlabel("$n_z$")
        ax.set_ylabel("RMSE")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(sweep_folder / "rmse_rec_vs_n_z.pdf")
        plt.show()

        # plot the SSIM of the reconstruction vs the number of latent variables
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="SSIM of reconstruction vs. number of latent variables",
        )
        ax.errorbar(
            sweep_results_stats["n_z"],
            sweep_results_stats["test_mean"]["ssim_rec_static"],
            yerr=sweep_results_stats["test_std"]["ssim_rec_static"],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
            label="SSIM rec static"
        )
        ax.errorbar(
            sweep_results_stats["n_z"], 
            sweep_results_stats["test_mean"]["ssim_rec_dynamic"], 
            yerr=sweep_results_stats["test_std"]["ssim_rec_dynamic"],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
            label="SSIM rec dynamic"
        )
        ax.set_xlabel("$n_z$")
        ax.set_ylabel("SSIM")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(sweep_folder / "ssim_rec_vs_n_z.pdf")
        plt.show()

        # plot number of trainable parameters vs number of latent variables
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="Number of trainable parameters vs. number of latent variables",
        )
        ax.plot(
            sweep_results_stats["n_z"],
            sweep_results_stats["num_trainable_params"]["dynamics"],
            label="Trainable parameters of dynamics model",
        )
        ax.set_xlabel("$n_z$")
        ax.set_ylabel("Model parameters")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(sweep_folder / "num_trainable_params_dynamics_vs_n_z.pdf")
        plt.show()

        # plot the RMSE of the reconstruction vs the number of trainable parameters
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="RMSE of reconstruction vs. number of trainable parameters",
        )
        ax.errorbar(
            sweep_results_stats["num_trainable_params"]["dynamics"],
            sweep_results_stats["test_mean"]["rmse_rec_static"],
            yerr=sweep_results_stats["test_std"]["rmse_rec_static"],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
            label="RMSE rec static",
        )
        ax.errorbar(
            sweep_results_stats["num_trainable_params"]["dynamics"],
            sweep_results_stats["test_mean"]["rmse_rec_dynamic"],
            yerr=sweep_results_stats["test_std"]["rmse_rec_dynamic"],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
            label="RMSE rec dynamic",
        )
        ax.set_xlabel("Model parameters")
        ax.set_ylabel("RMSE")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(sweep_folder / "rmse_rec_vs_num_trainable_params.pdf")
        plt.show()


def plot_model_comparison(sweep_ids: List[str], system_types: List[str], model_names: List[str], verbose: bool = False):
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # load sweep results and generate stats
    sweep_results_ms = []
    sweep_results_stats_ms = []
    for sweep_id, system_type, model_name in zip(sweep_ids, system_types, model_names):
        sweep_results, sweep_folder = load_sweep_results(sweep_id, system_type)
        sweep_results_stats = generate_sweep_stats(sweep_results, verbose=verbose)

        sweep_results_ms.append(sweep_results)
        sweep_results_stats_ms.append(sweep_results_stats)

    # plot the RMSE of the static reconstruction vs the number of latent variables
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        num=f"RMSE of static reconstruction vs. number of latent variables",
    )
    handles = []
    for model_idx, (sweep_results_stats, model_name) in enumerate(zip(sweep_results_stats_ms, model_names)):
        errorbar_container_rmse_static = ax.errorbar(
            sweep_results_stats["n_z"], 
            sweep_results_stats["test_mean"]["rmse_rec_static"], 
            yerr=sweep_results_stats["test_std"]["rmse_rec_static"],
            linewidth=lw,
            color=colors[model_idx],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
        )
        handles.append(errorbar_container_rmse_static.lines[0])
    ax.set_xlabel("$n_z$")
    ax.set_ylabel("RMSE")
    ax.legend(handles=handles, labels=model_names)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / f"sweep_rmse_rec_static_vs_n_z.pdf")
    plt.show()


    # plot the RMSE of the dynamic reconstruction vs the number of latent variables
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        num=f"RMSE of dynamic reconstruction vs. number of latent variables",
    )
    handles = []
    for model_idx, (sweep_results_stats, model_name) in enumerate(zip(sweep_results_stats_ms, model_names)):
        errorbar_container_rmse_dynamic = ax.errorbar(
            sweep_results_stats["n_z"], 
            sweep_results_stats["test_mean"]["rmse_rec_dynamic"], 
            yerr=sweep_results_stats["test_std"]["rmse_rec_dynamic"],
            linewidth=lw,
            color=colors[model_idx],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
        )
        handles.append(errorbar_container_rmse_dynamic.lines[0])
    ax.set_xlabel("$n_z$")
    ax.set_ylabel("RMSE")
    ax.legend(handles=handles, labels=model_names)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / f"sweep_rmse_rec_dynamic_vs_n_z.pdf")
    plt.show()

    # plot the RMSE of the static reconstruction vs the number of trainable parameters
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        num=f"RMSE of static reconstruction vs. number of trainable parameters",
    )
    handles = []
    for model_idx, (sweep_results_stats, model_name) in enumerate(zip(sweep_results_stats_ms, model_names)):
        errorbar_container_rmse_static = ax.errorbar(
            sweep_results_stats["num_trainable_params"]["dynamics"], 
            sweep_results_stats["test_mean"]["rmse_rec_static"], 
            yerr=sweep_results_stats["test_std"]["rmse_rec_static"],
            linewidth=lw,
            color=colors[model_idx],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
        )
        handles.append(errorbar_container_rmse_static.lines[0])
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("RMSE")
    ax.legend(handles=handles, labels=model_names)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / f"sweep_rmse_rec_static_vs_num_trainable_params.pdf")
    plt.show()

    # plot the RMSE of the dynamic reconstruction vs the number of trainable parameters
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        num=f"RMSE of dynamic reconstruction vs. number of trainable parameters",
    )
    handles = []
    for model_idx, (sweep_results_stats, model_name) in enumerate(zip(sweep_results_stats_ms, model_names)):
        errorbar_container_rmse_dynamic = ax.errorbar(
            sweep_results_stats["num_trainable_params"]["dynamics"], 
            sweep_results_stats["test_mean"]["rmse_rec_dynamic"], 
            yerr=sweep_results_stats["test_std"]["rmse_rec_dynamic"],
            linewidth=lw,
            color=colors[model_idx],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
        )
        handles.append(errorbar_container_rmse_dynamic.lines[0])
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("RMSE")
    ax.legend(handles=handles, labels=model_names)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / f"sweep_rmse_rec_dynamic_vs_num_trainable_params.pdf")
    plt.show()

    # plot the SSIM of the static reconstruction vs the number of latent variables
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        num=f"SSIM of static reconstruction vs. number of latent variables",
    )
    handles = []
    for model_idx, (sweep_results_stats, model_name) in enumerate(zip(sweep_results_stats_ms, model_names)):
        errorbar_container_ssim_static = ax.errorbar(
            sweep_results_stats["n_z"], 
            sweep_results_stats["test_mean"]["ssim_rec_static"], 
            yerr=sweep_results_stats["test_std"]["ssim_rec_static"],
            linewidth=lw,
            color=colors[model_idx],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
        )
        handles.append(errorbar_container_ssim_static.lines[0])
    ax.set_xlabel("$n_z$")
    ax.set_ylabel("SSIM")
    ax.legend(handles=handles, labels=model_names)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / f"sweep_ssim_rec_static_vs_n_z.pdf")
    plt.show()

    # plot the SSIM of the dynamic reconstruction vs the number of latent variables
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        num=f"SSIM of dynamic reconstruction vs. number of latent variables",
    )
    handles = []
    for model_idx, (sweep_results_stats, model_name) in enumerate(zip(sweep_results_stats_ms, model_names)):
        errorbar_container_ssim_dynamic = ax.errorbar(
            sweep_results_stats["n_z"], 
            sweep_results_stats["test_mean"]["ssim_rec_dynamic"], 
            yerr=sweep_results_stats["test_std"]["ssim_rec_dynamic"],
            linewidth=lw,
            color=colors[model_idx],
            elinewidth=elinewidth,
            ecolor=ecolor,
            capsize=capsize,
            capthick=capthick,
        )
        handles.append(errorbar_container_ssim_dynamic.lines[0])
    ax.set_xlabel("$n_z$")
    ax.set_ylabel("SSIM")
    ax.legend(handles=handles, labels=model_names)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / f"sweep_ssim_rec_dynamic_vs_n_z.pdf")
    plt.show()






def main():
    if len(sweep_ids) == 1:
        print(f"Analyzing sweep with ID: {sweep_ids[0]}")
        sweep_id = sweep_ids[0]
        system_type = system_types[0]
        analyze_single_sweep(sweep_id=sweep_id, system_type=system_type, verbose=verbose, plot_sweep=plot_sweep)
    elif len(sweep_ids) > 1:
        plot_model_comparison(sweep_ids, system_types, model_names, verbose=verbose)
    else:
        raise ValueError("No sweep IDs provided.")


if __name__ == "__main__":
    main()
