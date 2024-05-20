import dill
import jax

jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path

"""
# Constant strain
node-con-iae: sweep_id = "2024-05-19_17-07-08"
# Planar PCS with two segments
Short horizon dataset: 
node-mechanical-mlp: sweep_id = "2024-02-19_10-42-33"
node-mechanical-mlp-s: sweep_id = "2024-02-21_09-03-52"
node-w-con: sweep_id = "2024-02-19_00-38-28"
Long horizon dataset:
node-w-con: sweep_id = "2024-03-12_12-53-29"
node-con-iae: sweep_id = "2024-03-15_21-44-34"
node-con-iae-s: sweep_id = "2024-03-17_22-26-44"
dsim-con-iae-cfa: "2024-05-08_00-21-02"
# 2-body problem
node-mechanical-mlp: sweep_id = "2024-03-20_21-13-37"
"""
sweep_id = "2024-03-15_21-44-34"
system_type = "pcc_ns-2"  #  "cs", "pcc_ns-2" or "nb-2"
plot_sweep = True

# plotting settings
figsize = (8, 6)


def main():
    sweep_folder = Path("logs") / f"{system_type}_dynamics_autoencoder" / sweep_id
    with open(sweep_folder / "sweep_results.dill", "rb") as file:
        sweep_results = dill.load(file)

    print("sweep results keys", sweep_results.keys())

    n_z_range = jnp.unique(sweep_results["n_z"])
    num_n_z = n_z_range.shape[0]
    seed_range = jnp.unique(sweep_results["seed"])

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
        sweep_results_stats["num_trainable_params"] = {key: jnp.mean(value).astype(jnp.int32).item() for key, value in filtered_num_trainable_params.items()}
        print(f"Number of trainable parameters for n_z={n_z}: {sweep_results_stats['num_trainable_params']}")

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

            print(
                f"Test results for n_z={n_z} {key}: "
                f"{jnp.mean(filtered_test_results[key]).item():4f} \u00B1 {jnp.std(filtered_test_results[key]).item():4f}"
            )

        # put the mean and std of the results in the sweep_results_stats

    if plot_sweep:
        # Plot the RMSE of the reconstruction vs the number of latent variables
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            num="RMSE of reconstruction vs. number of latent variables",
        )
        ax.plot(
            sweep_results["n_z"], train_results["rmse_rec_static"], label="RMSE rec static"
        )
        ax.plot(
            sweep_results["n_z"], test_results["rmse_rec_dynamic"], label="RMSE rec dynamic"
        )
        ax.set_xlabel("$n_z$")
        ax.set_ylabel("RMSE of reconstruction")
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
        ax.plot(
            sweep_results["n_z"], train_results["ssim_rec_static"], label="SSIM rec static"
        )
        ax.plot(
            sweep_results["n_z"], test_results["ssim_rec_dynamic"], label="SSIM rec dynamic"
        )
        ax.set_xlabel("$n_z$")
        ax.set_ylabel("SSIM of reconstruction")
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
            sweep_results["n_z"],
            sweep_results["num_trainable_params"]["dynamics"],
            label="Trainable parameters of dynamics model",
        )
        ax.set_xlabel("$n_z$")
        ax.set_ylabel("Number of trainable parameters")
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
        ax.plot(
            sweep_results["num_trainable_params"]["dynamics"],
            train_results["rmse_rec_static"],
            label="RMSE rec static",
        )
        ax.plot(
            sweep_results["num_trainable_params"]["dynamics"],
            test_results["rmse_rec_dynamic"],
            label="RMSE rec dynamic",
        )
        ax.set_xlabel("Number of trainable parameters")
        ax.set_ylabel("RMSE of reconstruction")
        ax.legend()
        plt.grid(True)
        plt.box(True)
        plt.savefig(sweep_folder / "rmse_rec_vs_num_trainable_params.pdf")
        plt.show()


if __name__ == "__main__":
    main()
