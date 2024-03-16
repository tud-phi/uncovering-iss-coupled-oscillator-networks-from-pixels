import dill
import jax

jax.config.update("jax_platform_name", "cpu")  # set default device to 'cpu'
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path

"""
Short horizon dataset: 
node-mechanical-mlp: sweep_id = "2024-02-19_10-42-33"
node-mechanical-mlp-s: sweep_id = "2024-02-21_09-03-52"
node-w-con: sweep_id = "2024-02-19_00-38-28"

Long horizon dataset:
node-w-con: sweep_id = "2024-03-12_12-53-29"
node-con-iae: sweep_id = "2024-03-15_21-44-34"
"""
sweep_id = "2024-03-15_21-44-34"

# plotting settings
figsize = (8, 6)


def main():
    sweep_folder = Path("logs") / "pcc_ns-2_dynamics_autoencoder" / sweep_id
    with open(sweep_folder / "sweep_results.dill", "rb") as file:
        sweep_results = dill.load(file)

    train_results = sweep_results["train"]
    test_results = sweep_results["test"]

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
