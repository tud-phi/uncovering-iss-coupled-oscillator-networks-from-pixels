import jax

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.control.utils import compute_settling_time_on_setpoint_trajectory

seed = 0
system_type = "pcc_ns-2"  # "pcc_ns-2", "mass_spring_friction_actuation"
# set the dynamics_model_name
dynamics_model_name = "node-con-iae"  # "node-con-iae", "node-mechanical-mlp"
if system_type == "pcc_ns-2":
    n_z = 2
elif system_type == "mass_spring_friction_actuation":
    n_z = 1
else:
    raise ValueError(f"Invalid system_type: {system_type}")

if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Romand"],
        }
    )

    figsize = (5.0, 3.0)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linewidth_dashed = 2.7
    linewidth_dotted = 3.0
    linewidth_solid = 2.0
    dots = (1.2, 0.8)
    dashes = (2.5, 1.2)

    match system_type:
        case "pcc_ns-2":
            match dynamics_model_name:
                case "node-con-iae":
                    experiment_id = f"2024-05-20_13-14-46/n_z_{n_z}_seed_{seed}"
                case "node-con-iae-s":
                    experiment_id = f"2024-03-17_22-26-44/n_z_{n_z}_seed_{seed}"
                case "node-mechanical-mlp":
                    experiment_id = f"2024-05-21_07-45-14/n_z_{n_z}_seed_{seed}"
                case _:
                    raise ValueError(
                        f"No experiment_id for dynamics_model_name={dynamics_model_name}"
                    )
        case "mass_spring_friction_actuation":
            match dynamics_model_name:
                case "node-con-iae":
                    experiment_id = f"2024-09-26_16-00-56/n_z_{n_z}_seed_{seed}"
                case "node-mechanical-mlp":
                    experiment_id = f"2024-09-26_05-16-30/n_z_{n_z}_seed_{seed}"
                case _:
                    raise ValueError(
                        f"No experiment_id for dynamics_model_name={dynamics_model_name}"
                    )
        case _:
            raise ValueError(f"Invalid system_type: {system_type}")

    ckpt_dir = (
        Path("logs").resolve() / f"{system_type}_dynamics_autoencoder" / experiment_id
    )

    with open(ckpt_dir / "setpoint_sequence_controlled_rollout.npz"):
        sim_ts = np.load(ckpt_dir / "setpoint_sequence_controlled_rollout.npz")

    # compute the mean setpoint error
    rmse_q = np.sqrt(np.mean((sim_ts["q_des_ts"] - sim_ts["x_ts"][:, :2]) ** 2))
    rmse_z = np.sqrt(np.mean((sim_ts["z_des_ts"] - sim_ts["xi_ts"][:, :2]) ** 2))
    print(f"RMSE in q: {rmse_q:.4f}")
    print(f"RMSE in z: {rmse_z:.4f} m")
    if system_type == "pcc_ns-2":
        # compute the settling time
        print("Computing the settling time for the configuration...")
        settling_time_q_mean, settling_time_q_stdev = (
            compute_settling_time_on_setpoint_trajectory(
                sim_ts["ts"], sim_ts["q_des_ts"], sim_ts["x_ts"][:, :2], threshold=0.20
            )
        )
        print("Computing the settling time for the latent...")
        settling_time_z_mean, settling_time_z_stdev = (
            compute_settling_time_on_setpoint_trajectory(
                sim_ts["ts"], sim_ts["z_des_ts"], sim_ts["xi_ts"][:, :2], threshold=0.20
            )
        )
        print(
            f"Settling time in q: {settling_time_q_mean:.3f} \u00b1 {settling_time_q_stdev:.3f} s"
        )
        print(
            f"Settling time in z: {settling_time_z_mean:.3f} \u00b1 {settling_time_z_stdev:.3f} s"
        )

    # plot the configuration trajectory
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(sim_ts["x_ts"].shape[1] // 2):
        ax.plot(
            sim_ts["ts"],
            sim_ts["q_des_ts"][:, i],
            color=colors[i],
            linestyle=":",
            dashes=dots,
            linewidth=linewidth_dotted,
            label=r"$q^\mathrm{d}_" + str(i) + "$",
        )
        ax.plot(
            sim_ts["ts"],
            sim_ts["x_ts"][:, i],
            color=colors[i],
            linewidth=linewidth_solid,
            label=r"$q_" + str(i) + "$",
        )
    plt.xlabel(r"Time $t$ [s]")
    if system_type == "pcc_ns-2":
        plt.ylabel(r"Configuration $q$ [rad/m]")
    elif system_type == "mass_spring_friction_actuation":
        plt.ylabel(r"Configuration $q$ [m]")
    else:
        plt.ylabel(r"Configuration")
    plt.grid(True)
    plt.box(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(ckpt_dir / f"setpoint_control_sequence_q.pdf"))
    plt.savefig(str(ckpt_dir / f"setpoint_control_sequence_q.eps"))
    plt.show()

    # plot the latent trajectory
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(sim_ts["xi_ts"].shape[1] // 2):
        ax.plot(
            sim_ts["ts"],
            sim_ts["z_des_ts"][:, i],
            color=colors[i],
            linestyle=":",
            dashes=dots,
            linewidth=linewidth_dotted,
            label=r"$z^\mathrm{d}_" + str(i) + "$",
        )
        ax.plot(
            sim_ts["ts"],
            sim_ts["xi_ts"][:, i],
            color=colors[i],
            linewidth=linewidth_solid,
            label=r"$z_" + str(i) + "$",
        )
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"Latent variable $z$")
    plt.grid(True)
    plt.box(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(ckpt_dir / f"setpoint_control_sequence_z.pdf"))
    plt.savefig(str(ckpt_dir / f"setpoint_control_sequence_z.eps"))
    plt.show()

    # plot the control input trajectory
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(sim_ts["tau_ts"].shape[1]):
        ax.plot(
            sim_ts["ts"],
            sim_ts["tau_ts"][:, i],
            color=colors[i],
            linewidth=linewidth_solid,
            label=r"$u_" + str(i) + "$",
        )
    plt.xlabel(r"Time $t$ [s]")
    if system_type == "pcc_ns-2":
        plt.ylabel(r"Control input $u$ [Nm]")
    elif system_type == "mass_spring_friction_actuation":
        plt.ylabel(r"Control input $u$ [N]")
    else:
        plt.ylabel(r"Configuration $u$")
    plt.grid(True)
    plt.box(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(ckpt_dir / f"setpoint_control_sequence_u.pdf"))
    plt.savefig(str(ckpt_dir / f"setpoint_control_sequence_u.eps"))
    plt.show()

    if "U_des_ts" in sim_ts:
        # plot the potential and kinetic energy trajectory
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # plot the desired potential energy
        ax.plot(
            sim_ts["ts"],
            sim_ts["U_des_ts"],
            color=colors[0],
            linestyle=":",
            dashes=dots,
            linewidth=linewidth_dotted,
            label=r"$\mathcal{U}^\mathrm{d}$",
        )
        # plot the potential energy
        ax.plot(
            sim_ts["ts"],
            sim_ts["U_ts"],
            color=colors[0],
            linewidth=linewidth_solid,
            label=r"$\mathcal{U}$",
        )
        plt.xlabel(r"Time $t$ [s]")
        plt.ylabel(r"Energy")
        plt.grid(True)
        plt.box(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(ckpt_dir / f"setpoint_control_sequence_U.pdf"))
        plt.savefig(str(ckpt_dir / f"setpoint_control_sequence_U.eps"))
        plt.show()
