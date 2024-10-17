import jax

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


ctrl_data_filepaths = [
    Path(
        "examples/control/data/mass_spring_friction_actuation/node-con-iae_P-satI-D/setpoint_sequence_controlled_rollout.npz"
    ),
    Path(
        "examples/control/data/mass_spring_friction_actuation/node-con-iae_D+FF/setpoint_sequence_controlled_rollout.npz"
    ),
    Path(
        "examples/control/data/mass_spring_friction_actuation/node-con-iae_P-satI-D+FF/setpoint_sequence_controlled_rollout.npz"
    ),
]
ctrl_names = ["P-satI-D", "D+FF", "P-satI-D+FF"]
assert len(ctrl_data_filepaths) == len(ctrl_names)

system_type = (
    "mass_spring_friction_actuation"  # "pcc_ns-2", "mass_spring_friction_actuation"
)

# define output directory
outputs_dir = Path(__file__).parent / "outputs" / system_type
outputs_dir.mkdir(parents=True, exist_ok=True)

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

    sim_ts_ls = []
    for ctrl_data_filepath in ctrl_data_filepaths:
        with open(ctrl_data_filepath):
            sim_ts = np.load(ctrl_data_filepath)
        sim_ts_ls.append(sim_ts)

    # plot the configuration trajectory
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(len(sim_ts_ls)):
        sim_ts = sim_ts_ls[i]
        ctrl_label = ctrl_names[i]
        for j in range(sim_ts["x_ts"].shape[1] // 2):
            jlabel = "" if sim_ts["x_ts"].shape[1] == 2 else r"_" + str(j)
            if i == 0:
                ax.plot(
                    sim_ts["ts"],
                    sim_ts["q_des_ts"][:, j],
                    color="black",
                    linestyle=":",
                    dashes=dots,
                    linewidth=linewidth_dotted,
                    label=r"Target $q^\mathrm{d}" + jlabel + "$",
                )
            ax.plot(
                sim_ts["ts"],
                sim_ts["x_ts"][:, j],
                color=colors[i],
                linewidth=linewidth_solid,
                label=ctrl_label + " " + r"$q" + jlabel + "$",
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
    plt.savefig(str(outputs_dir / f"setpoint_control_sequences_q.pdf"))
    plt.savefig(str(outputs_dir / f"setpoint_control_sequences_q.eps"))
    plt.show()

    # plot the latent trajectory
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(len(sim_ts_ls)):
        sim_ts = sim_ts_ls[i]
        ctrl_label = ctrl_names[i]
        for j in range(sim_ts["xi_ts"].shape[1] // 2):
            jlabel = "" if sim_ts["xi_ts"].shape[1] == 2 else r"_" + str(j)
            if i == 0:
                ax.plot(
                    sim_ts["ts"],
                    sim_ts["z_des_ts"][:, j],
                    color="black",
                    linestyle=":",
                    dashes=dots,
                    linewidth=linewidth_dotted,
                    label=r"Target $z^\mathrm{d}" + jlabel + "$",
                )
            ax.plot(
                sim_ts["ts"],
                sim_ts["xi_ts"][:, j],
                color=colors[i],
                linewidth=linewidth_solid,
                label=ctrl_label + " " + r"$z" + jlabel + "$",
            )
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"Latent variable $z$")
    plt.grid(True)
    plt.box(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(outputs_dir / f"setpoint_control_sequences_z.pdf"))
    plt.savefig(str(outputs_dir / f"setpoint_control_sequences_z.eps"))
    plt.show()

    # plot the control input trajectory
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(len(sim_ts_ls)):
        sim_ts = sim_ts_ls[i]
        ctrl_label = ctrl_names[i]
        for j in range(sim_ts["tau_ts"].shape[1]):
            jlabel = "" if sim_ts["tau_ts"].shape[1] == 1 else r"_" + str(j)
            ax.plot(
                sim_ts["ts"],
                sim_ts["tau_ts"][:, j],
                color=colors[i],
                linewidth=linewidth_solid,
                label=ctrl_label + " " + r"$u" + jlabel + "$",
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
    plt.savefig(str(outputs_dir / f"setpoint_control_sequences_u.pdf"))
    plt.savefig(str(outputs_dir / f"setpoint_control_sequences_u.eps"))
    plt.show()

    if "U_des_ts" in sim_ts:
        # plot the potential and kinetic energy trajectory
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for i in range(len(sim_ts_ls)):
            sim_ts = sim_ts_ls[i]
            ctrl_label = ctrl_names[i]
            if i == 0:
                # plot the desired potential energy
                ax.plot(
                    sim_ts["ts"],
                    sim_ts["U_des_ts"],
                    color="black",
                    linestyle=":",
                    dashes=dots,
                    linewidth=linewidth_dotted,
                    label=r"Target $\mathcal{U}^\mathrm{d}$",
                )
            # plot the potential energy
            ax.plot(
                sim_ts["ts"],
                sim_ts["U_ts"],
                color=colors[i],
                linewidth=linewidth_solid,
                label=ctrl_label + " " + r"$\mathcal{U}$",
            )
        plt.xlabel(r"Time $t$ [s]")
        plt.ylabel(r"Energy")
        plt.grid(True)
        plt.box(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            str(outputs_dir / f"setpoint_control_sequences_potential_energy.pdf")
        )
        plt.savefig(
            str(outputs_dir / f"setpoint_control_sequences_potential_energy.eps")
        )
        plt.show()
