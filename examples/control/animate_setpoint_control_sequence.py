import jax

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.control.utils import compute_settling_time_on_setpoint_trajectory

seed = 0
system_type = "pcc_ns-2"  # "pcc_ns-2", "mass_spring_friction_actuation"
# set the dynamics_model_name
dynamics_model_name = (
    "node-con-iae"  # "node-con-iae", "node-mechanical-mlp"
)
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

    SPEEDUP = 1  # speedup factor for the animation
    SKIP_STEP = 3  # step skip for the animation

    figsize = (4.5, 3.0)
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

    t_ts = sim_ts["ts"]
    # frame rate
    frame_rate = SPEEDUP / SKIP_STEP * (1 / (t_ts[1:] - t_ts[:-1]).mean().item())
    print("Frame rate:", frame_rate)
    pbar = tqdm(total=t_ts.shape[0])

    def animate_configuration_trajectory():
        n_q = sim_ts["x_ts"].shape[1] // 2
        # plot the configuration trajectory
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
        q_lines = []
        q_des_lines = []
        for i in range(n_q):
            (line,) = ax.plot(
                [],
                [],
                color=colors[i],
                linestyle=":",
                dashes=dots,
                linewidth=linewidth_dotted,
                label=r"$q^\mathrm{d}_" + str(i) + "$",
            )
            q_des_lines.append(line)
            (line,) = ax.plot(
                [],
                [],
                color=colors[i],
                linewidth=linewidth_solid,
                label=r"$q_" + str(i) + "$",
            )
            q_lines.append(line)
        ax.set_xlim(t_ts[0], t_ts[-1])
        ax.set_ylim(
            np.min(sim_ts["x_ts"][:, :n_q]) - np.pi,
            np.max(sim_ts["x_ts"][:, :n_q]) + np.pi,
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

        def animate(time_idx):
            for _i, _line in enumerate(q_des_lines):
                _line.set_data(
                    sim_ts["ts"][:time_idx],
                    sim_ts["q_des_ts"][:time_idx, _i],
                )
            for _i, _line in enumerate(q_lines):
                _line.set_data(
                    sim_ts["ts"][:time_idx],
                    sim_ts["x_ts"][:time_idx, _i],
                )

            lines = q_des_lines + q_lines

            pbar.update(SKIP_STEP)
            return lines

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=np.arange(t_ts.shape[0], step=SKIP_STEP),
            interval=1000 / frame_rate,
            blit=True,
        )

        movie_writer = animation.FFMpegWriter(fps=frame_rate)
        movie_save_path = ckpt_dir / f"setpoint_control_sequence_q_{SPEEDUP:.0f}x.mp4"
        ani.save(
            str(movie_save_path),
            writer=movie_writer,
        )

        print(f"Saved movie to {movie_save_path}")

        plt.show()
        pbar.close()

    animate_configuration_trajectory()
