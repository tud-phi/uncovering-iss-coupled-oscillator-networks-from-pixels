from flax.training import checkpoints
import numpy as onp
from pathlib import Path

from src.structs import TrainState
from src.visualization.img_animation import animate_pred_vs_target_image_pyplot

model_id = "single_pendulum_autoencoding/2023-03-18_09-22-22"
logdir = Path("logs").resolve() / model_id
video_path = Path("videos") / model_id / "pred_vs_target_animation.mp4"

if __name__ == "__main__":
    # load the saved model
    #  state = TrainState()
    state = None
    state = checkpoints.restore_checkpoint(logdir, target=state)
    print("state", state["params"].keys())

    t_ts = onp.arange(0, 1, 0.1)
    img_pred_ts = onp.zeros((t_ts.shape[0], 64, 64, 3), dtype=onp.uint8)
    img_target_ts = onp.zeros((t_ts.shape[0], 64, 64, 3), dtype=onp.uint8)

    animate_pred_vs_target_image_pyplot(
        t_ts, img_pred_ts, img_target_ts, filepath=video_path, show=True
    )
