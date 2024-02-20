import cv2  # importing cv2
from jax import Array
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as onp
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union


def animate_pred_vs_target_image_cv2(
    t_ts: onp.ndarray,
    img_pred_ts: onp.ndarray,
    img_target_ts: onp.ndarray,
    filepath: os.PathLike,
    step_skip: int = 1,
):
    """
    Creates an animation of the predicted vs. target images.
    Args:
        t_ts: time steps of the data
        img_pred_ts: predicted images of shape (num_time_steps, width, height, channels)
        img_target_ts: target images of shape (num_time_steps, width, height, channels)
        filepath: path to the output video
        step_skip: number of time steps to skip between frames
    """
    img_h, img_w, num_channels = img_pred_ts.shape[-3:]  # height, width, channels

    assert (
        img_pred_ts.shape[-1] == img_target_ts.shape[-1]
    ), "The height of the predicted and target images must be the same."

    # averaged time step
    dt = onp.mean(t_ts[1:] - t_ts[:-1])

    # create the video writer
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    if not type(filepath) is Path:
        filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print("Writing video to", filepath.resolve())
    video = cv2.VideoWriter(
        str(filepath),
        fourcc,
        1 / (step_skip * dt),  # fps
        (2 * img_w, img_h),
    )

    for time_idx, t in enumerate(range(0, t_ts.shape[0], step_skip)):
        # concatenate the image
        img = onp.concatenate((img_pred_ts[t], img_target_ts[t]), axis=1)

        # write the image to the video
        video.write(img)

    video.release()


def animate_pred_vs_target_image_pyplot(
    t_ts: Union[Array, onp.ndarray],
    img_pred_ts: Union[Array, onp.ndarray],
    img_target_ts: Union[Array, onp.ndarray],
    filepath: Optional[os.PathLike] = None,
    show: bool = False,
    step_skip: int = 1,
    bgr_to_rgb: bool = False,
):
    """
    Creates an animation of the predicted vs. target images using matplotlib.
    Args:
        t_ts: time steps of the data
        img_pred_ts: predicted images of shape (num_time_steps, width, height, channels)
        img_target_ts: target images of shape (num_time_steps, width, height, channels)
        filepath: path to the output video
        show: whether to show the animation
        step_skip: number of time steps to skip between frames
        bgr_to_rgb: whether to convert the images from BGR to RGB
    """
    sample_rate = 1 / (onp.mean(t_ts[1:] - t_ts[:-1]))
    # frames
    frames = onp.arange(0, t_ts.shape[0], step=step_skip)

    # create the figure
    fig, axes = plt.subplots(
        1, 2, num="Prediction vs. target images", figsize=(6, 4), dpi=200
    )

    if bgr_to_rgb:
        img_pred_ts = cv2.cvtColor(img_pred_ts, cv2.COLOR_BGR2RGB)
        img_target_ts = cv2.cvtColor(img_target_ts, cv2.COLOR_BGR2RGB)

    im_pred = axes[0].imshow(img_pred_ts[0])
    im_target = axes[1].imshow(img_target_ts[1])
    text_time = fig.text(
        x=0.5, y=0.1, s="", color="black", fontsize=11, ha="center", va="center"
    )

    axes[0].set_title("Prediction")
    axes[1].set_title("Reconstruction")
    for ax in axes.flatten():
        ax.set_axis_off()
        ax.grid(False)
    plt.tight_layout()

    pbar = tqdm(total=frames.shape[0])

    def animate(frame_idx):
        text_time.set_text(f"t = {t_ts[frame_idx]:.2f} s")
        im_pred.set_data(img_pred_ts[frame_idx])
        im_target.set_data(img_target_ts[frame_idx])
        pbar.update(1)
        return text_time, im_pred, im_target

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=step_skip * 1000 / sample_rate,
        blit=False,
    )

    if filepath is not None:
        if not type(filepath) is Path:
            filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        print("Writing video to", filepath.resolve())

        movie_writer = animation.FFMpegWriter(fps=sample_rate)
        ani.save(str(filepath))

    if show:
        plt.show()

    pbar.close()
