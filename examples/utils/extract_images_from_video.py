import cv2
import numpy as np
from pathlib import Path

# path to the video
video_path = Path("/home/mstoelzle/Downloads/pendulum_friction/rollout_6_target.mp4")

# dt between saved image frames
save_img_dt = 2.75 / 5
print(f"Saving image every {save_img_dt} seconds")

if __name__ == "__main__":
    # read the video
    cap = cv2.VideoCapture(str(Path(video_path).expanduser()))
    # extract the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video frame rate: {fps}")

    # iterate over the frames
    time_last_saved = -np.inf
    while cap.isOpened():
        # current time in the video
        time_current = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ret, frame = cap.read()
        if not ret:
            break

        # calculate the time since the last saved frame
        time_since_last_saved = time_current - time_last_saved
        print(
            f"Current time: {time_current:.2f}, Time since last saved: {time_since_last_saved:.2f}"
        )
        if time_since_last_saved >= (save_img_dt - 1e-6):
            img_path = video_path.parent / f"{video_path.stem}_{time_current:.2f}.png"
            # save the frame
            print(f"Saving frame of t={time_current} to {img_path.resolve()}")
            cv2.imwrite(str(img_path), frame)
            time_last_saved = time_current
