import os
import shutil
import cv2
import glob
from natsort import natsorted


source_dir = "data/images/exp"
target_dir = "data/videos"


if __name__ == "__main__":
    def record_video(episode_dir):
        episode_name = episode_dir.split("/")[-1]
        print(f"Recording video {episode_name}")

        # Semantic map vis
        img_array = []
        for filename in natsorted(glob.glob(f"{episode_dir}/snapshot*.png")):
            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(f"{target_dir}/{episode_name}.avi",
                              cv2.VideoWriter_fourcc(*"DIVX"), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        # Planner vis
        img_array = []
        for filename in natsorted(glob.glob(f"{episode_dir}/planner_snapshot*.png")):
            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(f"{target_dir}/planner_{episode_name}.avi",
                              cv2.VideoWriter_fourcc(*"DIVX"), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)

    for episode_dir in glob.glob(f"{source_dir}/*"):
        record_video(episode_dir)
