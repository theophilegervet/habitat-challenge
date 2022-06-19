import os
import shutil
import cv2
import glob
from natsort import natsorted


source_dir = "data/images/all_val"
target_dir = "data/videos"


if __name__ == "__main__":
    def record_video(episode_dir):
        episode_name = episode_dir.split("/")[-1]

        if episode_name not in ['TEEsavR23oF_3', '6s7QHgap2fW_35', '6s7QHgap2fW_17', 'TEEsavR23oF_38', 'ziup5kvtCCR_62', 'TEEsavR23oF_5', 'ziup5kvtCCR_26', 'ziup5kvtCCR_97', '6s7QHgap2fW_29', 'TEEsavR23oF_84', '6s7QHgap2fW_69', 'ziup5kvtCCR_0', '6s7QHgap2fW_6', 'TEEsavR23oF_17', 'cvZr5TUy5C5_72', 'TEEsavR23oF_89', 'TEEsavR23oF_36', 'TEEsavR23oF_61', 'cvZr5TUy5C5_4', 'TEEsavR23oF_64', 'cvZr5TUy5C5_65', 'TEEsavR23oF_86', '6s7QHgap2fW_47', 'TEEsavR23oF_6', 'cvZr5TUy5C5_95', 'ziup5kvtCCR_5', 'TEEsavR23oF_74', 'TEEsavR23oF_62', 'ziup5kvtCCR_94', 'TEEsavR23oF_26', 'ziup5kvtCCR_9', 'cvZr5TUy5C5_5', 'TEEsavR23oF_73', 'ziup5kvtCCR_98', 'ziup5kvtCCR_90', 'TEEsavR23oF_13', 'cvZr5TUy5C5_16', 'ziup5kvtCCR_68', '6s7QHgap2fW_78', 'TEEsavR23oF_72', '6s7QHgap2fW_22', 'TEEsavR23oF_97', 'cvZr5TUy5C5_25', 'ziup5kvtCCR_15', 'cvZr5TUy5C5_31', 'TEEsavR23oF_35', 'TEEsavR23oF_49', 'ziup5kvtCCR_84', 'TEEsavR23oF_21', 'cvZr5TUy5C5_30', '6s7QHgap2fW_57', '6s7QHgap2fW_92']:
            return

        print(f"Recording video {episode_name}")

        # Semantic map vis
        img_array = []
        filenames = natsorted(glob.glob(f"{episode_dir}/snapshot*.png"))
        if len(filenames) == 0:
            return
        for filename in filenames:
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
