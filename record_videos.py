import os
import shutil
import cv2
import glob
from natsort import natsorted


source_dir = "data/images/june15night_50ep"
target_dir = "data/videos"


if __name__ == "__main__":
    def record_video(episode_dir):
        episode_name = episode_dir.split("/")[-1]
        print(f"Recording video {episode_name}")

        if episode_name not in ['QaLdnwvtxbs_78', 'mL8ThkuaVTM_41', 'svBbv1Pavdk_17', 'TEEsavR23oF_65', '6s7QHgap2fW_6', 'qyAac8rV8Zk_14', '5cdEh9F2hJL_81', 'DYehNKdT76V_37', 'q3zU7Yy5E5s_17', 'mL8ThkuaVTM_34', 'mL8ThkuaVTM_22', '6s7QHgap2fW_33', 'mv2HUxq3B53_50', '5cdEh9F2hJL_20', '4ok3usBNeis_48', 'qyAac8rV8Zk_47', 'wcojb4TFT35_78', 'DYehNKdT76V_80', 'Dd4bFSTQ8gi_7', 'svBbv1Pavdk_53', 'q3zU7Yy5E5s_27', 'QaLdnwvtxbs_51', 'mv2HUxq3B53_75', 'TEEsavR23oF_70', 'Dd4bFSTQ8gi_41', 'QaLdnwvtxbs_22', 'svBbv1Pavdk_66', 'zt1RVoi7PcG_48', 'DYehNKdT76V_86', 'Dd4bFSTQ8gi_14', 'svBbv1Pavdk_65', 'QaLdnwvtxbs_44', 'zt1RVoi7PcG_54', 'mL8ThkuaVTM_26', 'ziup5kvtCCR_98', '6s7QHgap2fW_28', 'zt1RVoi7PcG_91', 'svBbv1Pavdk_60', 'svBbv1Pavdk_30', '4ok3usBNeis_11', 'mL8ThkuaVTM_35', 'TEEsavR23oF_75', 'bxsVRursffK_24', 'XB4GS9ShBRE_65', 'Nfvxx8J5NCo_87', 'mv2HUxq3B53_45', 'Dd4bFSTQ8gi_19', 'ziup5kvtCCR_45', 'zt1RVoi7PcG_71', 'QaLdnwvtxbs_25', 'XB4GS9ShBRE_66', 'QaLdnwvtxbs_43', 'q3zU7Yy5E5s_40', 'ziup5kvtCCR_17', 'zt1RVoi7PcG_60', 'QaLdnwvtxbs_66', 'q3zU7Yy5E5s_43', 'mL8ThkuaVTM_7', 'svBbv1Pavdk_8', 'zt1RVoi7PcG_97', 'XB4GS9ShBRE_93', 'DYehNKdT76V_3', '4ok3usBNeis_1', 'p53SfW6mjZe_27', 'mv2HUxq3B53_76', 'DYehNKdT76V_58', '6s7QHgap2fW_85', 'q3zU7Yy5E5s_81', 'ziup5kvtCCR_64', 'bxsVRursffK_51', 'q3zU7Yy5E5s_5', 'zt1RVoi7PcG_113', 'bxsVRursffK_4', 'ziup5kvtCCR_90', 'XB4GS9ShBRE_32', 'wcojb4TFT35_48', 'zt1RVoi7PcG_27', 'XB4GS9ShBRE_81', 'mv2HUxq3B53_28', 'q3zU7Yy5E5s_68', 'cvZr5TUy5C5_50', 'DYehNKdT76V_48', 'q3zU7Yy5E5s_59', 'XB4GS9ShBRE_30', 'bxsVRursffK_66', 'XB4GS9ShBRE_94', 'XB4GS9ShBRE_98', 'q3zU7Yy5E5s_3', 'q3zU7Yy5E5s_84', 'qyAac8rV8Zk_65', '6s7QHgap2fW_9', '4ok3usBNeis_36', 'wcojb4TFT35_91', 'XB4GS9ShBRE_52', 'q3zU7Yy5E5s_69', 'q3zU7Yy5E5s_94', 'XB4GS9ShBRE_95', 'q3zU7Yy5E5s_77', 'XB4GS9ShBRE_79', 'DYehNKdT76V_96', 'XB4GS9ShBRE_12', 'XB4GS9ShBRE_45', '4ok3usBNeis_49', 'DYehNKdT76V_72', '4ok3usBNeis_2', 'q3zU7Yy5E5s_35', 'DYehNKdT76V_26', 'bxsVRursffK_78', 'qyAac8rV8Zk_33', 'XB4GS9ShBRE_15', 'DYehNKdT76V_43', 'q3zU7Yy5E5s_57', 'q3zU7Yy5E5s_22', 'p53SfW6mjZe_81', 'q3zU7Yy5E5s_26', 'XB4GS9ShBRE_16']:
            return

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
