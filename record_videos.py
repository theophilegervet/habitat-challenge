import os
import shutil
import cv2
import glob
from natsort import natsorted


source_dir = "data/images/all_val"
# source_dir = "data/images/remove_fp1"
target_dir = "data/tv_videos"


if __name__ == "__main__":
    def record_video(episode_dir):
        episode_name = episode_dir.split("/")[-1]

        # potted plant failures
        # if episode_name not in ['TEEsavR23oF_3', '6s7QHgap2fW_35', '6s7QHgap2fW_17', 'TEEsavR23oF_38', 'ziup5kvtCCR_62', 'TEEsavR23oF_5', 'ziup5kvtCCR_26', 'ziup5kvtCCR_97', '6s7QHgap2fW_29', 'TEEsavR23oF_84', '6s7QHgap2fW_69', 'ziup5kvtCCR_0', '6s7QHgap2fW_6', 'TEEsavR23oF_17', 'cvZr5TUy5C5_72', 'TEEsavR23oF_89', 'TEEsavR23oF_36', 'TEEsavR23oF_61', 'cvZr5TUy5C5_4', 'TEEsavR23oF_64', 'cvZr5TUy5C5_65', 'TEEsavR23oF_86', '6s7QHgap2fW_47', 'TEEsavR23oF_6', 'cvZr5TUy5C5_95', 'ziup5kvtCCR_5', 'TEEsavR23oF_74', 'TEEsavR23oF_62', 'ziup5kvtCCR_94', 'TEEsavR23oF_26', 'ziup5kvtCCR_9', 'cvZr5TUy5C5_5', 'TEEsavR23oF_73', 'ziup5kvtCCR_98', 'ziup5kvtCCR_90', 'TEEsavR23oF_13', 'cvZr5TUy5C5_16', 'ziup5kvtCCR_68', '6s7QHgap2fW_78', 'TEEsavR23oF_72', '6s7QHgap2fW_22', 'TEEsavR23oF_97', 'cvZr5TUy5C5_25', 'ziup5kvtCCR_15', 'cvZr5TUy5C5_31', 'TEEsavR23oF_35', 'TEEsavR23oF_49', 'ziup5kvtCCR_84', 'TEEsavR23oF_21', 'cvZr5TUy5C5_30', '6s7QHgap2fW_57', '6s7QHgap2fW_92']:
        #     return
        # tv failures
        if episode_name not in ['5cdEh9F2hJL_73', 'Nfvxx8J5NCo_57', 'mv2HUxq3B53_60', 'p53SfW6mjZe_47', 'mv2HUxq3B53_7', '4ok3usBNeis_91', 'p53SfW6mjZe_20', 'p53SfW6mjZe_5', 'mv2HUxq3B53_31', 'Nfvxx8J5NCo_77', 'zt1RVoi7PcG_47', 'zt1RVoi7PcG_93', 'mv2HUxq3B53_3', '5cdEh9F2hJL_75', 'QaLdnwvtxbs_23', '5cdEh9F2hJL_62', 'Nfvxx8J5NCo_72', '4ok3usBNeis_24', 'p53SfW6mjZe_85', 'Nfvxx8J5NCo_15', '5cdEh9F2hJL_45', '6s7QHgap2fW_73', 'qyAac8rV8Zk_75', 'qyAac8rV8Zk_80', 'cvZr5TUy5C5_78', 'Nfvxx8J5NCo_29', 'cvZr5TUy5C5_77', 'zt1RVoi7PcG_101', 'Nfvxx8J5NCo_93', '5cdEh9F2hJL_63', 'mv2HUxq3B53_36', 'Nfvxx8J5NCo_20', 'p53SfW6mjZe_16', 'Nfvxx8J5NCo_7', 'Nfvxx8J5NCo_6', 'p53SfW6mjZe_79', '5cdEh9F2hJL_70', 'QaLdnwvtxbs_28', 'QaLdnwvtxbs_82', 'p53SfW6mjZe_15', '5cdEh9F2hJL_18', 'zt1RVoi7PcG_103', 'p53SfW6mjZe_31', 'q3zU7Yy5E5s_33', '5cdEh9F2hJL_81', '5cdEh9F2hJL_42', 'q3zU7Yy5E5s_25', '4ok3usBNeis_61', '5cdEh9F2hJL_0', 'cvZr5TUy5C5_51', 'qyAac8rV8Zk_41', 'Nfvxx8J5NCo_78', '5cdEh9F2hJL_87', 'q3zU7Yy5E5s_28', '5cdEh9F2hJL_72', 'mv2HUxq3B53_16', 'q3zU7Yy5E5s_51', 'qyAac8rV8Zk_46', 'Nfvxx8J5NCo_66', 'zt1RVoi7PcG_73', '5cdEh9F2hJL_24', 'p53SfW6mjZe_53', 'Nfvxx8J5NCo_49', '6s7QHgap2fW_81', '5cdEh9F2hJL_30', 'Nfvxx8J5NCo_43', '5cdEh9F2hJL_3', 'p53SfW6mjZe_73', 'q3zU7Yy5E5s_89', 'qyAac8rV8Zk_31', 'Nfvxx8J5NCo_98', 'p53SfW6mjZe_3', '5cdEh9F2hJL_28', 'q3zU7Yy5E5s_80', 'mv2HUxq3B53_46', '4ok3usBNeis_96', 'q3zU7Yy5E5s_4', 'Nfvxx8J5NCo_2', 'mv2HUxq3B53_71', '5cdEh9F2hJL_59', 'Nfvxx8J5NCo_83', 'q3zU7Yy5E5s_82', 'cvZr5TUy5C5_59', '5cdEh9F2hJL_61', 'p53SfW6mjZe_96', '4ok3usBNeis_33', 'p53SfW6mjZe_84', 'q3zU7Yy5E5s_43', 'p53SfW6mjZe_24', 'zt1RVoi7PcG_92', 'p53SfW6mjZe_44', '4ok3usBNeis_27', 'zt1RVoi7PcG_53', '6s7QHgap2fW_20', 'p53SfW6mjZe_67', '6s7QHgap2fW_21', 'p53SfW6mjZe_51', 'zt1RVoi7PcG_69', '6s7QHgap2fW_13', 'q3zU7Yy5E5s_47', 'p53SfW6mjZe_25', 'qyAac8rV8Zk_55', 'cvZr5TUy5C5_1', 'qyAac8rV8Zk_97', 'p53SfW6mjZe_40', 'zt1RVoi7PcG_44', 'qyAac8rV8Zk_83', 'p53SfW6mjZe_95', 'zt1RVoi7PcG_81', 'p53SfW6mjZe_66', 'q3zU7Yy5E5s_6', 'p53SfW6mjZe_64', '6s7QHgap2fW_55', '6s7QHgap2fW_68', 'p53SfW6mjZe_57']:
            return
        # couch close but not close enough
        # if episode_name not in ['TEEsavR23oF_2', 'XB4GS9ShBRE_50', 'XB4GS9ShBRE_98', 'TEEsavR23oF_47', 'TEEsavR23oF_41', 'XB4GS9ShBRE_45', 'TEEsavR23oF_66', 'mv2HUxq3B53_50', 'TEEsavR23oF_31', 'TEEsavR23oF_69', 'TEEsavR23oF_75', 'mv2HUxq3B53_1', 'TEEsavR23oF_1', 'XB4GS9ShBRE_22', 'cvZr5TUy5C5_90', 'TEEsavR23oF_54', 'XB4GS9ShBRE_30', 'XB4GS9ShBRE_44', 'XB4GS9ShBRE_54']:
        #     return
        # chair close but not close enough
        # if episode_name not in ['p53SfW6mjZe_89', 'wcojb4TFT35_54', 'wcojb4TFT35_57', 'q3zU7Yy5E5s_87', 'ziup5kvtCCR_11', 'q3zU7Yy5E5s_26', 'TEEsavR23oF_68', 'zt1RVoi7PcG_96', 'q3zU7Yy5E5s_97', 'zt1RVoi7PcG_77', 'wcojb4TFT35_49', 'wcojb4TFT35_80', 'XB4GS9ShBRE_74', 'bxsVRursffK_4', 'bxsVRursffK_51']:
        #     return
        # # toilet close but not close enough
        # if episode_name not in ['q3zU7Yy5E5s_8', 'q3zU7Yy5E5s_50', 'zt1RVoi7PcG_80', 'Dd4bFSTQ8gi_14', 'XB4GS9ShBRE_52', 'XB4GS9ShBRE_79', 'XB4GS9ShBRE_94']:
        #     return
        # # tv close but not close enough
        # if episode_name not in ['5cdEh9F2hJL_63', '5cdEh9F2hJL_81', '5cdEh9F2hJL_0', '5cdEh9F2hJL_87', '5cdEh9F2hJL_3', '5cdEh9F2hJL_28', 'q3zU7Yy5E5s_43', 'qyAac8rV8Zk_97']:
        #     return

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
