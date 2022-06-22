import os
import shutil
import cv2
import glob
from natsort import natsorted


# source_dir = "data/images/all_val"
# source_dir = "data/images/remove_fp1"

# source_dir = "data/images/fix_depth_preprocessing_couch"
# target_dir = "data/new_successes_NEW"
# source_dir = "data/images/train_june19_14env_30ep_rollback3"
# target_dir = "data/new_successes_OLD"
# source_dir = "data/images/fix_depth_preprocessing_couch"
# target_dir = "data/new_failures_NEW"
source_dir = "data/images/train_june19_14env_30ep_rollback3"
target_dir = "data/new_failures_OLD"


if __name__ == "__main__":
    def record_video(episode_dir):
        episode_name = episode_dir.split("/")[-1]

        # potted plant failures
        # if episode_name not in ['TEEsavR23oF_3', '6s7QHgap2fW_35', '6s7QHgap2fW_17', 'TEEsavR23oF_38', 'ziup5kvtCCR_62', 'TEEsavR23oF_5', 'ziup5kvtCCR_26', 'ziup5kvtCCR_97', '6s7QHgap2fW_29', 'TEEsavR23oF_84', '6s7QHgap2fW_69', 'ziup5kvtCCR_0', '6s7QHgap2fW_6', 'TEEsavR23oF_17', 'cvZr5TUy5C5_72', 'TEEsavR23oF_89', 'TEEsavR23oF_36', 'TEEsavR23oF_61', 'cvZr5TUy5C5_4', 'TEEsavR23oF_64', 'cvZr5TUy5C5_65', 'TEEsavR23oF_86', '6s7QHgap2fW_47', 'TEEsavR23oF_6', 'cvZr5TUy5C5_95', 'ziup5kvtCCR_5', 'TEEsavR23oF_74', 'TEEsavR23oF_62', 'ziup5kvtCCR_94', 'TEEsavR23oF_26', 'ziup5kvtCCR_9', 'cvZr5TUy5C5_5', 'TEEsavR23oF_73', 'ziup5kvtCCR_98', 'ziup5kvtCCR_90', 'TEEsavR23oF_13', 'cvZr5TUy5C5_16', 'ziup5kvtCCR_68', '6s7QHgap2fW_78', 'TEEsavR23oF_72', '6s7QHgap2fW_22', 'TEEsavR23oF_97', 'cvZr5TUy5C5_25', 'ziup5kvtCCR_15', 'cvZr5TUy5C5_31', 'TEEsavR23oF_35', 'TEEsavR23oF_49', 'ziup5kvtCCR_84', 'TEEsavR23oF_21', 'cvZr5TUy5C5_30', '6s7QHgap2fW_57', '6s7QHgap2fW_92']:
        #     return
        # tv failures
        # if episode_name not in ['5cdEh9F2hJL_73', 'Nfvxx8J5NCo_57', 'mv2HUxq3B53_60', 'p53SfW6mjZe_47', 'mv2HUxq3B53_7', '4ok3usBNeis_91', 'p53SfW6mjZe_20', 'p53SfW6mjZe_5', 'mv2HUxq3B53_31', 'Nfvxx8J5NCo_77', 'zt1RVoi7PcG_47', 'zt1RVoi7PcG_93', 'mv2HUxq3B53_3', '5cdEh9F2hJL_75', 'QaLdnwvtxbs_23', '5cdEh9F2hJL_62', 'Nfvxx8J5NCo_72', '4ok3usBNeis_24', 'p53SfW6mjZe_85', 'Nfvxx8J5NCo_15', '5cdEh9F2hJL_45', '6s7QHgap2fW_73', 'qyAac8rV8Zk_75', 'qyAac8rV8Zk_80', 'cvZr5TUy5C5_78', 'Nfvxx8J5NCo_29', 'cvZr5TUy5C5_77', 'zt1RVoi7PcG_101', 'Nfvxx8J5NCo_93', '5cdEh9F2hJL_63', 'mv2HUxq3B53_36', 'Nfvxx8J5NCo_20', 'p53SfW6mjZe_16', 'Nfvxx8J5NCo_7', 'Nfvxx8J5NCo_6', 'p53SfW6mjZe_79', '5cdEh9F2hJL_70', 'QaLdnwvtxbs_28', 'QaLdnwvtxbs_82', 'p53SfW6mjZe_15', '5cdEh9F2hJL_18', 'zt1RVoi7PcG_103', 'p53SfW6mjZe_31', 'q3zU7Yy5E5s_33', '5cdEh9F2hJL_81', '5cdEh9F2hJL_42', 'q3zU7Yy5E5s_25', '4ok3usBNeis_61', '5cdEh9F2hJL_0', 'cvZr5TUy5C5_51', 'qyAac8rV8Zk_41', 'Nfvxx8J5NCo_78', '5cdEh9F2hJL_87', 'q3zU7Yy5E5s_28', '5cdEh9F2hJL_72', 'mv2HUxq3B53_16', 'q3zU7Yy5E5s_51', 'qyAac8rV8Zk_46', 'Nfvxx8J5NCo_66', 'zt1RVoi7PcG_73', '5cdEh9F2hJL_24', 'p53SfW6mjZe_53', 'Nfvxx8J5NCo_49', '6s7QHgap2fW_81', '5cdEh9F2hJL_30', 'Nfvxx8J5NCo_43', '5cdEh9F2hJL_3', 'p53SfW6mjZe_73', 'q3zU7Yy5E5s_89', 'qyAac8rV8Zk_31', 'Nfvxx8J5NCo_98', 'p53SfW6mjZe_3', '5cdEh9F2hJL_28', 'q3zU7Yy5E5s_80', 'mv2HUxq3B53_46', '4ok3usBNeis_96', 'q3zU7Yy5E5s_4', 'Nfvxx8J5NCo_2', 'mv2HUxq3B53_71', '5cdEh9F2hJL_59', 'Nfvxx8J5NCo_83', 'q3zU7Yy5E5s_82', 'cvZr5TUy5C5_59', '5cdEh9F2hJL_61', 'p53SfW6mjZe_96', '4ok3usBNeis_33', 'p53SfW6mjZe_84', 'q3zU7Yy5E5s_43', 'p53SfW6mjZe_24', 'zt1RVoi7PcG_92', 'p53SfW6mjZe_44', '4ok3usBNeis_27', 'zt1RVoi7PcG_53', '6s7QHgap2fW_20', 'p53SfW6mjZe_67', '6s7QHgap2fW_21', 'p53SfW6mjZe_51', 'zt1RVoi7PcG_69', '6s7QHgap2fW_13', 'q3zU7Yy5E5s_47', 'p53SfW6mjZe_25', 'qyAac8rV8Zk_55', 'cvZr5TUy5C5_1', 'qyAac8rV8Zk_97', 'p53SfW6mjZe_40', 'zt1RVoi7PcG_44', 'qyAac8rV8Zk_83', 'p53SfW6mjZe_95', 'zt1RVoi7PcG_81', 'p53SfW6mjZe_66', 'q3zU7Yy5E5s_6', 'p53SfW6mjZe_64', '6s7QHgap2fW_55', '6s7QHgap2fW_68', 'p53SfW6mjZe_57']:
        #     return
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
        # train potted plant failures
        # if episode_name not in ['W16Bm4ysK8v_33337', 'TSJmdttd2GV_26433', 'W16Bm4ysK8v_36745', 'HxmXPBbFCkH_44951', 'QN2dRqwd84J_47943', 'xAHnY3QzFUN_27726', '1S7LAXRdDqK_10977', '1S7LAXRdDqK_13561', 'xWvSkKiWQpC_26734', 'qk9eeNeR4vw_33831', 'xWvSkKiWQpC_20630', '8wJuSPJ9FXG_7354', 'CQWES1bawee_31181', 'XiJhRLvpKpX_32469', 'fxbzYAGkrtm_16240', 'ACZZiU6BXLz_1736', 'TSJmdttd2GV_28757', 'u9rPN5cHWBg_1505', 'TSJmdttd2GV_26776']:
        #     return
        # train potted plant successes
        # if episode_name not in ['TSJmdttd2GV_25799', 'g7hUFVNac26_31011', 'HxmXPBbFCkH_44557', 'gQ3xxshDiCz_19929', 'FRQ75PjD278_31155', 'VoVGtfYrpuQ_19032', 'VoVGtfYrpuQ_23930', 'QVAA6zecMHu_15405', 'u9rPN5cHWBg_4616', 'TSJmdttd2GV_21329', 'TSJmdttd2GV_26039']:
        #     return

        # new successes new keys
        # if episode_name not in ['8wJuSPJ9FXG_32258', '3XYAD64HpDr_17947', 'Jfyvj3xn2aJ_9082', 'Jfyvj3xn2aJ_9082', 'xAHnY3QzFUN_45570', '3XYAD64HpDr_22429', '3XYAD64HpDr_19062', 'xAHnY3QzFUN_26274', 'GGBvSFddQgs_18141', 'qz3829g1Lzf_14881', 'JptJPosx1Z6_43234', '3CBBjsNkhqW_11865', 'HxmXPBbFCkH_35025', 'GGBvSFddQgs_38636', 'GGBvSFddQgs_38636', 'GtM3JtRvvvR_33979', 'qz3829g1Lzf_6780', 'HxmXPBbFCkH_27298', 'HxmXPBbFCkH_3719', 'HxmXPBbFCkH_27968', 'gQ3xxshDiCz_21465', 'gQ3xxshDiCz_21465', 'GtM3JtRvvvR_31794', 'wsAYBFtQaL7_18495', 'wsAYBFtQaL7_18495', 'wsAYBFtQaL7_18495', 'U3oQjwTuMX8_22484', 'U3oQjwTuMX8_22484', 'W9YAR9qcuvN_44195', 'W9YAR9qcuvN_44195', 'U3oQjwTuMX8_25658', 'U3oQjwTuMX8_25658', 'wsAYBFtQaL7_44841', 'xWvSkKiWQpC_11830', 'xWvSkKiWQpC_11830', 'W9YAR9qcuvN_1786', 'ixTj1aTMup2_17992', 'CQWES1bawee_5365', 'CQWES1bawee_285', 'TSJmdttd2GV_22493', 'FnDDfrBZPhh_33935', 'oEPjPNSPmzL_39601', 'oEPjPNSPmzL_34120', 'FnDDfrBZPhh_13023', 'TSJmdttd2GV_23939', 'NtnvZSMK3en_641', 'iKFn6fzyRqs_24285', 'oahi4u45xMf_18830', 'YMNvYDhK8mB_31002', 'xgLmjqzoAzF_16227', 'xgLmjqzoAzF_22251', 'xgLmjqzoAzF_22251', 'iKFn6fzyRqs_2251', 'iKFn6fzyRqs_2251', 'YMNvYDhK8mB_33215', 'nS8T59Aw3sf_15566', 'XiJhRLvpKpX_43810', 'h6nwVLpAKQz_39616', 'g8Xrdbe9fir_21591', 'v7DzfFFEpsD_43080', 'h6nwVLpAKQz_32790']:
        #     return
        # new successes old keys
        # if episode_name not in ['8wJuSPJ9FXG_29199', '3XYAD64HpDr_18895', 'Jfyvj3xn2aJ_432', 'Jfyvj3xn2aJ_3685', 'xAHnY3QzFUN_40627', '3XYAD64HpDr_18895', '3XYAD64HpDr_18895', 'xAHnY3QzFUN_25779', 'GGBvSFddQgs_16157', 'qz3829g1Lzf_18141', 'JptJPosx1Z6_48533', '3CBBjsNkhqW_15605', 'HxmXPBbFCkH_25387', 'GGBvSFddQgs_49379', 'GGBvSFddQgs_38062', 'GtM3JtRvvvR_37656', 'qz3829g1Lzf_3499', 'HxmXPBbFCkH_25387', 'HxmXPBbFCkH_4616', 'HxmXPBbFCkH_25387', 'gQ3xxshDiCz_24311', 'gQ3xxshDiCz_20118', 'GtM3JtRvvvR_37656', 'wsAYBFtQaL7_15988', 'wsAYBFtQaL7_11015', 'wsAYBFtQaL7_12168', 'U3oQjwTuMX8_20162', 'U3oQjwTuMX8_24480', 'W9YAR9qcuvN_48848', 'W9YAR9qcuvN_41788', 'U3oQjwTuMX8_20162', 'U3oQjwTuMX8_24480', 'wsAYBFtQaL7_47711', 'xWvSkKiWQpC_13409', 'xWvSkKiWQpC_12503', 'W9YAR9qcuvN_2952', 'ixTj1aTMup2_18971', 'CQWES1bawee_7842', 'CQWES1bawee_7842', 'TSJmdttd2GV_27183', 'FnDDfrBZPhh_32983', 'oEPjPNSPmzL_32739', 'oEPjPNSPmzL_32739', 'FnDDfrBZPhh_18280', 'TSJmdttd2GV_27183', 'NtnvZSMK3en_7970', 'iKFn6fzyRqs_23818', 'oahi4u45xMf_16608', 'YMNvYDhK8mB_42897', 'xgLmjqzoAzF_18687', 'xgLmjqzoAzF_22221', 'xgLmjqzoAzF_24583', 'iKFn6fzyRqs_13', 'iKFn6fzyRqs_9123', 'YMNvYDhK8mB_42897', 'nS8T59Aw3sf_15358', 'XiJhRLvpKpX_43393', 'h6nwVLpAKQz_39733', 'g8Xrdbe9fir_16978', 'v7DzfFFEpsD_47687', 'h6nwVLpAKQz_39733']:
        #     return
        # new failures new keys
        # if episode_name not in ['LcAd9dhvVwh_18759', 'LcAd9dhvVwh_18759', 'LcAd9dhvVwh_18759', '3XYAD64HpDr_17316', 'gmuS7Wgsbrx_8510', '3XYAD64HpDr_2244', 'j6fHrce9pHR_18032', '3CBBjsNkhqW_15605', 'JptJPosx1Z6_1160', 'JptJPosx1Z6_1160', 'ggNAcMh8JPT_4315', 'GGBvSFddQgs_21970', '8wJuSPJ9FXG_29199', 'j6fHrce9pHR_22221', 'j6fHrce9pHR_18687', 'j6fHrce9pHR_24583', 'gQ3xxshDiCz_1648', 'GGBvSFddQgs_3499', 'vLpv2VX547B_31682', 'Wo6kuutE9i7_43361', 'vDfkYo5VqEQ_29422', 'gjhYih4upQ9_39471', 'gjhYih4upQ9_39471', 'Wo6kuutE9i7_18973', 'ixTj1aTMup2_18204', 'xWvSkKiWQpC_13374', 'TYDavTf8oyy_34054', 'gjhYih4upQ9_9391', 'ixTj1aTMup2_45339', 'VoVGtfYrpuQ_36143', 'pcpn6mFqFCg_6659', 'g7hUFVNac26_42461', 'g7hUFVNac26_41332', 'g7hUFVNac26_41332', '5biL7VEkByM_7650', 'Wo6kuutE9i7_42585', 'oEPjPNSPmzL_16978', 'oEPjPNSPmzL_16978', 'xWvSkKiWQpC_7777', 'TSJmdttd2GV_14603', 'VoVGtfYrpuQ_34391', 'NGyoyh91xXJ_48730', 'NGyoyh91xXJ_48730', 'nACV8wLu1u5_40418', 'MVVzj944atG_46872', 'MVVzj944atG_46872', 'iKFn6fzyRqs_41929', 'TSJmdttd2GV_14766', '5biL7VEkByM_27986', 'RaYrxWt5pR1_5528', 'RaYrxWt5pR1_5528', 'iigzG1rtanx_21307', 'HfMobPm86Xn_33234', '226REUyJh2K_48475', 'nS8T59Aw3sf_45913', 'nS8T59Aw3sf_45913', 'NEVASPhcrxR_5023', '1S7LAXRdDqK_22193', 'g8Xrdbe9fir_6152', 'xgLmjqzoAzF_16001', 'iigzG1rtanx_24968', 'XiJhRLvpKpX_39575', '226REUyJh2K_44342', 'Jfyvj3xn2aJ_4847', 'XiJhRLvpKpX_2028', 'wPLokgvCnuk_24939', 'wPLokgvCnuk_24939']:
        #     return
        # new failures old keys
        if episode_name not in ['LcAd9dhvVwh_17059', 'LcAd9dhvVwh_19490', 'LcAd9dhvVwh_19508', '3XYAD64HpDr_19815', 'gmuS7Wgsbrx_11364', '3XYAD64HpDr_4679', 'j6fHrce9pHR_15980', '3CBBjsNkhqW_11865', 'JptJPosx1Z6_3533', 'JptJPosx1Z6_2785', 'ggNAcMh8JPT_641', 'GGBvSFddQgs_13190', '8wJuSPJ9FXG_32258', 'j6fHrce9pHR_15980', 'j6fHrce9pHR_15980', 'j6fHrce9pHR_15980', 'gQ3xxshDiCz_480', 'GGBvSFddQgs_4090', 'vLpv2VX547B_29016', 'Wo6kuutE9i7_42449', 'vDfkYo5VqEQ_24283', 'gjhYih4upQ9_44698', 'gjhYih4upQ9_48553', 'Wo6kuutE9i7_18830', 'ixTj1aTMup2_22415', 'xWvSkKiWQpC_10225', 'TYDavTf8oyy_25710', 'gjhYih4upQ9_11496', 'ixTj1aTMup2_43359', 'VoVGtfYrpuQ_41057', 'pcpn6mFqFCg_8236', 'g7hUFVNac26_43080', 'g7hUFVNac26_34094', 'g7hUFVNac26_33574', '5biL7VEkByM_3719', 'Wo6kuutE9i7_42449', 'oEPjPNSPmzL_10892', 'oEPjPNSPmzL_18410', 'xWvSkKiWQpC_5481', 'TSJmdttd2GV_12249', 'VoVGtfYrpuQ_41057', 'NGyoyh91xXJ_45267', 'NGyoyh91xXJ_40631', 'nACV8wLu1u5_37687', 'MVVzj944atG_42679', 'MVVzj944atG_46557', 'iKFn6fzyRqs_43672', 'TSJmdttd2GV_12249', '5biL7VEkByM_27968', 'RaYrxWt5pR1_1709', 'RaYrxWt5pR1_1846', 'iigzG1rtanx_29778', 'HfMobPm86Xn_39658', '226REUyJh2K_40424', 'nS8T59Aw3sf_43939', 'nS8T59Aw3sf_43785', 'NEVASPhcrxR_9406', '1S7LAXRdDqK_18267', 'g8Xrdbe9fir_6271', 'xgLmjqzoAzF_18032', 'iigzG1rtanx_29778', 'XiJhRLvpKpX_38665', '226REUyJh2K_40424', 'Jfyvj3xn2aJ_117', 'XiJhRLvpKpX_7376', 'wPLokgvCnuk_19479', 'wPLokgvCnuk_24055']:
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
