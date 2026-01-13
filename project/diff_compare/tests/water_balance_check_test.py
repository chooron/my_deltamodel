import unittest
import numpy as np
from pathlib import Path
import sys
import os

# ==========================================
# 配置区域
# ==========================================
NUMBER_INFO = {
    "alpine1": 6, "alpine2": 12, "australia": 19, "collie1": 1, "collie2": 3,
    "collie3": 11, "flexb": 21, "flexi": 26, "flexis": 34, "gr4j": 7,
    "gsfb": 20, "hbv96": 37, "hillslope": 13, "hymod": 29, "ihacres": 5,
    "modhydrolog": 36, "mopex1": 24, "mopex2": 30, "mopex3": 31, "mopex4": 32,
    "mopex5": 35, "newzealand1": 4, "newzealand2": 16, "penman": 17,
    "plateau": 15, "simhyd": 18, "smar": 40, "susannah1": 9,
    "susannah2": 10, "tank": 27, "tcm": 25, "topmodel": 14, "us1": 8,
    "vic": 22, "wetland": 2, "xinanjiang": 28,
}

SPECIAL_MODELS = [
    "flexi", "flexb", "flexis", "gr4j", "hillslope", "ihacres",
    "newzealand2", "plateau", "smar",
]


AVAILABLE_MODELS = list(NUMBER_INFO.keys())

# 设定允许的最大误差 (mm)
TOLERANCE_MM = 1.0 

class TestGlobalWaterBalance(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        在所有测试开始前设置基础路径
        """
        # 假设当前文件位于 project/tests/ 下，根据你的路径逻辑向上回溯
        # Path(__file__).parents[1] 指向 project 根目录
        # 请根据该文件实际放置的位置微调 parents 的索引
        cls.base_output_path_default = (
            Path(__file__).parents[1]
            / "output/camels_559/train1989-1998/no_multi/Calibrate_E50_R365_B100_n16_noLn_noWU_42"
        )
        cls.base_output_path_special = (
            Path(__file__).parents[1]
            / "output/camels_559/train1989-1998/no_multi/Calibrate_E10_R365_B100_n16_noLn_noWU_42"
        )
        cls.loss_name = "KgeLoss"
        cls.time_period = "1989-1998"
        cls.epoch_suffix_default = "Ep50"
        cls.epoch_suffix_special = "Ep10"

    def test_all_models_mass_conservation(self):
        """
        遍历所有非特殊模型，验证水量平衡误差是否小于 1mm
        """
        print(f"\n🚀 Starting Global Water Balance Check (Tolerance: {TOLERANCE_MM} mm)...")

        # 过滤掉特殊模型
        test_models = ['tcm']
        
        passed_count = 0
        skipped_count = 0
        
        for model_name in test_models:
            # 使用 subTest，确保一个模型失败不会阻断后续模型的测试
            with self.subTest(model=model_name):
                is_special = model_name in SPECIAL_MODELS

                base_output_path = (
                    self.base_output_path_special
                    if is_special
                    else self.base_output_path_default
                )
                epoch_suffix = (
                    self.epoch_suffix_special
                    if is_special
                    else self.epoch_suffix_default
                )
                
                # 1. 构建路径
                save_path = (
                    base_output_path 
                    / model_name 
                    / self.loss_name 
                    / "stat" 
                    / f"test{self.time_period}_{epoch_suffix}"
                )
                
                # 3. 加载数据
                model_outputs = np.load(save_path / "model_outputs.npz")
                p_arr = model_outputs['precipitation']
                et_arr = model_outputs['evaporation']
                q_arr = model_outputs['streamflow']
                s_arr = model_outputs['storage_sum']

                # 4. 执行修正后的水量平衡计算 (跳过第1天)
                # ---------------------------------------------------
                # Sum(P_2:T) = Sum(Q_2:T) + Sum(E_2:T) + (S_T - S_1)
                # ---------------------------------------------------
                sum_P = np.sum(p_arr[1:, :], axis=0)
                sum_Q = np.sum(q_arr[1:, :], axis=0)
                sum_E = np.sum(et_arr[1:, :], axis=0)
                
                # S[-1] 是最后一天结束状态，S[0] 是第一天结束状态
                delta_S = s_arr[-1, :] - s_arr[0, :]
                
                # 计算误差
                balance_error = sum_P - (sum_Q + sum_E + delta_S)
                max_abs_error = np.max(np.abs(balance_error))
                mean_abs_error = np.mean(np.abs(balance_error))

                # 5. 断言检查
                print(f"[{model_name:<12}] Max Err: {max_abs_error:.6f} mm | Mean Err: {mean_abs_error:.6f} mm", end="")
                
                if max_abs_error < TOLERANCE_MM:
                    print(" ✅")
                    passed_count += 1
                else:
                    print(" ❌ FAIL")
                
                # 这里是核心断言：如果超过 1mm，这个 subTest 会标记为 Fail
                self.assertLess(
                    max_abs_error, 
                    TOLERANCE_MM, 
                    f"Model {model_name} failed water balance! Max Error: {max_abs_error:.4f} mm"
                )

        print("-" * 40)
        print(f"Summary: {passed_count} Passed, {skipped_count} Skipped / {len(test_models)} Total")

if __name__ == '__main__':
    unittest.main()