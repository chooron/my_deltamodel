import sys
import os
import torch
import unittest
from torch.autograd import gradcheck
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH", "."))
# 导入你项目中的核心注册表
# 请根据实际路径调整 import
from dmg.models.phy_models.core import PARAM_INFO, STFN_INFO, INIT_INFO # noqa

SPECIAL_MODELS = [
    "flexi",
    "flexb",
    "flexis",
    "gr4j",
    "hillslope",
    "ihacres",
    "mopex4",
    "mopex5",
    "newzealand2",
    "plateau",
    "smar",
]

MODELS_WITH_DOY = {"mopex4", "mopex5"}

class TestAllModelsGradient(unittest.TestCase):
    def setUp(self):
        # 1. 基础配置：使用双精度 float64 以确保 gradcheck 通过
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)
        
        # 2. 这里的 nearzero 要稍微大一点，避免随机生成的数据太靠近 0 导致数值不稳定
        self.nearzero = 1e-5
        
        # 3. 测试维度 (Batch Size)
        # 不需要很大，2x1 足够验证物理逻辑，且计算快
        self.n_grid = 2
        self.nmul = 1 
        self.shape = (self.n_grid, self.nmul)

    def _generate_random_tensor(self, low, high, requires_grad=True):
        """辅助函数：生成指定范围内的随机张量"""
        data = torch.rand(self.shape, device=self.device, dtype=self.dtype)
        # 线性映射到 [low, high]
        data = data * (high - low) + low
        if requires_grad:
            data.requires_grad_(True)
        return data

    def test_batch_models(self):
        """
        核心测试逻辑：遍历所有注册的模型并进行梯度检查
        """
        # 获取所有模型名称
        # model_names = ['mopex4','mopex5']
        model_names = list(PARAM_INFO.keys())
        print(f"\n🚀 Starting Batch Gradient Check for {len(model_names)} models...")
        
        failed_models = []
        
        for model_name in model_names:
            print(f"\n[Testing] Model: {model_name} ...", end=" ", flush=True)
            try:
                self.run_single_model_check(model_name)
                print("✅ PASSED")
            except Exception as e:
                print(f"❌ FAILED")
                print(f"    Error: {str(e)}")
                failed_models.append(model_name)
                
        print("\n" + "="*40)
        if len(failed_models) == 0:
            print("🎉 ALL MODELS PASSED GRADIENT CHECK!")
        else:
            print(f"⚠️  {len(failed_models)} MODELS FAILED:")
            for m in failed_models:
                print(f"   - {m}")
        print("="*40)
        
        # 如果有失败的模型，让 unittest 报错
        self.assertEqual(len(failed_models), 0, f"Models failed: {failed_models}")

    def run_single_model_check(self, model_name):
        # 1. 获取模型元数据
        step_fn = STFN_INFO[model_name]       # 物理过程函数
        param_bounds = PARAM_INFO[model_name] # 参数边界
        init_fn = INIT_INFO[model_name]       # 【修改点1】获取初始化函数
        
        param_names = list(param_bounds.keys())
        n_params = len(param_names)

        needs_doy = model_name in MODELS_WITH_DOY

        # 2. 构造随机输入 (P, T, PET, [DOY])
        P = self._generate_random_tensor(0.1, 10.0)
        T = self._generate_random_tensor(-5.0, 25.0)
        PET = self._generate_random_tensor(0.1, 5.0)
        if needs_doy:
            doy = self._generate_random_tensor(1.0, 365.0)

        # 3. 构造随机参数 (Params)
        params_list = []
        for p_name in param_names:
            low, high = param_bounds[p_name]
            margin = (high - low) * 0.1
            p_val = self._generate_random_tensor(low + margin, high - margin)
            params_list.append(p_val)

        # 4. 构造随机状态 (States) - 【核心修改逻辑】
        # 不要相信 STATE_INFO 的数字，直接运行 init_fn 看看它到底返回几个状态
        # 注意：init_fn 返回的是全0或nearzero，我们需要保持形状但填入随机数
        with torch.no_grad():
            # 获取正确的状态数量和形状模板
            dummy_states = init_fn(self.n_grid, self.nmul, self.device, self.nearzero)
        
        states_list = []
        for s_template in dummy_states:
            # 生成一个形状一样，但数值在 [10, 100] 的随机 Tensor
            # 必须重新生成，因为 s_template 通常是 0，会导致梯度检查在 ReLU 处失败
            s_val = torch.rand_like(s_template, dtype=self.dtype, device=self.device)
            s_val = s_val * (100.0 - 10.0) + 10.0 
            s_val.requires_grad_(True)
            states_list.append(s_val)

        # 5. 定义 Wrapper 函数
        def functional_wrapper(*wrapper_args):
            # 动态切分 args：先基础输入，再参数，再状态
            p_in, t_in, pet_in = wrapper_args[0:3]
            offset = 3
            doy_in = None
            if needs_doy:
                doy_in = wrapper_args[offset]
                offset += 1

            current_params = wrapper_args[offset : offset + n_params]
            current_states = wrapper_args[offset + n_params :]

            call_args = [p_in, t_in, pet_in]
            if needs_doy:
                call_args.append(doy_in)

            return step_fn(
                *call_args,
                *current_params,
                *current_states,
                self.nearzero,
            )

        # 6. 准备输入 Tuple
        # 顺序: P, T, PET, [DOY], Param1...ParamN, State1...StateM
        if needs_doy:
            inputs = (P, T, PET, doy) + tuple(params_list) + tuple(states_list)
        else:
            inputs = (P, T, PET) + tuple(params_list) + tuple(states_list)

        # 7. 执行 Gradcheck
        gradcheck(functional_wrapper, inputs, eps=1e-6, atol=1e-5, raise_exception=True)

if __name__ == '__main__':
    unittest.main()