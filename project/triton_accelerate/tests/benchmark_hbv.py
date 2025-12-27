"""
HBV 模型性能基准测试

比较三种实现方式在不同 batch size 和时间步长下的性能:
1. Triton: 使用 Triton 加速的实现
2. PyTorch: 原生 PyTorch 实现
3. PyTorch JIT: 使用 torch.jit.script 编译的 PyTorch 实现

测试指标:
- 显存占用 (峰值)
- 前向传播时间
- 反向传播时间
- 总运行时间

Author: Generated for performance comparison
Date: 2025-12-10
"""

import torch
import torch.nn.functional as F
import time
import gc
import sys
import os
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from tabulate import tabulate
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到路径
sys.path.append(os.getenv("PROJ_PATH", "."))

# 导入 Triton 实现
from project.triton_accelerate.models.hbv_manual_core import hbv_step_triton, hbv_run_triton


# ==========================================
# PyTorch 版本的平滑函数
# ==========================================

def smooth_step(x: torch.Tensor) -> torch.Tensor:
    """平滑阶跃函数: 0.5 * (tanh(5x) + 1)"""
    return 0.5 * (torch.tanh(5.0 * x) + 1.0)


def smooth_relu(x: torch.Tensor) -> torch.Tensor:
    """平滑 ReLU: x * smooth_step(x)"""
    return x * smooth_step(x)


# ==========================================
# PyTorch 原生实现 (无 JIT)
# ==========================================

def hbv_step_pytorch(
    p: torch.Tensor,
    t_val: torch.Tensor,
    pet: torch.Tensor,
    snow: torch.Tensor,
    melt: torch.Tensor,
    sm: torch.Tensor,
    suz: torch.Tensor,
    slz: torch.Tensor,
    tt: torch.Tensor,
    cfmax: torch.Tensor,
    cfr: torch.Tensor,
    cwh: torch.Tensor,
    fc: torch.Tensor,
    beta: torch.Tensor,
    lp: torch.Tensor,
    betaet: torch.Tensor,
    c_par: torch.Tensor,
    perc: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    uzl: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    """HBV 单步前向传播 - PyTorch 原生实现"""
    # 1. 降水分离
    temp_diff = t_val - tt
    is_warm = smooth_step(temp_diff)
    
    rain = p * is_warm
    snow_input = p * (1.0 - is_warm)

    # 2. 积雪模块
    snow = snow + snow_input
    
    pot_melt = cfmax * smooth_relu(temp_diff)
    melt_amount = torch.min(pot_melt, snow)
    
    snow = snow - melt_amount
    melt = melt + melt_amount

    # 再冻结
    pot_refreeze = cfr * cfmax * smooth_relu(-temp_diff)
    refreezing = torch.min(pot_refreeze, melt)
    
    snow = snow + refreezing
    melt = melt - refreezing

    # 融雪出流
    tosoil = smooth_relu(melt - cwh * snow)
    melt = melt - tosoil

    # 3. 土壤模块
    ratio = sm / (fc + 1e-6)
    ratio_safe = torch.clamp(ratio, 0.0, 1.0)
    
    soil_wetness = torch.pow(ratio_safe + 1e-6, beta)
    
    recharge = (rain + tosoil) * soil_wetness
    sm = sm + rain + tosoil - recharge

    # 超渗
    excess = smooth_relu(sm - fc)
    sm = sm - excess

    # 蒸发
    limit_val = lp * fc
    evap_ratio = sm / (limit_val + 1e-6)
    evap_ratio_safe = torch.clamp(evap_ratio, 0.0, 1.0)
    
    evapfactor = torch.pow(evap_ratio_safe + 1e-6, betaet)
    
    potential_et = pet * evapfactor
    etact = sm - smooth_relu(sm - potential_et)
    
    sm = torch.clamp(sm - etact, min=nearzero)

    # 4. 汇流模块
    capillary = torch.min(slz, c_par * slz * (1.0 - ratio_safe))
    sm = sm + capillary
    slz = slz - capillary

    suz = suz + recharge + excess
    perc_flux = torch.min(suz, perc)
    suz = suz - perc_flux

    # Q0
    q0 = k0 * smooth_relu(suz - uzl)
    suz = suz - q0

    q1 = k1 * suz
    suz = suz - q1

    slz = slz + perc_flux
    q2 = k2 * slz
    slz = slz - q2

    q_total = q0 + q1 + q2

    return snow, melt, sm, suz, slz, q_total


def hbv_run_pytorch(
    p: torch.Tensor,          # [batch, seq_len]
    t_val: torch.Tensor,      # [batch, seq_len]
    pet: torch.Tensor,        # [batch, seq_len]
    snow_init: torch.Tensor,  # [batch]
    melt_init: torch.Tensor,
    sm_init: torch.Tensor,
    suz_init: torch.Tensor,
    slz_init: torch.Tensor,
    tt: torch.Tensor,         # [batch] 参数
    cfmax: torch.Tensor,
    cfr: torch.Tensor,
    cwh: torch.Tensor,
    fc: torch.Tensor,
    beta: torch.Tensor,
    lp: torch.Tensor,
    betaet: torch.Tensor,
    c_par: torch.Tensor,
    perc: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    uzl: torch.Tensor,
) -> torch.Tensor:
    """运行 HBV 模型多个时间步 - PyTorch 原生实现"""
    batch_size, seq_len = p.shape
    device = p.device
    
    # 初始化状态
    snow = snow_init.clone()
    melt = melt_init.clone()
    sm = sm_init.clone()
    suz = suz_init.clone()
    slz = slz_init.clone()
    
    # 存储输出
    q_list = []
    
    for t in range(seq_len):
        snow, melt, sm, suz, slz, q = hbv_step_pytorch(
            p[:, t], t_val[:, t], pet[:, t],
            snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
            c_par, perc, k0, k1, k2, uzl
        )
        q_list.append(q)
    
    return torch.stack(q_list, dim=1)  # [batch, seq_len]


# ==========================================
# PyTorch JIT 编译版本
# ==========================================

@torch.jit.script
def smooth_step_jit(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.tanh(5.0 * x) + 1.0)


@torch.jit.script
def smooth_relu_jit(x: torch.Tensor) -> torch.Tensor:
    return x * smooth_step_jit(x)


@torch.jit.script
def hbv_step_jit(
    p: torch.Tensor,
    t_val: torch.Tensor,
    pet: torch.Tensor,
    snow: torch.Tensor,
    melt: torch.Tensor,
    sm: torch.Tensor,
    suz: torch.Tensor,
    slz: torch.Tensor,
    tt: torch.Tensor,
    cfmax: torch.Tensor,
    cfr: torch.Tensor,
    cwh: torch.Tensor,
    fc: torch.Tensor,
    beta: torch.Tensor,
    lp: torch.Tensor,
    betaet: torch.Tensor,
    c_par: torch.Tensor,
    perc: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    uzl: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """HBV 单步前向传播 - PyTorch JIT 编译版本"""
    # 1. 降水分离
    temp_diff = t_val - tt
    is_warm = smooth_step_jit(temp_diff)
    
    rain = p * is_warm
    snow_input = p * (1.0 - is_warm)

    # 2. 积雪模块
    snow = snow + snow_input
    
    pot_melt = cfmax * smooth_relu_jit(temp_diff)
    melt_amount = torch.min(pot_melt, snow)
    
    snow = snow - melt_amount
    melt = melt + melt_amount

    # 再冻结
    pot_refreeze = cfr * cfmax * smooth_relu_jit(-temp_diff)
    refreezing = torch.min(pot_refreeze, melt)
    
    snow = snow + refreezing
    melt = melt - refreezing

    # 融雪出流
    tosoil = smooth_relu_jit(melt - cwh * snow)
    melt = melt - tosoil

    # 3. 土壤模块
    ratio = sm / (fc + 1e-6)
    ratio_safe = torch.clamp(ratio, 0.0, 1.0)
    
    soil_wetness = torch.pow(ratio_safe + 1e-6, beta)
    
    recharge = (rain + tosoil) * soil_wetness
    sm = sm + rain + tosoil - recharge

    # 超渗
    excess = smooth_relu_jit(sm - fc)
    sm = sm - excess

    # 蒸发
    limit_val = lp * fc
    evap_ratio = sm / (limit_val + 1e-6)
    evap_ratio_safe = torch.clamp(evap_ratio, 0.0, 1.0)
    
    evapfactor = torch.pow(evap_ratio_safe + 1e-6, betaet)
    
    potential_et = pet * evapfactor
    etact = sm - smooth_relu_jit(sm - potential_et)
    
    sm = torch.clamp(sm - etact, min=nearzero)

    # 4. 汇流模块
    capillary = torch.min(slz, c_par * slz * (1.0 - ratio_safe))
    sm = sm + capillary
    slz = slz - capillary

    suz = suz + recharge + excess
    perc_flux = torch.min(suz, perc)
    suz = suz - perc_flux

    # Q0
    q0 = k0 * smooth_relu_jit(suz - uzl)
    suz = suz - q0

    q1 = k1 * suz
    suz = suz - q1

    slz = slz + perc_flux
    q2 = k2 * slz
    slz = slz - q2

    q_total = q0 + q1 + q2

    return snow, melt, sm, suz, slz, q_total

@torch.jit.script
def hbv_run_jit(
    p: torch.Tensor,
    t_val: torch.Tensor,
    pet: torch.Tensor,
    snow_init: torch.Tensor,
    melt_init: torch.Tensor,
    sm_init: torch.Tensor,
    suz_init: torch.Tensor,
    slz_init: torch.Tensor,
    tt: torch.Tensor,
    cfmax: torch.Tensor,
    cfr: torch.Tensor,
    cwh: torch.Tensor,
    fc: torch.Tensor,
    beta: torch.Tensor,
    lp: torch.Tensor,
    betaet: torch.Tensor,
    c_par: torch.Tensor,
    perc: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    uzl: torch.Tensor,
) -> torch.Tensor:
    """运行 HBV 模型多个时间步 - PyTorch JIT 版本"""
    batch_size, seq_len = p.shape
    
    snow = snow_init.clone()
    melt = melt_init.clone()
    sm = sm_init.clone()
    suz = suz_init.clone()
    slz = slz_init.clone()
    
    q_list = []
    
    for t in range(seq_len):
        snow, melt, sm, suz, slz, q = hbv_step_jit(
            p[:, t], t_val[:, t], pet[:, t],
            snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
            c_par, perc, k0, k1, k2, uzl
        )
        q_list.append(q)
    
    return torch.stack(q_list, dim=1)


# ==========================================
# 性能测试工具
# ==========================================

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    method: str
    batch_size: int
    seq_len: int
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    peak_memory_mb: float
    success: bool
    error_msg: str = ""


def clear_cuda_cache():
    """清理 CUDA 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def get_peak_memory_mb() -> float:
    """获取 CUDA 峰值显存占用 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def generate_test_data(batch_size: int, seq_len: int, device: torch.device, seed: int = 42):
    """生成测试数据"""
    torch.manual_seed(seed)
    
    # 输入数据
    p = torch.rand(batch_size, seq_len, device=device) * 10.0
    t_val = torch.rand(batch_size, seq_len, device=device) * 20.0 - 5.0
    pet = torch.rand(batch_size, seq_len, device=device) * 5.0
    
    # 初始状态
    snow_init = torch.rand(batch_size, device=device) * 10.0
    melt_init = torch.rand(batch_size, device=device) * 5.0
    sm_init = torch.rand(batch_size, device=device) * 100.0 + 50.0
    suz_init = torch.rand(batch_size, device=device) * 20.0
    slz_init = torch.rand(batch_size, device=device) * 30.0
    
    # HBV 参数
    params = {
        'tt': torch.rand(batch_size, device=device) * 2.0 - 1.0,
        'cfmax': torch.rand(batch_size, device=device) * 3.0 + 1.0,
        'cfr': torch.rand(batch_size, device=device) * 0.1,
        'cwh': torch.rand(batch_size, device=device) * 0.2,
        'fc': torch.rand(batch_size, device=device) * 200.0 + 100.0,
        'beta': torch.rand(batch_size, device=device) * 3.0 + 1.0,
        'lp': torch.rand(batch_size, device=device) * 0.5 + 0.3,
        'betaet': torch.rand(batch_size, device=device) * 2.0 + 0.5,
        'c_par': torch.rand(batch_size, device=device) * 0.1,
        'perc': torch.rand(batch_size, device=device) * 3.0 + 0.5,
        'k0': torch.rand(batch_size, device=device) * 0.3 + 0.1,
        'k1': torch.rand(batch_size, device=device) * 0.1 + 0.01,
        'k2': torch.rand(batch_size, device=device) * 0.05 + 0.001,
        'uzl': torch.rand(batch_size, device=device) * 30.0 + 10.0,
    }
    
    return p, t_val, pet, snow_init, melt_init, sm_init, suz_init, slz_init, params


def benchmark_triton(
    batch_size: int, seq_len: int, device: torch.device, 
    num_warmup: int = 3, num_runs: int = 10
) -> BenchmarkResult:
    """测试 Triton 实现"""
    try:
        clear_cuda_cache()
        
        # 生成数据
        p, t_val, pet, snow_init, melt_init, sm_init, suz_init, slz_init, params = \
            generate_test_data(batch_size, seq_len, device)
        
        # Triton 版本使用 [T, batch] 格式，需要转置
        p_t = p.transpose(0, 1).contiguous()  # [seq_len, batch]
        t_val_t = t_val.transpose(0, 1).contiguous()
        pet_t = pet.transpose(0, 1).contiguous()
        
        # 初始状态字典
        init_states = {
            'snow': snow_init,
            'melt': melt_init,
            'sm': sm_init,
            'suz': suz_init,
            'slz': slz_init,
        }
        
        # 设置参数需要梯度
        for key in params:
            params[key].requires_grad_(True)
        
        # Warmup
        for _ in range(num_warmup):
            q, _ = hbv_run_triton(p_t, t_val_t, pet_t, params, init_states)
            loss = q.sum()
            loss.backward()
            for key in params:
                if params[key].grad is not None:
                    params[key].grad.zero_()
        
        clear_cuda_cache()
        torch.cuda.synchronize()
        
        # 正式测试
        forward_times = []
        backward_times = []
        
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            q, _ = hbv_run_triton(p_t, t_val_t, pet_t, params, init_states)
            
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            
            loss = q.sum()
            loss.backward()
            
            torch.cuda.synchronize()
            backward_end = time.perf_counter()
            
            forward_times.append((forward_end - start) * 1000)
            backward_times.append((backward_end - forward_end) * 1000)
            
            for key in params:
                if params[key].grad is not None:
                    params[key].grad.zero_()
        
        peak_memory = get_peak_memory_mb()
        
        return BenchmarkResult(
            method="Triton",
            batch_size=batch_size,
            seq_len=seq_len,
            forward_time_ms=np.mean(forward_times),
            backward_time_ms=np.mean(backward_times),
            total_time_ms=np.mean(forward_times) + np.mean(backward_times),
            peak_memory_mb=peak_memory,
            success=True
        )
    except Exception as e:
        return BenchmarkResult(
            method="Triton",
            batch_size=batch_size,
            seq_len=seq_len,
            forward_time_ms=0,
            backward_time_ms=0,
            total_time_ms=0,
            peak_memory_mb=0,
            success=False,
            error_msg=str(e)
        )


def benchmark_pytorch(
    batch_size: int, seq_len: int, device: torch.device,
    num_warmup: int = 3, num_runs: int = 10
) -> BenchmarkResult:
    """测试 PyTorch 原生实现"""
    try:
        clear_cuda_cache()
        
        # 生成数据
        p, t_val, pet, snow_init, melt_init, sm_init, suz_init, slz_init, params = \
            generate_test_data(batch_size, seq_len, device)
        
        # 设置参数需要梯度
        for key in params:
            params[key].requires_grad_(True)
        
        # Warmup
        for _ in range(num_warmup):
            q = hbv_run_pytorch(
                p, t_val, pet, snow_init, melt_init, sm_init, suz_init, slz_init,
                **params
            )
            loss = q.sum()
            loss.backward()
            for key in params:
                if params[key].grad is not None:
                    params[key].grad.zero_()
        
        clear_cuda_cache()
        torch.cuda.synchronize()
        
        # 正式测试
        forward_times = []
        backward_times = []
        
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            q = hbv_run_pytorch(
                p, t_val, pet, snow_init, melt_init, sm_init, suz_init, slz_init,
                **params
            )
            
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            
            loss = q.sum()
            loss.backward()
            
            torch.cuda.synchronize()
            backward_end = time.perf_counter()
            
            forward_times.append((forward_end - start) * 1000)
            backward_times.append((backward_end - forward_end) * 1000)
            
            for key in params:
                if params[key].grad is not None:
                    params[key].grad.zero_()
        
        peak_memory = get_peak_memory_mb()
        
        return BenchmarkResult(
            method="PyTorch",
            batch_size=batch_size,
            seq_len=seq_len,
            forward_time_ms=np.mean(forward_times),
            backward_time_ms=np.mean(backward_times),
            total_time_ms=np.mean(forward_times) + np.mean(backward_times),
            peak_memory_mb=peak_memory,
            success=True
        )
    except Exception as e:
        return BenchmarkResult(
            method="PyTorch",
            batch_size=batch_size,
            seq_len=seq_len,
            forward_time_ms=0,
            backward_time_ms=0,
            total_time_ms=0,
            peak_memory_mb=0,
            success=False,
            error_msg=str(e)
        )


def benchmark_pytorch_jit(
    batch_size: int, seq_len: int, device: torch.device,
    num_warmup: int = 3, num_runs: int = 10
) -> BenchmarkResult:
    """测试 PyTorch JIT 实现"""
    try:
        clear_cuda_cache()
        
        # 生成数据
        p, t_val, pet, snow_init, melt_init, sm_init, suz_init, slz_init, params = \
            generate_test_data(batch_size, seq_len, device)
        
        # 设置参数需要梯度
        for key in params:
            params[key].requires_grad_(True)
        
        # Warmup (JIT 编译发生在第一次调用)
        for _ in range(num_warmup):
            q = hbv_run_jit(
                p, t_val, pet, snow_init, melt_init, sm_init, suz_init, slz_init,
                **params
            )
            loss = q.sum()
            loss.backward()
            for key in params:
                if params[key].grad is not None:
                    params[key].grad.zero_()
        
        clear_cuda_cache()
        torch.cuda.synchronize()
        
        # 正式测试
        forward_times = []
        backward_times = []
        
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            q = hbv_run_jit(
                p, t_val, pet, snow_init, melt_init, sm_init, suz_init, slz_init,
                **params
            )
            
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            
            loss = q.sum()
            loss.backward()
            
            torch.cuda.synchronize()
            backward_end = time.perf_counter()
            
            forward_times.append((forward_end - start) * 1000)
            backward_times.append((backward_end - forward_end) * 1000)
            
            for key in params:
                if params[key].grad is not None:
                    params[key].grad.zero_()
        
        peak_memory = get_peak_memory_mb()
        
        return BenchmarkResult(
            method="PyTorch JIT",
            batch_size=batch_size,
            seq_len=seq_len,
            forward_time_ms=np.mean(forward_times),
            backward_time_ms=np.mean(backward_times),
            total_time_ms=np.mean(forward_times) + np.mean(backward_times),
            peak_memory_mb=peak_memory,
            success=True
        )
    except Exception as e:
        return BenchmarkResult(
            method="PyTorch JIT",
            batch_size=batch_size,
            seq_len=seq_len,
            forward_time_ms=0,
            backward_time_ms=0,
            total_time_ms=0,
            peak_memory_mb=0,
            success=False,
            error_msg=str(e)
        )


def run_benchmark_suite(
    batch_sizes: List[int] = [32, 64, 128, 256, 512, 1024],
    seq_lens: List[int] = [100, 365, 730, 1461],
    num_warmup: int = 3,
    num_runs: int = 10
) -> List[BenchmarkResult]:
    """运行完整的基准测试套件"""
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，无法进行 GPU 性能测试")
        return []
    
    device = torch.device("cuda")
    
    # 打印 GPU 信息
    print("=" * 80)
    print("GPU 信息")
    print("=" * 80)
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    print()
    
    results = []
    total_tests = len(batch_sizes) * len(seq_lens) * 3
    current_test = 0
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"\n{'='*80}")
            print(f"测试配置: batch_size={batch_size}, seq_len={seq_len}")
            print(f"{'='*80}")
            
            # Triton
            current_test += 1
            print(f"[{current_test}/{total_tests}] 测试 Triton...")
            result = benchmark_triton(batch_size, seq_len, device, num_warmup, num_runs)
            results.append(result)
            if result.success:
                print(f"  ✓ Forward: {result.forward_time_ms:.2f}ms, "
                      f"Backward: {result.backward_time_ms:.2f}ms, "
                      f"Memory: {result.peak_memory_mb:.1f}MB")
            else:
                print(f"  ✗ 失败: {result.error_msg}")
            
            # PyTorch
            current_test += 1
            print(f"[{current_test}/{total_tests}] 测试 PyTorch...")
            result = benchmark_pytorch(batch_size, seq_len, device, num_warmup, num_runs)
            results.append(result)
            if result.success:
                print(f"  ✓ Forward: {result.forward_time_ms:.2f}ms, "
                      f"Backward: {result.backward_time_ms:.2f}ms, "
                      f"Memory: {result.peak_memory_mb:.1f}MB")
            else:
                print(f"  ✗ 失败: {result.error_msg}")
            
            # PyTorch JIT
            current_test += 1
            print(f"[{current_test}/{total_tests}] 测试 PyTorch JIT...")
            result = benchmark_pytorch_jit(batch_size, seq_len, device, num_warmup, num_runs)
            results.append(result)
            if result.success:
                print(f"  ✓ Forward: {result.forward_time_ms:.2f}ms, "
                      f"Backward: {result.backward_time_ms:.2f}ms, "
                      f"Memory: {result.peak_memory_mb:.1f}MB")
            else:
                print(f"  ✗ 失败: {result.error_msg}")
    
    return results


def print_results_table(results: List[BenchmarkResult]):
    """以表格形式打印结果"""
    
    print("\n" + "=" * 100)
    print("基准测试结果汇总")
    print("=" * 100)
    
    # 按配置分组
    configs = {}
    for r in results:
        key = (r.batch_size, r.seq_len)
        if key not in configs:
            configs[key] = {}
        configs[key][r.method] = r
    
    # 打印每个配置的比较
    for (batch_size, seq_len), methods in sorted(configs.items()):
        print(f"\n配置: batch_size={batch_size}, seq_len={seq_len}")
        print("-" * 90)
        
        headers = ["方法", "前向(ms)", "反向(ms)", "总计(ms)", "显存(MB)", "状态", "加速比"]
        table_data = []
        
        # 以 PyTorch 为基准计算加速比
        pytorch_time = methods.get("PyTorch")
        base_time = pytorch_time.total_time_ms if pytorch_time and pytorch_time.success else None
        
        for method_name in ["Triton", "PyTorch", "PyTorch JIT"]:
            r = methods.get(method_name)
            if r is None:
                continue
            
            if r.success:
                speedup = f"{base_time / r.total_time_ms:.2f}x" if base_time else "N/A"
                table_data.append([
                    r.method,
                    f"{r.forward_time_ms:.2f}",
                    f"{r.backward_time_ms:.2f}",
                    f"{r.total_time_ms:.2f}",
                    f"{r.peak_memory_mb:.1f}",
                    "✓",
                    speedup
                ])
            else:
                table_data.append([
                    r.method, "N/A", "N/A", "N/A", "N/A", "✗", "N/A"
                ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_summary_comparison(results: List[BenchmarkResult]):
    """打印总体比较摘要"""
    
    print("\n" + "=" * 100)
    print("性能对比摘要 (Triton vs PyTorch)")
    print("=" * 100)
    
    # 计算平均加速比
    triton_speedups = []
    jit_speedups = []
    triton_memory_ratios = []
    jit_memory_ratios = []
    
    configs = {}
    for r in results:
        key = (r.batch_size, r.seq_len)
        if key not in configs:
            configs[key] = {}
        configs[key][r.method] = r
    
    for (batch_size, seq_len), methods in configs.items():
        pytorch = methods.get("PyTorch")
        triton = methods.get("Triton")
        jit = methods.get("PyTorch JIT")
        
        if pytorch and pytorch.success:
            if triton and triton.success:
                triton_speedups.append(pytorch.total_time_ms / triton.total_time_ms)
                if triton.peak_memory_mb > 0:
                    triton_memory_ratios.append(triton.peak_memory_mb / pytorch.peak_memory_mb)
            
            if jit and jit.success:
                jit_speedups.append(pytorch.total_time_ms / jit.total_time_ms)
                if jit.peak_memory_mb > 0:
                    jit_memory_ratios.append(jit.peak_memory_mb / pytorch.peak_memory_mb)
    
    if triton_speedups:
        print(f"\nTriton vs PyTorch:")
        print(f"  平均加速比: {np.mean(triton_speedups):.2f}x")
        print(f"  最大加速比: {np.max(triton_speedups):.2f}x")
        print(f"  最小加速比: {np.min(triton_speedups):.2f}x")
        if triton_memory_ratios:
            print(f"  平均显存比: {np.mean(triton_memory_ratios):.2f}x")
    
    if jit_speedups:
        print(f"\nPyTorch JIT vs PyTorch:")
        print(f"  平均加速比: {np.mean(jit_speedups):.2f}x")
        print(f"  最大加速比: {np.max(jit_speedups):.2f}x")
        print(f"  最小加速比: {np.min(jit_speedups):.2f}x")
        if jit_memory_ratios:
            print(f"  平均显存比: {np.mean(jit_memory_ratios):.2f}x")


def main():
    """主函数"""
    print("=" * 80)
    print("HBV 模型性能基准测试")
    print("比较: Triton vs PyTorch vs PyTorch JIT")
    print("=" * 80)
    
    # 测试配置
    # 可以根据需要调整
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    seq_lens = [100, 365, 730, 1461]  # 约100天, 1年, 2年, 4年
    
    # 运行测试
    results = run_benchmark_suite(
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        num_warmup=3,
        num_runs=10
    )
    
    if results:
        # 打印详细结果
        print_results_table(results)
        
        # 打印总结
        print_summary_comparison(results)
        
        # 保存结果到文件
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(output_dir, "benchmark_results.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("HBV 模型性能基准测试结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA: {torch.version.cuda}\n")
            f.write(f"PyTorch: {torch.__version__}\n\n")
            
            for r in results:
                if r.success:
                    f.write(f"{r.method} | batch={r.batch_size} | seq={r.seq_len} | "
                           f"fwd={r.forward_time_ms:.2f}ms | bwd={r.backward_time_ms:.2f}ms | "
                           f"mem={r.peak_memory_mb:.1f}MB\n")
                else:
                    f.write(f"{r.method} | batch={r.batch_size} | seq={r.seq_len} | FAILED: {r.error_msg}\n")
        
        print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
