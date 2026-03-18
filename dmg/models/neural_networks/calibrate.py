import numpy as np
import torch
import torch.nn as nn
from typing import Union
from scipy.stats import qmc

def compute_nmul(ny: int, multiplier: int = 20) -> int:
    """根据参数数量动态计算成员数，向上取最近的2的幂次"""
    raw = max(multiplier * ny, 32)
    return int(2 ** np.ceil(np.log2(raw)))


class Calibrate(torch.nn.Module):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        num_basins: int = 100,
        num_start: int = 10,
        init_strategy: str = "lhs_logit",  # 新增：选择初始化策略
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = "Calibrate"
        
        # 保存参数以备查
        self.num_basins = num_basins
        self.ny = ny
        self.num_start = num_start
        self.device = device

        # 根据配置选择初始化方法
        self.params = self._initialize_params(init_strategy)

    def _initialize_params(self, strategy: str) -> nn.Parameter:
        """
        根据策略生成初始参数
        Shape: (num_basins, ny, num_start)
        """
        sampler = qmc.LatinHypercube(d=self.ny)
        basin_samples = []
        for _ in range(self.num_basins):
            # shape: (num_start, ny)
            s = sampler.random(n=self.num_start)
            basin_samples.append(s)
        # stack: (num_basins, num_start, ny)
        sample_np = np.stack(basin_samples, axis=0)

        # 转为 Tensor: (num_basins, num_start, ny) -> (num_basins, ny, num_start)
        u = torch.from_numpy(sample_np).float().to(self.device).transpose(1, 2)

        # 边界保护 (防止 Logit 溢出)
        u = u * 0.9 + 0.05

        # Logit 变换 (Inverse Sigmoid)
        init_val = torch.log(u / (1 - u))

        print(f"[Calibrate] Initialized with per-basin Latin Hypercube Sampling (LHS) + Logit With {self.num_start} Starts.")
        return nn.Parameter(init_val)


    @classmethod
    def build_by_config(cls, config: dict, device: str = "cpu"):
        # 从 config 中获取流域总数，如果没有则默认 559
        # 建议在 config 中增加 'num_basins' 字段
        n_basins = config.get("num_basins", 559) 
        
        # 获取初始化策略，默认为 lhs_logit
        init_strat = config.get("init_strategy", "lhs_logit")
        
        ny = config["ny"]
        nmul_cfg = config.get("nmul", 16)
        num_start = compute_nmul(ny, multiplier=nmul_cfg)

        return cls(
            nx=config["nx2"],
            ny=ny,
            num_basins=n_basins,
            num_start=num_start,
            init_strategy=init_strat,
            device=device,
        )

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        batch_indices = x['batch_sample']
        cur_params = self.params[batch_indices]
        return None, torch.sigmoid(cur_params)