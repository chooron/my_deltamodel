import torch
import torch.nn as nn
from typing import Union
from scipy.stats import qmc  # 需要安装 scipy: pip install scipy

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
        if strategy == "lhs_logit":
            # --- 方案 A (最推荐): LHS + Logit ---
            # 1. 使用拉丁超立方生成 [0, 1] 样本
            # d=ny (参数维度)
            sampler = qmc.LatinHypercube(d=self.ny)
            
            # 我们需要为 (所有流域 * 所有起点) 生成样本
            total_samples = self.num_basins * self.num_start
            sample_np = sampler.random(n=total_samples)
            
            # 2. 转为 Tensor 并移动到 GPU
            u = torch.from_numpy(sample_np).float().to(self.device)
            
            # 3. Reshape: [Total, ny] -> [num_basins, num_start, ny] -> [num_basins, ny, num_start]
            u = u.view(self.num_basins, self.num_start, self.ny).transpose(1, 2)
            
            # 4. 边界保护 (防止 Logit 溢出)
            u = u * 0.9 + 0.05
            
            # 5. Logit 变换 (Inverse Sigmoid)
            init_val = torch.log(u / (1 - u))
            
            print(f"[Calibrate] Initialized with Latin Hypercube Sampling (LHS) + Logit.")
            return nn.Parameter(init_val)

        elif strategy == "uniform":
            # --- 方案 B: 宽范围均匀分布 ---
            # 生成 [-3, 3] 之间的均匀分布
            # 对应 Sigmoid 后覆盖 [0.047, 0.952]
            init_val = torch.rand(self.num_basins, self.ny, self.num_start, device=self.device) * 6 - 3
            print(f"[Calibrate] Initialized with Broad Uniform Distribution [-3, 3].")
            return nn.Parameter(init_val)

        elif strategy == "normal":
            # --- 方案 C: 原始正态分布 (旧方法) ---
            # 警告：存在中心聚集问题
            print(f"[Calibrate] Initialized with Standard Normal Distribution (Center Biased).")
            return nn.Parameter(
                torch.randn(self.num_basins, self.ny, self.num_start, device=self.device)
            )
        
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cpu"):
        # 从 config 中获取流域总数，如果没有则默认 559
        # 建议在 config 中增加 'num_basins' 字段
        n_basins = config.get("num_basins", 559) 
        
        # 获取初始化策略，默认为 lhs_logit
        init_strat = config.get("init_strategy", "lhs_logit")
        
        return cls(
            nx=config["nx2"],
            ny=config["ny"],
            num_basins=n_basins, 
            num_start=config["nmul"],
            init_strategy=init_strat, # 传入策略
            device=device,
        )

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        batch_indices = x['batch_sample']
        cur_params = self.params[batch_indices]
        return None, torch.sigmoid(cur_params)