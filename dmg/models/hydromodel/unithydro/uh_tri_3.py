import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplTri3(DplUHBase):
    """
    Unit Hydrograph 3: Half Triangle (Linear)
    对应 MATLAB: uh_3_half(d_base, delta_t)
    
    物理形状:
    直角三角形，权重随时间 t 线性增加，直到 t = d_base。
    S-Curve 公式: S(t) = (t / d_base)^2  (当 t <= d_base)
    """
    
    def get_weights(self, params):
        # 1. 获取参数 d_base (汇流时间)
        d_base = params
        d_base = torch.clamp(d_base, min=1e-3) # 保护除零
        
        # 2. 计算比率 t / d_base
        # self.t_idx 来自基类 [1, 2, ..., max_lag]
        ratio = self.t_idx.to(d_base.device) / d_base.unsqueeze(-1)
        
        # 3. 计算 S-Curve
        # 公式: (t/d)^2
        # clamp(max=1.0) 确保超过 d_base 后 S 保持为 1
        s_curve = torch.clamp(ratio, max=1.0).pow(2.0)
        
        # 4. 差分计算权重 UH(t) = S(t) - S(t-1)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights