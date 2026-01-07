import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplUniform7(DplUHBase):
    """
    Unit Hydrograph 7: Uniform Distribution
    对应 MATLAB: uh_7_uniform(d_base, delta_t)
    
    物理含义:
    在 d_base 时间内，流量均匀分布。
    
    S-Curve 解析解:
    S(t) = t / d_base (当 t < d_base)
    S(t) = 1          (当 t >= d_base)
    这是一个简单的线性斜坡函数 (Linear Ramp)。
    """
    
    def get_weights(self, params):
        # 1. 获取参数 d_base
        d_base = torch.clamp(params, min=1e-3)
        
        # 2. 计算比率 t / d_base
        # (1,1,L) / (B,1,1) -> (B,1,L)
        ratio = self.t_idx.to(d_base.device) / d_base.unsqueeze(-1)
        
        # 3. 计算 S-Curve
        # 线性斜坡，最大值为 1.0
        s_curve = torch.clamp(ratio, max=1.0)
        
        # 4. 差分计算权重
        # UH(t) = S(t) - S(t-1)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights