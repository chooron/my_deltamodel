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
        d_base = torch.clamp(params, min=1e-3)

        # MATLAB uh_7_uniform 按整步边界计算，使用 t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        ratio = t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights