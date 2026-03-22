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
        d_base = torch.clamp(params, min=1e-3)

        # MATLAB uh_3_half 按整步边界 [t-1, t] 积分，使用 t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        ratio = t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0).pow(2.0)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights