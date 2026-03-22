import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplExp5(DplUHBase):
    """
    Unit Hydrograph 5: Half Triangle (Exponential Decay)
    对应 MATLAB: uh_5_half(d_base, delta_t)
    
    物理形状:
    指数衰减曲线 y = exp(-x)，将时间 t=[0, d_base] 映射到 x=[0, 7]。
    S-Curve 解析解: S(t) = 1 - exp( -t * 7 / d_base )
    """
    
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)

        # IHACRES 的 MARRMoT 单位线按整步边界积分，不能使用基类的半步采样。
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        scale_factor = 7.0 / d_base.unsqueeze(-1)
        scaled_t = t_idx * scale_factor
        clamped_t = torch.clamp(scaled_t, max=7.0)
        s_curve = 1.0 - torch.exp(-clamped_t)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights
