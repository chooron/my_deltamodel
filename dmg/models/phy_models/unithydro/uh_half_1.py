import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplHalf1(DplUHBase):
    """ 
    GR4J UH1 (Half Bell Curve) 
    对应 MATLAB: uh_1_half
    """
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)

        # MATLAB uh_1_half 按整步边界 SH(t)-SH(t-1) 计算，使用 t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        ratio = t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0).pow(2.5)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights