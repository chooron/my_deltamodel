import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplFull2(DplUHBase):
    """
    GR4J UH2 (Full Bell Curve) 单位线
    对应 MATLAB: uh_2_full(d_base, delta_t)
    
    物理含义:
    这是一个对称的 S 曲线，总时长为 2 * d_base。
    - 前半段 (t <= d): 0.5 * (t/d)^2.5
    - 后半段 (t > d): 1 - 0.5 * (2 - t/d)^2.5
    """
    
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)

        # MATLAB uh_2_full 按整步边界 SH(t)-SH(t-1) 计算，使用 t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        ratio = t_idx / d_base.unsqueeze(-1)

        s_part1 = 0.5 * ratio.pow(2.5)
        term_b = torch.clamp(2.0 - ratio, min=0.0)
        s_part2 = 1.0 - 0.5 * term_b.pow(2.5)
        s_curve = torch.where(ratio <= 1.0, s_part1, s_part2)
        s_curve = torch.clamp(s_curve, max=1.0)

        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights