import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplHalf1(DplUHBase):
    """ 
    GR4J UH1 (Half Bell Curve) 
    对应 MATLAB: uh_1_half
    """
    def get_weights(self, params):
        # params: d_base (Batch, 1)
        d_base = params
        
        # 保护机制: d_base 极小值
        d_base = torch.clamp(d_base, min=1e-3)
        
        # 计算 S-Curve
        # ratio: (1,1,L) / (B,1,1) -> (B,1,L)
        ratio = self.t_idx.to(d_base.device) / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0).pow(2.5)
        
        # 差分计算 UH
        # UH(t) = SH(t) - SH(t-1)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights