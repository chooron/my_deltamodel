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
        # 1. 预处理参数
        d_base = torch.clamp(params, min=1e-3)
        
        # 2. 计算映射后的指数坐标 x
        # 对应 MATLAB: stepsize = 7 / delay
        # x = t * (7 / d_base)
        scale_factor = 7.0 / d_base.unsqueeze(-1)
        scaled_t = self.t_idx.to(d_base.device) * scale_factor
        
        # 3. 实施边界截断 [0, 7]
        # MATLAB: limits(end+1) = 7; 意味着积分上限最大为 7
        # 当 t > d_base 时，scaled_t > 7，我们将其钳位在 7.0
        # 这样 S-Curve 在 d_base 之后保持为 (1 - e^-7) 不变，差分权重自然为 0
        clamped_t = torch.clamp(scaled_t, max=7.0)
        
        # 4. 计算 S-Curve
        # 积分 exp(-x) -> 1 - exp(-x)
        s_curve = 1.0 - torch.exp(-clamped_t)
        
        # 5. 差分计算权重
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights