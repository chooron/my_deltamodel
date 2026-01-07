import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplTri4(DplUHBase):
    """
    Unit Hydrograph 4: Full Triangle (Linear Rise & Fall)
    对应 MATLAB: uh_4_full(d_base, delta_t)
    
    物理形状:
    等腰三角形，峰值位于 d_base / 2 处。
    S-Curve 解析解:
    - Phase 1 (t <= d/2): 2 * (t/d)^2
    - Phase 2 (t > d/2):  1 - 2 * (1 - t/d)^2
    """
    
    def get_weights(self, params):
        # 1. 预处理参数
        d_base = torch.clamp(params, min=1e-3)
        
        # 2. 计算归一化时间比率 r = t / d_base
        ratio = self.t_idx.to(d_base.device) / d_base.unsqueeze(-1)
        
        # 3. 计算 S-Curve (分段函数)
        # ----------------------------------------------------
        # Phase 1: 上升段 (ratio <= 0.5)
        # S = 2 * r^2
        s1 = 2.0 * ratio.pow(2)
        
        # Phase 2: 下降段 (ratio > 0.5)
        # S = 1 - 2 * (1 - r)^2
        # 注意: 当 ratio > 1.0 时，(1-r)^2 会增大导致 S 减小，
        # 所以必须 clamp (1-ratio) 最小为 0，确保 ratio > 1 时 S 恒为 1.0
        term2 = torch.clamp(1.0 - ratio, min=0.0)
        s2 = 1.0 - 2.0 * term2.pow(2)
        
        # 组合两段
        s_curve = torch.where(ratio <= 0.5, s1, s2)
        
        # 再次截断 (消除浮点误差，确保不超过1)
        s_curve = torch.clamp(s_curve, max=1.0)
        
        # 4. 差分计算权重 UH(t) = S(t) - S(t-1)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights