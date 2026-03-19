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
        """
        计算双边 S-Curve 权重。
        Args:
            params: d_base (Batch, 1) [单位: 时间步]
        """
        # 1. 预处理参数
        # 对应 MATLAB: delay = d_base/delta_t (假设 delta_t=1)
        d_base = torch.clamp(params, min=1e-3)
        
        # 2. 计算比率 ratio = t / d_base
        # shape: (Batch, 1, Lag)
        ratio = self.t_idx.to(d_base.device) / d_base.unsqueeze(-1)
        
        # 3. 计算分段 S-Curve (SH)
        # -----------------------------------------------------------
        # 逻辑分支 A: 前半段 (ratio <= 1.0)
        # MATLAB: if t <= delay; SH = 0.5 * (t/delay)^2.5
        # -----------------------------------------------------------
        s_part1 = 0.5 * ratio.pow(2.5)
        
        # -----------------------------------------------------------
        # 逻辑分支 B: 后半段 (ratio > 1.0)
        # MATLAB: elseif t < 2*delay; SH = 1 - 0.5 * (2 - t/delay)^2.5
        #         elseif t >= 2*delay; SH = 1
        # 
        # 技巧: 当 ratio > 2.0 时，(2 - ratio) 会变成负数，导致 pow(2.5) 产生 NaN。
        # 所以必须先用 clamp(..., min=0) 截断，这样当 ratio > 2 时，项变为 0，SH 自然变为 1.0。
        # 这样就用一个公式同时覆盖了 MATLAB 的 elseif 和 else 两种情况。
        # -----------------------------------------------------------
        term_b = torch.clamp(2.0 - ratio, min=0.0) # 对应 (2 - t/delay)
        s_part2 = 1.0 - 0.5 * term_b.pow(2.5)
        
        # 4. 组合分段函数
        # 使用 torch.where 根据 ratio 的值选择使用 s_part1 还是 s_part2
        s_curve = torch.where(ratio <= 1.0, s_part1, s_part2)
        
        # 再次确保最大不超过 1.0 (消除浮点误差)
        s_curve = torch.clamp(s_curve, max=1.0)
        
        # 5. 差分计算权重 UH(t) = SH(t) - SH(t-1)
        # 使用 Pad 在左侧补 0
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights