import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplDelay8(DplUHBase):
    """
    Unit Hydrograph 8: Pure Delay (No Transformation)
    对应 MATLAB: uh_8_delay(t_delay, delta_t)
    
    物理含义:
    纯平移操作。输入仅在时间上移动，不改变形状。
    如果延迟 t_delay 是小数 (e.g. 3.8)，则流量分配到最近的两个时间步 (Lag 3 和 Lag 4)，
    相当于线性插值。
    
    可微实现:
    使用 ReLU 三角核函数: W = ReLU( 1 - |t_idx - (delay + 1)| )
    """
    
    def get_weights(self, params):
        t_delay = torch.clamp(params, min=0.0)

        # IHACRES 的纯延迟核按整数 lag 位置分配权重，不能使用半步采样。
        t_idx = torch.arange(
            1, self.max_lag + 1, device=t_delay.device, dtype=t_delay.dtype
        ).view(1, 1, -1)

        center = t_delay + 1.0
        dist = torch.abs(t_idx - center.unsqueeze(-1))
        weights = F.relu(1.0 - dist)
        return weights
