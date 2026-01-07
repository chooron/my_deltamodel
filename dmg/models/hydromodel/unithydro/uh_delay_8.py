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
        # 1. 获取延时参数 t_delay
        t_delay = params
        
        # 2. 约束 t_delay
        # 必须 >= 0，且不能超过 max_lag - 1 (否则核移出边界)
        # 实际使用中通常 t_delay 不会非常大
        t_delay = torch.clamp(t_delay, min=0.0)
        
        # 3. 计算距离中心的绝对差值
        # 注意: t_idx=[1,2,3...] 对应 Lag=[0,1,2...]
        # 所以 Lag t 的位置在 t_idx = t + 1
        # 目标中心 center = t_delay + 1
        center = t_delay + 1.0
        
        # 广播计算: |t_idx - center|
        # (1, 1, L) - (B, 1, 1) -> (B, 1, L)
        dist = torch.abs(self.t_idx.to(t_delay.device) - center.unsqueeze(-1))
        
        # 4. 线性插值核 (Triangular Kernel / Linear Interpolation)
        # 公式: max(0, 1 - dist)
        # 当 dist=0 (正中) -> 1
        # 当 dist=0.2 -> 0.8
        # 当 dist=0.8 -> 0.2
        # 当 dist>=1 -> 0
        weights = F.relu(1.0 - dist)
        
        # 理论上这个 Kernel 的 sum 恒等于 1 (只要 peak 不移出边界)
        # 但基类 DplUHBase 会再次执行归一化，双重保险
        return weights