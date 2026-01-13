import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplGamma6(DplUHBase):
    """
    Unit Hydrograph 6: Gamma Distribution (Nash Cascade)
    对应 MATLAB: uh_6_gamma(n, k, delta_t)
    
    物理含义:
    n 个线性水库串联的解析解。
    
    参数要求:
    params: (Batch, 2) 
            - params[:, 0] -> n (形状参数/水库个数)
            - params[:, 1] -> k (尺度参数/滞后时间)
    """
    
    def get_weights(self, params):
        # 1. 解析参数
        # 确保输入是 (Batch, 2)
        if params.shape[-1] != 2:
            raise ValueError("DplGamma6 需要 2 个参数 (n, k)，请确保 params 形状为 (Batch, 2)")
            
        n = params[:, 0:1] # Shape parameter
        k = params[:, 1:2] # Scale parameter (delay)
        
        # 2. 参数安全性约束
        # n 和 k 都必须 > 0，且为了数值稳定，Gamma函数对极小值敏感，给个 safe bound
        n = torch.clamp(n, min=0.1, max=20.0) # n 过大容易导致梯度爆炸，通常水文里 < 10
        k = torch.clamp(k, min=1e-3)
        
        # 3. 计算 Gamma CDF (S-Curve)
        # S(t) = gammainc(n, t/k)
        # torch.special.gammainc 计算的是正则化下不完全 Gamma 函数 P(a, x)
        # 数学定义: 1/Gamma(a) * integral(t^(a-1) * e^-t) from 0 to x
        # 这里的 x 对应 t_idx / k
        
        # 广播计算: (1,1,L) / (B,1,1) -> (B,1,L)
        x_val = self.t_idx.to(n.device) / k.unsqueeze(-1)
        
        # 注意: gammainc 的参数顺序是 (a, x) -> (n, t/k)
        s_curve = torch.special.gammainc(n.unsqueeze(-1), x_val)
        
        # 4. 差分计算权重
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights