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
        if params.shape[-1] != 2:
            raise ValueError("DplGamma6 需要 2 个参数 (n, k)，请确保 params 形状为 (Batch, 2)")

        n = params[:, 0:1]
        k = params[:, 1:2]
        n = torch.clamp(n, min=0.1, max=20.0)
        k = torch.clamp(k, min=1e-3)

        # MATLAB uh_6_gamma 按整步边界 [t-1, t] 积分，使用 t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=n.device, dtype=n.dtype
        ).view(1, 1, -1)

        x_val = t_idx / k.unsqueeze(-1)
        s_curve = torch.special.gammainc(n.unsqueeze(-1), x_val)

        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights