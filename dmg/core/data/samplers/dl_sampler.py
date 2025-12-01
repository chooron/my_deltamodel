from typing import Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.data.samplers.base import BaseSampler


class DlSampler(BaseSampler):
    """Deep Learning data sampler for pure neural network prediction.
    
    This sampler is designed for sequence-to-one prediction tasks,
    using historical data (e.g., 365 days) to predict future data (e.g., 1 day).
    
    支持两种 batch 模式:
    1. batch_size = 流域数量 (每次采样一个时间点)
    2. batch_size = 流域数量 * n_time_batch (每次采样多个时间点，加速训练)
    
    Parameters
    ----------
    config
        Configuration dictionary.
    """
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = config['device']
        # 历史序列长度（用于预测的历史天数）
        self.seq_len = config.get('seq_len', 365)
        # 预测长度（预测未来多少天）
        self.pred_len = config.get('pred_len', 1)
        # 每个batch采样的时间步数量（默认100）
        self.n_time_batch = config.get('n_time_batch', 100)

    def load_data(self):
        """Custom implementation for loading data."""
        print("Loading data...")

    def preprocess_data(self):
        """Custom implementation for preprocessing data."""
        print("Preprocessing data...")

    def select_subset(
        self,
        x: Union[torch.Tensor, NDArray[np.float32]],
        i_grid: NDArray[np.float32],
        i_t: Optional[NDArray[np.float32]] = None,
        c: Optional[NDArray[np.float32]] = None,
        tuple_out: bool = False,
        has_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Select a subset of input tensor.
        
        Parameters
        ----------
        x
            Input tensor with shape (time, grid, features) or (grid, features).
        i_grid
            Grid/basin indices to select.
        i_t
            Time indices to select (start of sequence).
        c
            Static attributes (optional).
        tuple_out
            If True, return tuple of (x_tensor, c_tensor).
        has_grad
            If True, enable gradient computation.
            
        Returns
        -------
        torch.Tensor
            Selected subset tensor.
        """
        batch_size, nx = len(i_grid), x.shape[-1]

        # Handle time indexing and create an empty tensor for selection
        if i_t is not None:
            # 创建空张量用于存储选择的数据
            x_tensor = torch.zeros(
                [self.seq_len, batch_size, nx],
                device=self.device,
                requires_grad=has_grad,
            )
            for k in range(batch_size):
                # 选择从 i_t[k] 开始的 seq_len 长度的序列
                x_tensor[:, k:k + 1, :] = x[
                    i_t[k]:i_t[k] + self.seq_len,
                    i_grid[k]:i_grid[k] + 1,
                    :
                ]
        else:
            if x.ndim == 3:
                x_tensor = x[:, i_grid, :].float().to(self.device)
            else:
                x_tensor = x[i_grid, :].float().to(self.device)

        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(self.device)
            c_tensor = c_tensor[i_grid].unsqueeze(1).repeat(1, self.seq_len, 1)
            if tuple_out:
                return (x_tensor, c_tensor)
            return torch.cat((x_tensor, c_tensor), dim=2)

        return x_tensor

    def select_target(
        self,
        target: Union[torch.Tensor, NDArray[np.float32]],
        i_grid: NDArray[np.float32],
        i_t: NDArray[np.float32],
    ) -> torch.Tensor:
        """Select target values for prediction.
        
        Parameters
        ----------
        target
            Target tensor with shape (time, grid, features).
        i_grid
            Grid/basin indices to select.
        i_t
            Time indices (start of input sequence).
            
        Returns
        -------
        torch.Tensor
            Selected target tensor with shape (pred_len, batch_size, features).
        """
        batch_size = len(i_grid)
        n_features = target.shape[-1]
        
        target_tensor = torch.zeros(
            [self.pred_len, batch_size, n_features],
            device=self.device,
        )
        
        for k in range(batch_size):
            # 目标是输入序列之后的 pred_len 天
            target_start = i_t[k] + self.seq_len
            target_tensor[:, k:k + 1, :] = target[
                target_start:target_start + self.pred_len,
                i_grid[k]:i_grid[k] + 1,
                :
            ]
        
        return target_tensor

    def get_training_sample(
        self,
        dataset: dict[str, NDArray[np.float32]],
        ngrid_train: int,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        """Generate a training batch with multiple time steps.
        
        采样 ngrid_train * n_time_batch 个样本，每个样本是一个
        (seq_len, features) 的序列，预测 (pred_len, 1) 的目标。
        
        实际 batch_size = 流域数量 * 时间步数量
        
        Parameters
        ----------
        dataset
            Dictionary containing training data arrays.
        ngrid_train
            Number of training grids/basins.
        nt
            Total number of time steps.
            
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing batched training data.
            输出形状: (seq_len, ngrid * n_time_batch, features)
        """
        n_basins = self.config['train']['batch_size']
        n_time = self.n_time_batch
        total_batch = n_basins * n_time
        
        # 生成随机索引
        # 为每个 (basin, time) 组合生成索引
        max_t = nt - self.seq_len - self.pred_len
        
        # 随机选择流域和时间点
        i_grid = np.random.randint(0, ngrid_train, size=total_batch)
        i_t = np.random.randint(0, max_t, size=total_batch)
        
        return {
            'xc_nn_norm': self.select_subset(
                dataset['xc_nn_norm'], i_grid, i_t, has_grad=False
            ),
            'c_nn': dataset['c_nn'][i_grid],
            'c_nn_norm': dataset['c_nn_norm'][i_grid],
            'target': self.select_target(dataset['target'], i_grid, i_t),
            'batch_sample': i_grid,
        }

    def get_validation_sample(
        self,
        dataset: dict[str, torch.Tensor],
        i_s: int,
        i_e: int,
    ) -> dict[str, torch.Tensor]:
        """Generate batch for model validation/inference.
        
        For validation, we typically process all basins sequentially,
        using the full time series.
        
        Parameters
        ----------
        dataset
            Dictionary containing validation data tensors.
        i_s
            Start index for basin selection.
        i_e
            End index for basin selection.
            
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing batched validation data.
        """
        return {
            key: (
                value[:, i_s:i_e, :] if value.ndim == 3 else value[i_s:i_e, :]
            ).to(dtype=torch.float32, device=self.device)
            for key, value in dataset.items()
        }

    def create_sequences(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sliding window sequences for the entire dataset.
        
        This is useful for creating a complete dataset for validation
        or when you want to process all possible sequences.
        
        Parameters
        ----------
        x
            Input tensor with shape (time, grid, features).
        target
            Target tensor with shape (time, grid, features).
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (input_sequences, target_sequences).
            input_sequences: (n_sequences, seq_len, grid, features)
            target_sequences: (n_sequences, pred_len, grid, features)
        """
        nt = x.shape[0]
        n_sequences = nt - self.seq_len - self.pred_len + 1
        
        x_sequences = []
        target_sequences = []
        
        for i in range(n_sequences):
            x_sequences.append(x[i:i + self.seq_len])
            target_sequences.append(
                target[i + self.seq_len:i + self.seq_len + self.pred_len]
            )
        
        return torch.stack(x_sequences), torch.stack(target_sequences)
