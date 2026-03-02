import numpy as np
import torch
from dmg.models.criterion.nse_aic_batch_loss import NseAicBatchLoss


def run_check():
    # Create historical observations with one grid all-NaN
    y_obs = np.random.rand(30, 4, 1).astype(np.float32)
    y_obs[:, 2, :] = np.nan  # grid index 2 has NaNs in history -> std NaN
    y_obs_t = torch.tensor(y_obs)

    loss_fn = NseAicBatchLoss(config={}, device='cpu', y_obs=y_obs_t)

    # Create a batch where the grid with NaN history has valid current obs
    n_time = 5
    n_grid = 4
    pred = torch.randn(n_time, n_grid, 1)
    obs = torch.randn(n_time, n_grid, 1)
    obs[:, 2, :] = 0.5  # valid values for previously-NaN grid

    sample_ids = np.arange(n_grid)

    loss = loss_fn(pred, obs, sample_ids=sample_ids)
    print('loss:', loss.item())
    print('isfinite:', torch.isfinite(loss).item())


if __name__ == '__main__':
    run_check()
