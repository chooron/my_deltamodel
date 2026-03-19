import torch.nn as nn

class DplIdentity0(nn.Module):
    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()

    def forward(self, flux_in, params):
        return flux_in
        