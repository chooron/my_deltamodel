import torch
import torch.nn.functional as F

# 模拟你的数据
batch_size = 4
time_steps = 100
max_lag = 5

flux_in = torch.randn(batch_size, time_steps)
params = torch.rand(batch_size, 1) * 4 + 0.5  # [0.5, 4.5]

# 模拟权重生成
raw_weights = torch.randn(batch_size, 1, max_lag).abs()
sum_w = raw_weights.sum(dim=-1, keepdim=True)
norm_weights = raw_weights / (sum_w + 1e-6)
flipped_weights = torch.flip(norm_weights, dims=[-1])

print('flipped_weights shape:', flipped_weights.shape)

# 你的卷积方式
x = flux_in.view(1, batch_size, time_steps)
padd = max_lag - 1

print('x shape:', x.shape)
print('weight shape:', flipped_weights.shape)

try:
    flux_out = F.conv1d(input=x, weight=flipped_weights, groups=batch_size, padding=padd)
    print('conv output shape:', flux_out.shape)
except Exception as e:
    print('Error:', e)