"""诊断 NaN 来源：分解 loss 各组件，检查梯度和中间值"""
import os, sys
import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg import ModelHandler
from dmg.core.utils import import_data_loader, import_trainer, set_randomseed
from project.blend_formula import load_config

CONFIG_PATH = r'conf/config_dblend_v1.yaml'
config = load_config(CONFIG_PATH)
config['mode'] = 'train'
set_randomseed(config['random_seed'])

model = ModelHandler(config, verbose=False)
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)
trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(config, model, train_dataset=data_loader.train_dataset, verbose=False)

# 用 trainer 内部逻辑取 batch
from dmg.core.data.data import create_training_grid
n_samples, n_minibatch, n_timesteps = create_training_grid(
    data_loader.train_dataset["xc_nn_norm"], config
)
sample = trainer.sampler.get_training_sample(
    trainer.train_dataset, n_samples, n_timesteps
)

print("=" * 60)
print("Step 1: Forward pass")
print("=" * 60)
output = model(sample)

# 检查 model output 中的 balance 项
print("\n--- Balance tensors in output ---")
for k, v in model.output_dict.items():
    for sub_k, sub_v in v.items():
        if 'balance' in sub_k:
            print(f"  {sub_k}: shape={sub_v.shape}, "
                  f"min={sub_v.min().item():.4f}, max={sub_v.max().item():.4f}, "
                  f"mean={sub_v.mean().item():.4f}, std={sub_v.std().item():.4f}, "
                  f"nan={torch.isnan(sub_v).sum().item()}, inf={torch.isinf(sub_v).sum().item()}")
        if sub_k == 'streamflow':
            print(f"  {sub_k}: shape={sub_v.shape}, "
                  f"min={sub_v.min().item():.4f}, max={sub_v.max().item():.4f}, "
                  f"mean={sub_v.mean().item():.4f}")

# 手动计算 loss 分解
print("\n" + "=" * 60)
print("Step 2: Loss breakdown")
print("=" * 60)

criterion = model.loss_func
pred = list(model.output_dict.values())[0]['streamflow']
obs = sample['target']

# KGE 部分
prediction, target = criterion._format(pred, obs)
mask = ~torch.isnan(target)
target_filled = torch.nan_to_num(target, nan=0.0)
pred_masked = prediction * mask
target_masked = target_filled * mask
count = mask.sum(dim=0).clamp(min=1.0)

mean_p = pred_masked.sum(dim=0) / count
mean_t = target_masked.sum(dim=0) / count

dev_p = (prediction - mean_p.unsqueeze(0)) * mask
dev_t = (target_filled - mean_t.unsqueeze(0)) * mask
var_p = (dev_p ** 2).sum(dim=0) / count
var_t = (dev_t ** 2).sum(dim=0) / count
std_p = torch.sqrt(var_p + criterion.stability_eps)
std_t = torch.sqrt(var_t + criterion.stability_eps)

num = (dev_p * dev_t).sum(dim=0)
den = torch.sqrt((dev_p ** 2).sum(dim=0) + criterion.stability_eps) * torch.sqrt((dev_t ** 2).sum(dim=0) + criterion.stability_eps)
r = num / (den + criterion.stability_eps)
beta = mean_p / (mean_t + criterion.stability_eps)
gamma = std_p / (std_t + criterion.stability_eps)

kge_val = 1.0 - torch.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2)
loss_kge = (1.0 - kge_val).sum()

print(f"  KGE loss (sum over batch): {loss_kge.item():.4f}")
print(f"  KGE per basin (mean): {(1.0 - kge_val).mean().item():.4f}")
print(f"  r: min={r.min().item():.4f}, max={r.max().item():.4f}")
print(f"  beta: min={beta.min().item():.4f}, max={beta.max().item():.4f}")
print(f"  gamma: min={gamma.min().item():.4f}, max={gamma.max().item():.4f}")

# Balance 部分
model_output = list(model.output_dict.values())[0]
loss_balance = 0.0
balance_count = 0
for proc_name, n_options in criterion.process_options.items():
    for j in range(n_options):
        key = f"balance_{proc_name}_{j}"
        if key in model_output:
            nd = model_output[key]
            if torch.isnan(nd).any() or torch.isinf(nd).any():
                print(f"  [SKIP] {key} has NaN/Inf")
                continue
            comp = (nd ** 2).mean().item()
            print(f"  {key}: loss_component={comp:.4f}")
            loss_balance += comp
            balance_count += 1

print(f"\n  Balance loss (sum of components): {loss_balance:.4f}")
print(f"  Balance count: {balance_count}")

w_kge = criterion.w_kge
w_balance = criterion.w_balance
total = w_kge * loss_kge.item() + w_balance * loss_balance
print(f"\n  w_kge={w_kge}, w_balance={w_balance}")
print(f"  Total = {w_kge}*{loss_kge.item():.2f} + {w_balance}*{loss_balance:.2f} = {total:.2f}")

# 反向传播检查梯度
print("\n" + "=" * 60)
print("Step 3: Gradient check after backward")
print("=" * 60)

loss = model.calc_loss(sample)
print(f"  Actual loss from calc_loss: {loss.item():.4f}")
loss.backward()

max_grad = 0.0
max_grad_name = "(none)"
nan_params = 0
all_params = model.get_parameters()
# get_parameters returns a list of tensors, use model_dict for names
for name, sub_model in model.model_dict.items():
    for pname, p in sub_model.named_parameters():
        if p.grad is not None:
            g = p.grad
            gmax = g.abs().max().item()
            if gmax > max_grad:
                max_grad = gmax
                max_grad_name = f"{name}.{pname}"
            if torch.isnan(g).any():
                nan_params += 1
                print(f"  [NaN GRAD] {name}.{pname}")

print(f"  Max gradient: {max_grad:.4f} in {max_grad_name}")

# 打印 top-5 梯度最大的参数
grad_list = []
for name, sub_model in model.model_dict.items():
    for pname, p in sub_model.named_parameters():
        if p.grad is not None:
            grad_list.append((f"{name}.{pname}", p.grad.abs().max().item(), p.grad.norm().item()))
grad_list.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Total params with grad: {len(grad_list)}")
print("  Top-10 largest gradients:")
for nm, gmax, gnorm in grad_list[:10]:
    print(f"    {nm}: max={gmax:.4f}, norm={gnorm:.4f}")
print(f"  Params with NaN gradients: {nan_params}")

# Clip 后
torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
print(f"  After clip_grad_norm(1.0):")
max_grad_after = 0.0
for name, sub_model in model.model_dict.items():
    for pname, p in sub_model.named_parameters():
        if p.grad is not None:
            gmax = p.grad.abs().max().item()
            if gmax > max_grad_after:
                max_grad_after = gmax
print(f"  Max gradient after clip: {max_grad_after:.6f}")
