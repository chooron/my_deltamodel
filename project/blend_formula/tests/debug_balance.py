"""诊断 balance loss 各过程的贡献，分析对 KGE 的影响"""
import os, sys
import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg import ModelHandler
from dmg.core.utils import import_data_loader, import_trainer, set_randomseed
from dmg.core.data.data import create_training_grid
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

n_samples, n_minibatch, n_timesteps = create_training_grid(
    data_loader.train_dataset["xc_nn_norm"], config
)

criterion = model.loss_func
w_kge = criterion.w_kge
w_balance = criterion.w_balance
print(f"Config: w_kge={w_kge}, w_balance={w_balance}")
print(f"Process options: {criterion.process_options}")
print()

# 跑 5 个 batch 取平均
n_batches = 5
kge_losses = []
balance_per_process = {proc: [] for proc in criterion.process_options}
balance_totals = []

for b in range(n_batches):
    sample = trainer.sampler.get_training_sample(
        trainer.train_dataset, n_samples, n_timesteps
    )
    with torch.no_grad():
        _ = model(sample)

    model_output = list(model.output_dict.values())[0]
    pred = model_output['streamflow']
    obs = sample['target']

    # KGE (batch-level, 与 KgeBatchLoss 一致)
    prediction, target = criterion._format(pred, obs)
    mask = ~torch.isnan(target)
    p_sub = prediction[mask]
    t_sub = target[mask]
    mean_p = torch.mean(p_sub)
    mean_t = torch.mean(t_sub)
    std_p = torch.std(p_sub)
    std_t = torch.std(t_sub)
    numerator = torch.sum((p_sub - mean_p) * (t_sub - mean_t))
    denominator = torch.sqrt(torch.sum((p_sub - mean_p)**2) * torch.sum((t_sub - mean_t)**2))
    r = numerator / (denominator + criterion.stability_eps)
    beta = mean_p / (mean_t + criterion.stability_eps)
    gamma = std_p / (std_t + criterion.stability_eps)
    kge_val = 1.0 - torch.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2)
    loss_kge = (1.0 - kge_val).item()
    kge_losses.append(loss_kge)

    # Balance per process
    batch_balance_total = 0.0
    for proc_name, n_options in criterion.process_options.items():
        proc_loss = 0.0
        proc_count = 0
        for j in range(n_options):
            key = f"balance_{proc_name}_{j}"
            if key in model_output:
                nd = model_output[key]
                if torch.isnan(nd).any() or torch.isinf(nd).any():
                    continue
                # smooth_l1_loss (与实际 loss 一致)
                comp = torch.nn.functional.smooth_l1_loss(
                    nd, torch.zeros_like(nd), beta=1.0, reduction="mean"
                ).item()
                proc_loss += comp
                proc_count += 1
        balance_per_process[proc_name].append(proc_loss)
        batch_balance_total += proc_loss
    balance_totals.append(batch_balance_total)

# 汇总
print("=" * 70)
print(f"{'Component':<25} {'Mean Loss':>12} {'Weighted':>12} {'% of Total':>12}")
print("=" * 70)

avg_kge = sum(kge_losses) / n_batches
weighted_kge = w_kge * avg_kge
print(f"{'KGE (sum over batch)':<25} {avg_kge:>12.2f} {weighted_kge:>12.2f}")

avg_balance_total = sum(balance_totals) / n_batches
weighted_balance_total = w_balance * avg_balance_total

print(f"\n{'--- Balance per process ---'}")
for proc_name in criterion.process_options:
    avg = sum(balance_per_process[proc_name]) / n_batches
    weighted = w_balance * avg
    pct = (weighted / (weighted_kge + weighted_balance_total)) * 100 if (weighted_kge + weighted_balance_total) > 0 else 0
    print(f"  {proc_name:<23} {avg:>12.4f} {weighted:>12.4f} {pct:>11.2f}%")

print(f"\n{'Balance Total':<25} {avg_balance_total:>12.4f} {weighted_balance_total:>12.4f}")
total = weighted_kge + weighted_balance_total
print(f"\n{'TOTAL LOSS':<25} {'':>12} {total:>12.2f}")
print(f"\nBalance / Total ratio: {weighted_balance_total / total * 100:.2f}%")
print(f"KGE / Total ratio: {weighted_kge / total * 100:.2f}%")

# 检查各过程 balance tensor 的统计
print("\n" + "=" * 70)
print("Balance tensor statistics (last batch):")
print(f"{'Key':<30} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'|mean|':>8}")
print("-" * 70)
for proc_name, n_options in criterion.process_options.items():
    for j in range(n_options):
        key = f"balance_{proc_name}_{j}"
        if key in model_output:
            nd = model_output[key]
            print(f"  {key:<28} {nd.mean().item():>8.4f} {nd.std().item():>8.4f} "
                  f"{nd.min().item():>8.4f} {nd.max().item():>8.4f} {nd.abs().mean().item():>8.4f}")
