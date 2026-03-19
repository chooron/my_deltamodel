# TwoStageTrainer 使用说明

## 问题诊断

### 当前问题
在 `run_model.py` 中，**训练代码被注释掉了**（第36行），导致模型没有经过训练就直接进入评估阶段。因此权重保持初始状态（均匀分布 0.5, 0.5）。

```python
# trainer.train()  # ← 这行被注释了！
print(f"Training complete. Model saved to \n{config['model_path']}")
```

### 根本原因
1. **没有执行训练**: `trainer.train()` 被注释，模型参数和权重都没有更新
2. **评估阶段加载的是未训练模型**: 测试阶段创建了新的 ModelHandler，但由于没有训练过的 checkpoint，加载的是初始化状态
3. **Stage 默认值**: DiffBlendV2 初始化时 `self.stage = 1`，在 stage=1 时权重固定为均匀分布

## 解决方案

**取消注释训练代码**:
```python
trainer.train()  # 启用训练
print(f"Training complete. Model saved to \n{config['model_path']}")
```

---

## TwoStageTrainer 工作原理

### 架构设计

TwoStageTrainer 专为 DiffBlendV2 模型设计，实现两阶段训练策略：

```
Stage 1 (预训练)          Stage 2 (权重精调)
─────────────────         ─────────────────
固定均匀权重              学习流域特异性权重
训练物理参数              差异化学习率
                          Gumbel-Softmax 温度退火
```

### Stage 1: 预训练物理参数

**目标**: 在均匀权重下学习稳定的物理参数

**机制**:
- 冻结 `process_weight_logits` head（权重预测网络）
- 权重固定为均匀分布: `w = [1/n, 1/n, ..., 1/n]`
- 只训练 backbone + phy_head + rout_head
- 学习率: `stage1_lr` (默认 1e-3)

**代码实现**:
```python
# dmg/trainers/two_stage_trainer.py:212-221
def _init_stage1_optimizer(self):
    nn_model = self._get_nn_model()

    # 冻结 weight head
    for param in nn_model.heads["process_weight_logits"].parameters():
        param.requires_grad_(False)

    trainable = [p for p in nn_model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=self.stage1_lr)
```

**物理模型行为**:
```python
# project/blend_formula/models/diff_blend_v2.py:169-171
if self.stage == 1:
    uniform = torch.ones(batch_size, n_opt, device=self.device) / n_opt
    w[proc] = uniform
```

### Stage 2: 权重精调

**目标**: 学习流域特异性的过程权重

**机制**:
- 解冻 `process_weight_logits` head
- 差异化学习率:
  - 物理参数: `stage2_param_lr` (默认 1e-5，小学习率保持稳定)
  - 权重参数: `stage2_weight_lr` (默认 1e-3，大学习率快速学习)
- Gumbel-Softmax 温度退火: `tau_start → tau_end`
- 权重熵监控: 防止权重坍塌到单一选项

**代码实现**:
```python
# dmg/trainers/two_stage_trainer.py:223-241
def _init_stage2_optimizer(self):
    nn_model = self._get_nn_model()

    # 解冻 weight head
    for param in nn_model.heads["process_weight_logits"].parameters():
        param.requires_grad_(True)

    weight_head_params = list(
        nn_model.heads["process_weight_logits"].parameters()
    )
    other_params = [
        p for n, p in nn_model.named_parameters()
        if "process_weight_logits" not in n
    ]
    return torch.optim.Adam([
        {"params": other_params,       "lr": self.stage2_param_lr},
        {"params": weight_head_params, "lr": self.stage2_weight_lr},
    ])
```

**物理模型行为**:
```python
# project/blend_formula/models/diff_blend_v2.py:172-176
else:  # stage == 2
    logits = raw_w[:, idx: idx + n_opt]  # 每个流域不同的 logits
    w[proc] = activate_weights(
        logits, self.weight_method, self.tau, self.training
    )
```

### Tau 温度调度

**目的**: 控制 Gumbel-Softmax 的探索-收敛平衡

**调度策略**:
```
Warmup 阶段 (0 → warmup_epochs):
  tau = tau_start (高温，鼓励探索)

衰减阶段 (warmup_epochs → total_epochs):
  tau = tau_start * exp(-decay_rate * t)
  最终收敛到 tau_end (低温，促进收敛)
```

**代码实现**:
```python
# dmg/trainers/two_stage_trainer.py:63-69
def get_tau(self, epoch: int) -> float:
    if epoch < self.warmup_epochs:
        return self.tau_start
    # 指数衰减
    t = epoch - self.warmup_epochs
    return self.tau_start * math.exp(-self.decay_rate * t)
```

**效果**:
- 高温 (tau=5.0): softmax 接近均匀分布，探索所有选项
- 低温 (tau=0.3): softmax 接近 one-hot，收敛到最优选项

---

## 配置参数详解

### 基础配置 (config.yaml)

```yaml
trainer: TwoStageTrainer

train:
  epochs: 30  # 总 epochs (会被 two_stage 覆盖)

two_stage:
  verbose: true              # 是否打印详细日志
  stage1_epochs: 10          # Stage 1 训练轮数
  stage2_epochs: 20          # Stage 2 训练轮数
  stage1_lr: 1e-3            # Stage 1 学习率
  stage2_param_lr: 1e-5      # Stage 2 物理参数学习率
  stage2_weight_lr: 1e-3     # Stage 2 权重学习率
  tau_start: 5.0             # 初始温度
  tau_end: 0.3               # 最终温度
  stage2_warmup_epochs: 20   # Stage 2 温度 warmup 轮数
  balance_loss_weight: 0.01  # 平衡损失权重
  log_interval: 5            # 权重熵监控间隔
```

### 参数调优建议

#### 1. Epoch 分配
```yaml
# 快速实验 (调试)
stage1_epochs: 5
stage2_epochs: 10

# 标准训练
stage1_epochs: 10
stage2_epochs: 20

# 充分训练
stage1_epochs: 50
stage2_epochs: 80
```

#### 2. 学习率
```yaml
# 保守策略 (稳定但慢)
stage1_lr: 5e-4
stage2_param_lr: 1e-6
stage2_weight_lr: 5e-4

# 标准策略
stage1_lr: 1e-3
stage2_param_lr: 1e-5
stage2_weight_lr: 1e-3

# 激进策略 (快速但可能不稳定)
stage1_lr: 5e-3
stage2_param_lr: 1e-4
stage2_weight_lr: 5e-3
```

#### 3. 温度调度
```yaml
# 强探索 (权重多样性)
tau_start: 10.0
tau_end: 0.5
stage2_warmup_epochs: 30

# 标准
tau_start: 5.0
tau_end: 0.3
stage2_warmup_epochs: 20

# 快速收敛 (可能过早坍塌)
tau_start: 3.0
tau_end: 0.1
stage2_warmup_epochs: 10
```

**警告**: 如果 `stage2_warmup_epochs >= stage2_epochs`，温度将始终保持 `tau_start`，不会衰减！

---

## 训练流程示例

### 完整训练脚本

```python
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg import ModelHandler
from dmg.core.utils import import_data_loader, import_trainer, set_randomseed
from project.blend_formula import load_config

# 1. 加载配置
CONFIG_PATH = 'conf/config_dblend_v2.yaml'
config = load_config(CONFIG_PATH)
config['mode'] = 'train'
set_randomseed(config['random_seed'])

# 2. 初始化模型和数据
model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)

# 3. 创建训练器并训练
trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(
    config,
    model,
    train_dataset=data_loader.train_dataset,
    verbose=True
)

# ★ 关键: 执行训练
history = trainer.train()

print(f"Training complete. Model saved to {config['model_path']}")
print(f"Stage 1 final loss: {history['stage1_loss'][-1]:.6f}")
print(f"Stage 2 final loss: {history['stage2_loss'][-1]:.6f}")

# 4. 评估
config['mode'] = 'test'
set_randomseed(config['random_seed'])

model = ModelHandler(config, verbose=True)
data_loader = data_loader_cls(config, test_split=True, overwrite=False)

trainer = trainer_cls(
    config,
    model,
    eval_dataset=data_loader.eval_dataset,
    verbose=True,
)

trainer.evaluate()
print(f"Metrics saved to {config['out_path']}")
```

### 训练日志示例

```
===== Stage 1: 预训练物理参数 (10 epochs) =====
[Stage1] Epoch 1: loss=0.856234 (45.2s, 3555Mb GPU)
[Stage1] Epoch 2: loss=0.723451 (44.8s, 3555Mb GPU)
...
[Stage1] Epoch 10: loss=0.234567 (45.1s, 3555Mb GPU)
Checkpoint saved: ./checkpoints/two_stage_s1_ep10.pt

===== Stage 2: 权重精调 (20 epochs) =====
[Stage2] tau=5.000
[Stage2] Epoch 1: loss=0.245678 (46.3s, 3555Mb GPU)
...
[Stage2] Epoch 5: loss=0.198765 (46.1s, 3555Mb GPU)
[Stage2] tau=5.000
[Epoch   5] rainsnow:      w=[0.45, 0.35, 0.20] H=1.023/1.099(93%)
[Epoch   5] snowbal:       w=[0.40, 0.38, 0.22] H=1.045/1.099(95%)
[Epoch   5] infiltration: w=[0.42, 0.33, 0.25] H=1.067/1.099(97%)
...
[Stage2] Epoch 20: loss=0.123456 (46.2s, 3555Mb GPU)
[Stage2] tau=0.312
[Epoch  20] rainsnow:      w=[0.78, 0.15, 0.07] H=0.678/1.099(62%)
[Epoch  20] snowbal:       w=[0.12, 0.82, 0.06] H=0.623/1.099(57%)
...
===== 两阶段训练完成 =====
```

---

## 权重熵监控

### 目的
防止权重过早坍塌到单一选项（权重退化）

### 指标解读

```
[Epoch  20] rainsnow:      w=[0.78, 0.15, 0.07] H=0.678/1.099(62%)
                              ↑                    ↑     ↑     ↑
                              权重分布              熵值  最大熵 百分比
```

**熵值 (H)**:
- H = -Σ(w_i * log(w_i))
- 衡量权重分布的不确定性

**最大熵 (H_max)**:
- H_max = log(n_options)
- 均匀分布时的熵值

**百分比**:
- H / H_max * 100%
- 100%: 完全均匀分布（探索）
- 0%: 完全坍塌到单一选项（收敛）

### 健康指标

```
训练初期 (Stage 2 前期):
  期望: 80-100% (充分探索)

训练中期:
  期望: 50-80% (逐渐收敛)

训练后期:
  期望: 30-60% (收敛但保持多样性)

警告信号:
  < 20%: 可能过早坍塌，考虑增大 tau_start 或延长 warmup
  > 95% (后期): 未收敛，考虑减小 tau_end 或增加 epochs
```

---

## Checkpoint 管理

### 保存策略

```python
# 自动保存
save_epoch = config["train"].get("save_epoch", 5)

# Stage 1: 每 save_epoch 保存一次
two_stage_s1_ep5.pt
two_stage_s1_ep10.pt

# Stage 2: 每 save_epoch 保存一次
two_stage_s2_ep5.pt
two_stage_s2_ep20.pt
```

### Checkpoint 内容

```python
{
    "model": phy_model.state_dict(),           # DiffBlendV2 状态
    "param_network": nn_model.state_dict(),    # MultiHeadNetV2 状态
    "epoch": epoch,
    "stage": stage,
    "loss": loss,
}
```

### 加载 Checkpoint

```python
# 测试阶段自动加载
config['mode'] = 'test'
config['test']['test_epoch'] = 30  # 加载第 30 epoch 的模型

trainer = TwoStageTrainer(config, model, eval_dataset=data_loader.eval_dataset)
# 自动调用 load_test_states()
```

---

## 常见问题

### Q1: 权重始终是 0.5, 0.5 怎么办？

**原因**:
1. 训练代码被注释 (`trainer.train()`)
2. Stage 1 阶段权重固定为均匀分布
3. 未进入 Stage 2

**解决**:
- 取消注释 `trainer.train()`
- 确保 `stage2_epochs > 0`
- 检查 checkpoint 是否正确保存和加载

### Q2: Stage 2 权重不更新？

**检查清单**:
1. `stage2_weight_lr` 是否太小？建议 >= 1e-4
2. `stage2_warmup_epochs` 是否 >= `stage2_epochs`？（温度不衰减）
3. 损失函数是否包含权重梯度？检查 `balance_loss_weight`

### Q3: 训练不稳定，出现 NaN？

**解决方案**:
1. 降低学习率（特别是 `stage1_lr`）
2. 增加梯度裁剪: 代码中已有 `clip_grad_norm_(max_norm=1.0)`
3. 检查数据归一化
4. 增大 `tau_start`（减少 Gumbel 噪声）

### Q4: 如何只训练 Stage 2？

```yaml
two_stage:
  stage1_epochs: 0  # 跳过 Stage 1
  stage2_epochs: 50
```

**注意**: 需要预先训练好的物理参数，否则效果不佳。

### Q5: 如何调试权重学习？

```python
# 在训练循环中添加
if epoch % 1 == 0:  # 每个 epoch 监控
    with torch.no_grad():
        sample = self.sampler.get_training_sample(...)
        output = self.model(sample)
        log_weight_entropy(output, epoch)
```

---

## 与 V1 的区别

| 特性 | DiffBlendV1 | DiffBlendV2 |
|------|-------------|-------------|
| 权重来源 | 参数网络输出 | 参数网络输出 |
| 训练策略 | 单阶段 | 两阶段 |
| 权重控制 | 始终学习 | Stage 1 固定，Stage 2 学习 |
| 优化器 | 统一学习率 | 差异化学习率 |
| 温度调度 | 手动设置 | 自动退火 |
| 适用场景 | 简单任务 | 复杂任务，需要稳定训练 |

---

## 总结

TwoStageTrainer 通过两阶段策略解决了端到端训练的稳定性问题：

1. **Stage 1**: 在均匀权重下学习稳定的物理参数基础
2. **Stage 2**: 在稳定基础上精调流域特异性权重

关键设计：
- 差异化学习率（物理参数小，权重大）
- Gumbel-Softmax 温度退火（探索→收敛）
- 权重熵监控（防止退化）

**记住**: 一定要执行 `trainer.train()`！
