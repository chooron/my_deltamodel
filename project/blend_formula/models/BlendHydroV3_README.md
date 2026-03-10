# BlendHydroV3 - 雪层状态增强版本

## 概述

BlendHydroV3 是 BlendHydroV2 的增强版本，主要改进是为具有融雪模块的水文模型添加了雪层覆盖状态作为门控网络的额外输入。

## 核心改进

### 1. 雪层状态感知

- **HBV模型**: 提取 S1 (snow storage) 作为雪层状态
- **EXPHYDRO模型**: 提取 snow_storage 作为雪层状态
- **SHM模型**: 无雪层模块，雪层状态填充为0
- **HYMOD模型**: 无雪层模块，雪层状态填充为0

### 2. 门控网络输入维度变化

**V2版本**:
```
输入维度 = 3(气象) + Nmul*NumModels(土壤状态) + NumAttributes(流域属性)
```

**V3版本**:
```
输入维度 = 3(气象) + Nmul*NumModels(土壤状态) + Nmul*NumModels(雪层状态) + NumAttributes(流域属性)
```

对于默认配置 (Nmul=1, NumModels=4, NumAttributes=10):
- V2: 3 + 4 + 10 = 17
- V3: 3 + 4 + 4 + 10 = 21

### 3. 新增方法

#### `_normalize_snow_states()`
```python
def _normalize_snow_states(
    self,
    raw_snow_states: torch.Tensor,
    params_dict: Dict[str, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """
    归一化各模型的雪层状态到 [0, 1] 区间

    对于没有雪层模块的模型（SHM, HYMOD），填充0

    Args:
        raw_snow_states: [Time, Grid, Nmul, NumModels] 原始雪层状态值
        params_dict: 各模型参数字典

    Returns:
        norm_snow_states: [Time, Grid, Nmul, NumModels] 归一化后的雪层状态
    """
```

**归一化策略**:
- 由于雪层没有明确的容量参数，使用动态归一化
- 除以时间序列中的最大值 + 100mm (作为基准)
- 裁剪到 [0, 2.0] 范围

### 4. 修改的方法

#### `_unified_timestep_loop()`
- **返回值变化**: 从返回2个值改为返回3个值
  - `outputs`: 各模型的流量输出字典
  - `soil_states_seq`: 土壤状态序列 [Time, Grid, Nmul, NumModels]
  - `snow_states_seq`: 雪层状态序列 [Time, Grid, Nmul, NumModels] (新增)

- **状态记录**:
  - HBV: 记录 S1 (snow storage) 到 snow_states_seq
  - EXPHYDRO: 记录 snow_storage 到 snow_states_seq
  - SHM/HYMOD: snow_states_seq 保持为0

#### `forward()`
- 接收3个返回值: `model_outputs, raw_soil_states_seq, raw_snow_states_seq`
- 分别归一化土壤状态和雪层状态
- 将两种状态特征拼接后输入门控网络

```python
# V3版本的特征拼接
x_dynamic = torch.cat([
    x_nn_norm,              # 气象驱动 [Time, Grid, 3]
    flat_soil_states_feat,  # 土壤状态 [Time, Grid, Nmul*NumModels]
    flat_snow_states_feat   # 雪层状态 [Time, Grid, Nmul*NumModels]
], dim=-1)
```

## 使用方法

### 基本使用

```python
from project.hydro_selection.models.blend_hydro_v3 import BlendHydroV3

config = {
    "warm_up": 365,
    "warm_up_states": True,
    "variables": ["prcp", "tmean", "pet"],
    "nmul": 1,
    "num_attributes": 10,
    "selected_models": ["HBV", "SHM", "EXPHYDRO", "HYMOD"]
}

model = BlendHydroV3(config=config, device=device)
```

### 前向传播

```python
output = model(x_dict, parameters)

# 输出包含:
# - streamflow: 最终融合的流量
# - model_avg_outputs: 各模型的平均产流
# - {MODEL}_weights: 各模型的权重
# - {model}_prerouting: 各模型路由前的产流
# - {model}_streamflow: 各模型路由后的流量
```

## 优势

1. **季节性感知增强**: 门控网络可以感知积雪和融雪过程，在有雪季节更好地调整模型权重
2. **物理一致性**: 雪层状态是重要的水文过程指示器，特别是在寒冷地区
3. **向后兼容**: 对于无雪层模块的模型，自动填充0，不影响原有逻辑
4. **灵活性**: 可以通过配置选择不同的模型组合

## 测试结果

```
✓ Model created successfully: BlendHydroV3
✓ Number of models: 4
✓ Model order: ['HBV', 'SHM', 'EXPHYDRO', 'HYMOD']
✓ Gating input dim: 21 (3气象 + 4土壤 + 4雪层 + 10属性)
✓ Forward pass successful!
✓ All tests passed!
```

## 文件位置

- 模型文件: `/workspace/my_deltamodel/project/hydro_selection/models/blend_hydro_v3.py`
- 测试文件: `/workspace/my_deltamodel/test_blend_hydro_v3.py`

## 与V2的兼容性

V3版本保持了V2的所有功能和接口，只是增加了雪层状态输入。如果需要回退到V2的行为，可以简单地将雪层状态设置为0（这在无雪层模块的模型中是自动的）。

## 未来改进方向

1. 可以考虑为雪层状态添加更精细的归一化策略（例如使用模型参数中的相关阈值）
2. 可以添加配置选项来控制是否使用雪层状态
3. 可以探索其他物理状态（如地下水、河道存储等）作为门控网络输入
