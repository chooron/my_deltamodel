---

### 🟢 实验板块一：扩展性测试 (针对 3.2.1 Scalability)

**目标：** 绘制出  (流域数量) 与  (计算耗时) 的关系曲线，证明 CPU 是线性  而 GPU 是近似常量 。

* [ ] **实验 1：变规模耗时对比 (Variable Batch Size Benchmark)**
* **变量：** 流域数量 。建议取点：`[1, 10, 50, 100, 200, 400, 559]`。
* **固定条件：** 选择一个中等复杂度的模型（例如 `m_14` 或 `m_21`），固定时间步长（例如 10年），固定迭代次数（例如 100 epochs）。
* **对比组：**
* **Baseline (CPU):** 传统的串行循环模式 (For-loop over basins)。

即使是用 Python 多进程 (Multiprocessing)，通常也难以达到 GPU 的吞吐量，建议测纯串行作为基准，或者注明 CPU 核心数。

* **Diff-MARRMoT (GPU):** 你的张量化并行版本。


* **记录指标：** “Forward + Backward” 一次完整 Pass 的平均耗时，或完整率定流程的总耗时。
* **预期结果：** 寻找 **Break-even point** (GPU 启动有开销，N=1 时 GPU 可能不如 CPU)，并展示当  增大时 GPU 耗时几乎不涨，而 CPU 线性暴涨。


* [ ] **实验 2：显存占用与硬件利用率监控 (Hardware Saturation Profiling)**
* **目的：** 证明“显存未饱和区间”的论点。
* **操作：** 在运行实验 1 的 GPU 组时，使用 `nvidia-smi` 或 PyTorch Profiler 记录：
* GPU Memory Usage (MB)
* GPU Compute Utilization (%)


* **数据支撑：** 你需要数据来写这句话：“在 N < 559 时，我们只是提高了 GPU 的利用率，而没有触及显存天花板。”



---

### 🔵 实验板块二：收敛动力学对比 (针对 3.2.2 Convergence Dynamics)

**目标：** 证明基于梯度的优化器（Gradient-based）比进化算法（CMA-ES）在寻找最优解时“路走得更直”。

* [ ] **实验 3：收敛轨迹可视化 (Loss vs. Wall-clock Time)**
* **痛点：** 很多论文只画 Loss vs. Epochs，但这不公平，因为进化算法的一个 Generation 和梯度下降的一个 Epoch 计算量不同。
* **修正：** 必须画 **Loss vs. Time (seconds)**。
* **操作：**
* 选取 3-5 个代表性流域（比如：一个易收敛的湿润区，一个难收敛的干旱区）。
* 运行 Diff-MARRMoT (Adam/LBFGS)。
* 运行 CMA-ES (设置相同的种群大小或调整至最佳状态)。
* 记录每一步优化后的 Loss 值和当前的时间戳。


* **预期图表：** Diff-MARRMoT 应该像悬崖跳水一样在几秒内 Loss 骤降并拉平；CMA-ES 则是缓慢下降的阶梯状。


* [ ] **实验 4：端到端加速比计算 (Global Speedup Quantification)**
* **目的：** 支撑文中提到的 " 量级加速"。
* **计算：**
*  = (单流域 CPU 平均耗时)  559  (CMA-ES 收敛所需平均代数)
*  = (559 流域并行 GPU 耗时)  (Adam 收敛所需 Epochs)


* **注意：** 这里要诚实。如果 CPU 版本跑完 559 个流域太慢，可以用前 10 个的平均时间乘以 559 进行**推算 (Projected Time)**，并在论文中注明是 "Estimated time"。