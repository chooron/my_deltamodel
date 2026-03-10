通过组合不同的水文过程选项，理论上可以组合出 $3 \times 3 \times 3 \times 3 \times 3 \times 2 \times 1 \times 1 = 486$ 种不同的模型结构。

如果您已经搭建好了各个过程的公式，现在需要将它们"组装"起来并进行参数率定，以下是该模型的**组成结构**、**参数率定范围**以及**极其重要的参数转换规则**。

---

### 一、 模型由哪些模块（水文过程）组成？

完整的模型由以下 8 类水文过程构成（按运行逻辑排序），每个过程通过权重 $\mathbf{w}$ 对多个公式选项进行可微分混合：

$$\hat{F} = \sum_{k} w_k \cdot F_k, \quad \sum_k w_k = 1, \quad w_k \geq 0$$

权重 $w_k$ 由神经网络输出的 logits 经 Softmax / Gumbel-Softmax / Sparsemax / Entmax-1.5 激活得到。

#### 1. 具有多个选项的过程（可替换模块）：
*   **雨雪分割 (Rain-Snow Partitioning):** 可选 HBV ($A_1$)、Dingman ($A_2$) 或 Threshold ($A_3$) 公式。
*   **雪平衡 (Snow Balance):** 可选 SIMPLE ($B_1$)、HBV ($B_2$) 或 HMETS ($B_3$) 公式。
*   **入渗 (Infiltration):** 可选 HMETS ($C_1$)、VIC_ARNO ($C_2$) 或 HBV ($C_3$) 公式。
*   **土壤蒸发 (Soil Evaporation):** 可选 ALL ($D_1$)、LINEAR ($D_2$) 或 VIC ($D_3$) 公式。
*   **快速壤中流 (Quickflow):** 可选 LINEAR_ANALYTIC ($E_1$)、VIC ($E_2$) 或 TOPMODEL ($E_3$) 公式。
*   **基流 (Baseflow):** 可选 LINEAR_ANALYTIC ($F_1$) 或 POWER_LAW ($F_2$) 公式。

#### 2. 固定选项的过程（单一模块）：
*   **渗漏 (Percolation):** 线性渗漏 ($G_1$)，固定
*   **汇流 (Routing):** Gamma 单位线 ($H_1$)，固定

---

### 二、 各过程数学公式

#### A. 雨雪分割

**变量说明：** $P$ = 总降水 (mm/d)，$T$ = 气温 (°C)，$S_f$ = 降雪量，$R_f$ = 降雨量

**$A_1$ HBV 线性过渡：**
$$S_f = P \cdot \text{clamp}\!\left(\frac{t_t + t_{ti}/2 - T}{t_{ti}},\ 0,\ 1\right), \quad R_f = P - S_f$$

> $t_t$ = 雨雪临界温度 ($x_{31}$)，$t_{ti}$ = 过渡区间宽度 ($x_{32}$)

**$A_2$ Dingman 双侧修正：**
$$S_f = P \cdot \text{clamp}\!\left(\frac{1}{2}\left[1 + e^{-2.2\,(d_w+\epsilon)^{1.3}} - e^{-2.2\,(d_c+\epsilon)^{1.3}}\right],\ 0,\ 1\right)$$

> $d_c = \max(t_s - T,\ 0)$（寒冷侧偏差），$d_w = \max(T - t_s,\ 0)$（温暖侧偏差），$t_s = x_{31}$

**$A_3$ Sigmoid 阈值：**
$$S_f = P \cdot \sigma\!\left(5\,(t_t - T)\right), \quad R_f = P - S_f$$

> $\sigma(\cdot)$ 为 sigmoid 函数，$t_t = x_{31}$

降水校正（三种方案共用）：
$$S_f \leftarrow S_f \cdot c_s, \quad R_f \leftarrow R_f \cdot c_r$$

> $c_r = x_{33}$（降雨校正系数），$c_s = x_{34}$（降雪校正系数）

---

#### B. 雪平衡

**状态变量：** $S_{snow}$ = 固态雪水当量 (mm)，$S_{liq}$ = 液态水储量 (mm)，$S_{cum}$ = 累积融雪量 (mm)

**$B_1$ SIMPLE_MELT（单层雪桶）：**
$$M = \min\!\left(S_{snow},\ ddf \cdot \max(T - T_{melt},\ 0)\right)$$
$$Q_{snow} = M + R_f, \quad S_{snow}' = S_{snow} + S_f - M$$

> $ddf = x_{24}$（度日融雪因子），$T_{melt} = x_{26}$（融雪温度阈值）；无液态水层

**$B_2$ SNOBAL_HBV（含液态水保持与重冻）：**
$$M = \min\!\left(S_{snow},\ ddf \cdot \max(T - T_{melt},\ 0)\right)$$
$$R_{frz} = \min\!\left(S_{liq},\ k_f \cdot \max(T_{frz} - T,\ 0)\right)$$
$$Q_{snow} = \max\!\left(S_{liq} + R_f + M - swi \cdot S_{snow},\ 0\right)$$
$$S_{snow}' = S_{snow} + S_f - M + R_{frz}, \quad S_{liq}' = \max(S_{liq} + M + R_f - R_{frz} - Q_{snow},\ 0)$$

> $ddf = x_{24}+x_{25}$（最大融雪因子），$T_{melt} = x_{26}$，$k_f = x_{18}$（重冻因子），$T_{frz} = x_{16}$（重冻温度），$swi = x_{19}$（液态水容量比）

**$B_3$ SNOBAL_HMETS（变液态水容量）：**
$$ddf = \min\!\left(ddf_{max},\ ddf_{min} \cdot (1 + \alpha_{agg} \cdot S_{cum})\right)$$
$$M = \min\!\left(S_{snow},\ ddf \cdot \max(T - T_{melt},\ 0)\right)$$
$$R_{frz} = \min\!\left(S_{liq},\ k_f \cdot \max(T_{frz} - T,\ 0)^{n_{frz}}\right)$$
$$swi = \max\!\left(swi_{min},\ swi_{max} \cdot (1 - \alpha_{swi} \cdot S_{cum})\right)$$
$$Q_{snow} = \max\!\left(S_{liq} + R_f + M - swi \cdot S_{snow},\ 0\right)$$

> $ddf_{min} = x_{24}$，$ddf_{max} = x_{24}+x_{25}$，$T_{melt} = x_{26}$，$k_f = x_{18}$，$T_{frz} = x_{16}$，$n_{frz} = x_{17}$，$swi_{min} = x_{13}$，$swi_{max} = x_{13}+x_{14}$，$\alpha_{swi} = x_{15}$，$\alpha_{agg} = x_{27}$

---

#### C. 入渗

**变量说明：** $P_{eff} = Q_{snow}$（雪模块出流），$S$ = 表层土壤含水量 (mm)，$S_{max}$ = 表层最大容量 (mm)

**$C_1$ HMETS：**
$$INF = P_{eff} \cdot \max\!\left(1 - c_r \cdot \frac{S}{S_{max}},\ 0\right)$$

> $c_r = x_1$（径流系数）

**$C_2$ VIC_ARNO（空间异质性产流）：**
$$INF = P_{eff} \cdot \left[1 - \left(1 - \frac{S}{S_{max}}\right)^{b}\right]$$

> $b = x_2$（VIC 形状指数）；土壤越满，入渗比例越高

**$C_3$ HBV：**
$$INF = P_{eff} \cdot \left[1 - \left(\frac{S}{S_{max}}\right)^{\beta}\right]$$

> $\beta = x_3$（HBV 形状指数）

地表径流：$Q_{surf} = P_{eff} - INF$

---

#### D. 土壤蒸发

**变量说明：** $PET$ = 潜在蒸散发 (mm/d)，$c_{pet} = x_8$（PET 校正系数），$S_{tension} = fc \cdot S_{max}$（张力水容量）

**$D_1$ SOILEVAP_ALL：**
$$E = \min\!\left(PET \cdot c_{pet},\ S\right)$$

**$D_2$ SOILEVAP_LINEAR（线性限制）：**
$$E = \min\!\left(PET \cdot c_{pet} \cdot \min\!\left(\frac{S}{S_{tension}},\ 1\right),\ S\right)$$

> $fc = x_9 + x_{10}$（田间持水量比）

**$D_3$ SOILEVAP_VIC（幂律限制）：**
$$E = \min\!\left(PET \cdot c_{pet} \cdot \left[1 - \left(1 - \frac{S}{S_{max}}\right)^{\gamma}\right],\ S\right)$$

> $\gamma = x_{36}$（VIC 蒸发形状指数）

---

#### E. 渗漏（固定，$G_1$）

**表层 → 潜水层：**
$$PERC_{top} = \min\!\left(k_{perc,top} \cdot S_{top},\ S_{top}\right)$$

**潜水层 → 深层（水量漏损，不进入径流）：**
$$PERC_{ph} = \min\!\left(k_{perc,ph} \cdot S_{ph},\ S_{ph}\right)$$

> $k_{perc,top} = x_{28}$，$k_{perc,ph} = x_{35}$

**表层土壤更新：**
$$S_{top}' = \max\!\left(S_{top} + INF - E - PERC_{top},\ 0\right)$$
$$Q_{overflow} = \max\!\left(S_{top}' - S_{max,top},\ 0\right), \quad S_{top}' \leftarrow S_{top}' - Q_{overflow}$$
$$Q_{surf} \leftarrow Q_{surf} + Q_{overflow}$$

---

#### F. 快速流

**变量说明：** $S$ = 更新后的表层土壤含水量

**$E_1$ LINEAR_ANALYTIC：**
$$Q_{quick} = S \cdot \left(1 - e^{-k_{quick}}\right)$$

> $k_{quick} = 10^{x_4}$（对数采样还原）

**$E_2$ VIC 幂律：**
$$Q_{quick} = \min\!\left(q_{max} \cdot \left(\frac{S}{S_{max}}\right)^{n_q},\ S\right)$$

> $q_{max} = x_5$，$n_q = x_6$

**$E_3$ TOPMODEL 指数：**
$$Q_{quick} = \min\!\left(q_{max} \cdot \exp\!\left(-\lambda \cdot \left(1 - \frac{S}{S_{max}}\right)\right),\ S\right)$$

> $q_{max} = x_5$，$n_q = x_6$，$\lambda = x_7$（TOPMODEL 地形指数均值）

---

#### G. 基流

**变量说明：** $S_{ph}$ = 潜水层含水量

**$F_1$ LINEAR_ANALYTIC：**
$$Q_{base} = S_{ph} \cdot \left(1 - e^{-k_{base}}\right)$$

> $k_{base} = 10^{x_{11}}$（对数采样还原）

**$F_2$ POWER_LAW 幂律：**
$$Q_{base} = \min\!\left(k_{base} \cdot S_{ph}^{n_{base}},\ S_{ph}\right)$$

> $k_{base} = 10^{x_{11}}$，$n_{base} = x_{12}$

**潜水层更新：**
$$S_{ph}' = \max\!\left(S_{ph} + PERC_{top} - PERC_{ph} - Q_{base},\ 0\right)$$
$$Q_{overflow,ph} = \max\!\left(S_{ph}' - S_{max,ph},\ 0\right), \quad Q_{base} \leftarrow Q_{base} + Q_{overflow,ph}$$

**总径流：**
$$Q_{total} = Q_{surf} + Q_{quick} + Q_{base}$$

---

#### H. Gamma 单位线路由（固定，$H_1$）

$$UH(t) \propto t^{a-1} e^{-t/b}, \quad Q_{routed}(t) = \sum_{\tau=0}^{L-1} UH(\tau) \cdot Q_{total}(t-\tau)$$

> $a = rout\_a \in [0, 2.9]$，$b = rout\_b \in [0, 6.5]$，$L=15$（截断长度）

---

### 三、 参数列表及率定范围 ($x_1$ - $x_{36}$)

模型共有 36 个候选参数（均匀分布 Uniform），根据选择的模块组合，只有部分参数会被激活。

#### 1. 入渗参数 (Infiltration)
*   $x_1$: `hmets_runoff_coeff` $c_r$ **[0.0, 1.0]** (-) *(仅C1)*
*   $x_2$: `b_exp` $b$ **[0.3, 3.0]** (-) *(仅C2)*
*   $x_3$: `hbv_beta` $\beta$ **[0.5, 3.0]** (-) *(仅C3)*

#### 2. 快速流参数 (Quickflow)
*   $x_4$: `log_k_quick` **[-5.0, -1.0]** → $k_{quick} = 10^{x_4}$ ($d^{-1}$) *(仅E1)*
*   $x_5$: `q_max` $q_{max}$ **[0.0, 100.0]** ($mm \cdot d^{-1}$) *(E2, E3)*
*   $x_6$: `n_quick` $n_q$ **[0.5, 2.0]** (-) *(E2, E3)*
*   $x_7$: `topmodel_lambda` $\lambda$ **[5.0, 10.0]** (m) *(仅E3)*

#### 3. 蒸发参数 (Evaporation)
*   $x_8$: `pet_correction` $c_{pet}$ **[0.0, 3.0]** (-) *(D1, D2, D3)*
*   $x_9$: `sat_wilt` (凋萎点) **[0.0, 0.05]** (frac) *(仅D2)*
*   $x_{10}$: `delta_fc` (增量) **[0.0, 0.45]** (frac) → $fc = x_9 + x_{10}$ *(仅D2)*
*   $x_{36}$: `soilevap_vic_gamma` $\gamma$ **[0.3, 3.0]** (-) *(仅D3)*

#### 4. 基流参数 (Baseflow)
*   $x_{11}$: `log_k_base` **[-5.0, -2.0]** → $k_{base} = 10^{x_{11}}$ ($d^{-1}$) *(F1, F2)*
*   $x_{12}$: `n_base` $n_{base}$ **[0.5, 2.0]** (-) *(仅F2)*

#### 5. 积雪平衡参数 (Snow Balance)
*   $x_{13}$: `swi_min` $swi_{min}$ **[0.0, 0.1]** (frac) *(仅B3)*
*   $x_{14}$: `delta_swi_max` (增量) **[0.01, 0.3]** (frac) → $swi_{max} = x_{13} + x_{14}$ *(仅B3)*
*   $x_{15}$: `swi_reduct` $\alpha_{swi}$ **[0.005, 0.1]** ($mm^{-1}$) *(仅B3)*
*   $x_{16}$: `refreeze_temp` $T_{frz}$ **[-5.0, 2.0]** ($^\circ C$) *(B2, B3)*
*   $x_{17}$: `refreeze_exp` $n_{frz}$ **[0.3, 1.0]** (-) *(仅B3)*
*   $x_{18}$: `refreeze_factor` $k_f$ **[0.0, 5.0]** ($mm \cdot d^{-1} \cdot ^\circ C^{-1}$) *(B2, B3)*
*   $x_{19}$: `snow_swi_hbv` $swi$ **[0.0, 0.4]** (frac) *(仅B2)*

#### 6. 汇流参数 (Convolution / Routing)
*   $x_{20}$: `gamma_shape_surf` **[0.3, 20.0]** (-) *(保留，未在当前路由中使用)*
*   $x_{21}$: `gamma_scale_surf` **[0.01, 5.0]** (-) *(保留)*
*   $x_{22}$: `gamma_shape_delay` **[0.5, 13.0]** (-) *(保留)*
*   $x_{23}$: `gamma_scale_delay` **[0.15, 1.5]** (-) *(保留)*
*   `rout_a`: Gamma UH 形状参数 **[0.0, 2.9]** (-) *(H1，独立路由参数)*
*   `rout_b`: Gamma UH 尺度参数 **[0.0, 6.5]** (-) *(H1，独立路由参数)*

#### 7. 潜在融雪参数 (Potential Melt)
*   $x_{24}$: `min_melt_factor` $ddf_{min}$ **[1.5, 3.0]** ($mm \cdot d^{-1} \cdot ^\circ C^{-1}$) *(B1, B3)*
*   $x_{25}$: `delta_melt_factor` (增量) **[0.0, 5.0]** → $ddf_{max} = x_{24} + x_{25}$ *(B2, B3)*
*   $x_{26}$: `dd_melt_temp` $T_{melt}$ **[-1.0, 1.0]** ($^\circ C$) *(B1, B2, B3)*
*   $x_{27}$: `dd_aggradation` $\alpha_{agg}$ **[0.01, 0.2]** ($mm^{-1}$) *(仅B3)*

#### 8. 渗漏与土壤参数 (Percolation & Soil)
*   $x_{28}$: `perc_coeff_top` $k_{perc,top}$ **[0.00001, 0.02]** ($d^{-1}$) *(G1)*
*   $x_{29}$: `thickness_top` **[0.0, 0.5]** (m) → $S_{max,top} = x_{29} \times 1000 + 1$
*   $x_{30}$: `thickness_phreatic` **[0.0, 2.0]** (m) → $S_{max,ph} = x_{30} \times 1000 + 1$
*   $x_{35}$: `perc_coeff_phreatic` $k_{perc,ph}$ **[0.0, 0.02]** ($d^{-1}$) *(G1)*

#### 9. 气象强迫处理 (Precipitation & Rain-Snow Partitioning)
*   $x_{31}$: `rainsnow_temp` $t_t / t_s$ **[-3.0, 3.0]** ($^\circ C$) *(A1, A2, A3)*
*   $x_{32}$: `rainsnow_delta` $t_{ti}$ **[0.5, 4.0]** ($^\circ C$) *(仅A1)*
*   $x_{33}$: `rain_correction` $c_r$ **[0.8, 1.2]** (-) *(A1, A2, A3)*
*   $x_{34}$: `snow_correction` $c_s$ **[0.8, 1.2]** (-) *(A1, A2, A3)*

---

### 四、 ⚠️ 模型构建与率定的关键注意事项 (Crucial Rules)

在将率定算法生成的 $[x_1, ..., x_{36}]$ 向量代入公式之前，**必须**进行以下数学转换：

1.  **对数采样还原 (Logarithmic sampling):**
    *   真实快速流系数 $k_{quick} = 10^{x_4}$
    *   真实基流系数 $k_{base} = 10^{x_{11}}$
    *   *(因为 $x_4$ 和 $x_{11}$ 的范围是负数，这样转换是为了在对数尺度上均匀采样极小的值)*

2.  **保证物理意义上限大于下限的参数转换 (Additive parameters):**
    *   田间持水量比 $fc = x_9 + x_{10}$（凋萎点 + 增量）
    *   雪最大液态水容量 $swi_{max} = x_{13} + x_{14}$（最小值 + 增量）
    *   最大融雪因子 $ddf_{max} = x_{24} + x_{25}$（最小值 + 增量）

3.  **土壤容量换算 (Soil capacity):**
    *   $S_{max,top} = x_{29} \times 1000 + 1$（m → mm，+1 避免零容量）
    *   $S_{max,ph} = x_{30} \times 1000 + 1$

4.  **土壤分层设定 (Soil Model):**
    模型在物理结构上设定为 **3层土壤**：
    *   第1层：表层 (TOPSOIL)，厚度由参数 $x_{29}$ 决定。
    *   第2层：潜水层 (PHREATIC)，厚度由参数 $x_{30}$ 决定。
    *   第3层：深层含水层 (Deep Groundwater)，水量漏损（不进入径流）。

5.  **权重激活方法 (Weight Activation):**
    过程权重 logits 支持四种激活方式：
    *   **Softmax:** $w_k = e^{z_k / \tau} / \sum_j e^{z_j / \tau}$（稠密，所有选项均有贡献）
    *   **Gumbel-Softmax:** 训练时加入 Gumbel 噪声，推理时退化为 Softmax（默认）
    *   **Sparsemax:** $\mathbf{w} = \arg\min_{\mathbf{p} \in \Delta} \|\mathbf{p} - \mathbf{z}\|^2$（稀疏，部分选项权重为零）
    *   **Entmax-1.5:** $\mathbf{w} = \arg\max_{\mathbf{p} \in \Delta} \left[\mathbf{p}^\top \mathbf{z} + H_{1.5}(\mathbf{p})\right]$（介于 Softmax 和 Sparsemax 之间）

**总结建议：**
代码架构设计为主控函数（接收 $[x_1, ..., x_{36}]$ 数组和权重 logits），在主控函数开头第一步先执行上述的**参数转换**，然后再将转换后的真实物理参数传递给各个公式模块，最后对各选项输出进行加权混合。
