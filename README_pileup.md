# 堆积脉冲数据集构建指南

## 快速开始

### 推荐运行命令

**1. 生成 Balanced 与 Realistic 两套数据集：**
```bash
python scripts/01_make_piled_dataset.py \
  --train single_split_train.mat \
  --test single_split_test.mat \
  --outdir results/piled_pulse \
  --seed 42 \
  --mix-mode both
```

**2. 检查生成的数据集：**
```bash
python scripts/02_check_piled_dataset.py \
  --npz results/piled_pulse/realistic_train_pileup.npz \
        results/piled_pulse/balanced_train_pileup.npz \
  --plot
```

---

## 核心概念

### 两套数据集策略

| 模式 | 特点 | 用途 |
|------|------|------|
| **Balanced** | 按"有序组合标签"均匀分配，每个分量组合类别数量近似 | **预训练 + 数据增强** |
| **Realistic** | 按原始 γ/n 比例随机抽样 | **微调 + 最终评估** |

- **预训练阶段**：用 Balanced 数据集，样本多样性好，各种组合均衡
- **微调阶段**：用 Realistic 数据集，更接近真实分布
- **测试评估**：用 `realistic_test` 数据集，最终模型性能评价

---

## 数据格式说明

### NPZ 文件结构

每个 `{mode}_{split}_pileup.npz` 包含以下字段：

| 字段 | 形状 | 说明 |
|------|------|------|
| **X** | (N, L) | 合成的堆积波形 |
| **y_is_pile** | (N,) | 全 1（标记为堆积脉冲） |
| **y_K** | (N,) | 2 或 3（堆积重数） |
| **comp_labels** | (N, 3) | 分量标签（γ=1, n=0, 不存在=-1）<br/>例：[1, 0, -1] = [γ, n, ∅] |
| **shifts_samples** | (N, 2) | [shift₂, shift₃]（样本点单位） |
| **lambda_hz** | (N,) | 每条样本的到达率 λ (Hz) |
| **targets** | (N, 3, L) | 分离监督目标（可选） |
| **targets_mask** | (N, 3) | 该分量是否存在 |
| **fs_hz** | 标量 | 采样率 (500e6 Hz) |
| **L** | 标量 | 波形长度 (10002) |
| **baseline_b** | 标量 | 基线计算点数 |
| **zero_prefix_b** | 标量 | 是否将前 B 点置零 |
| **seed** | 标量 | 随机种子 |

### 组合标签解释

**K=2（二重堆积）4 类：**
- `[1, 1, -1]` → [γ, γ]
- `[1, 0, -1]` → [γ, n]
- `[0, 1, -1]` → [n, γ]
- `[0, 0, -1]` → [n, n]

**K=3（三重堆积）8 类：**
- `[1, 1, 1]` → [γ, γ, γ]
- `[1, 1, 0]` → [γ, γ, n]
- `[1, 0, 1]` → [γ, n, γ]
- `[0, 1, 1]` → [n, γ, γ]
- `[1, 0, 0]` → [γ, n, n]
- `[0, 1, 0]` → [n, γ, n]
- `[0, 0, 1]` → [n, n, γ]
- `[0, 0, 0]` → [n, n, n]

---

## 脚本参数详解

### 01_make_piled_dataset.py

```
主要参数：
  --train PATH              训练集 .mat 路径（默认：single_split_train.mat）
  --test PATH               测试集 .mat 路径（默认：single_split_test.mat）
  --outdir PATH             输出目录（默认：results/piled_pulse）
  
  --lambda-mhz VALS         λ 档位 (MHz)（默认：0.1 0.2 0.4 0.8 1.5 3.0）
  --pile-mult VALS          堆积重数（默认：2 3，即都启用）
  --ratio-3 FLOAT           K=3 在 Balanced 中的占比（默认：0.5 = 50%）
  
  --n-pile-like-single      堆积样本数 = 单脉冲总数（默认启用）
  --n-pile N                显式指定堆积样本数（覆盖上一选项）
  
  --baseline-b N            基线计算点数（默认：200）
  --zero-prefix-b BOOL      是否将前 B 点置零（默认：True）
  --seed N                  随机种子（默认：42）
  
  --save-targets BOOL       是否保存 targets（默认：True）
  --mix-mode MODE           模式选择（默认：both）
                            - both：生成 realistic + balanced
                            - realistic：仅生成 realistic
                            - balanced：仅生成 balanced
```

**示例：高 λ 压力测试（最高 10 MHz）**
```bash
python scripts/01_make_piled_dataset.py \
  --lambda-mhz 0.5 1.0 2.0 5.0 10.0 \
  --n-pile 10000 \
  --seed 99
```

### 02_check_piled_dataset.py

```
参数：
  --npz PATHS               一个或多个 npz 文件路径
  --plot                    是否生成可视化图（随机 5 条波形 + targets 分解）
  --plot-dir DIR            图片输出目录（默认：results/piled_pulse/figs）
```

**示例：对比 realistic 与 balanced**
```bash
python scripts/02_check_piled_dataset.py \
  --npz results/piled_pulse/realistic_train_pileup.npz \
        results/piled_pulse/balanced_train_pileup.npz \
  --plot
```

---

## 工作流示例

### 完整训练流程

```bash
# Step 1: 生成数据集
python scripts/01_make_piled_dataset.py \
  --train single_split_train.mat \
  --test single_split_test.mat \
  --seed 42

# Step 2: 检查与可视化
python scripts/02_check_piled_dataset.py \
  --npz results/piled_pulse/balanced_train_pileup.npz \
        results/piled_pulse/realistic_train_pileup.npz \
  --plot

# Step 3: 模型预训练（伪代码，需自行编写）
# 用 balanced_train_pileup.npz 做预训练
# 输入：X, comp_labels
# 监督信息：targets（可选）、targets_mask

# Step 4: 模型微调（伪代码）
# 用 realistic_train_pileup.npz 做微调

# Step 5: 最终评估（伪代码）
# 在 realistic_test_pileup.npz 上评估性能
```

---

## 输出文件

生成完成后，`results/piled_pulse/` 目录结构如下：

```
results/piled_pulse/
├── realistic_train_pileup.npz    ← 训练集（Realistic 模式）
├── realistic_test_pileup.npz     ← 测试集（Realistic 模式）
├── balanced_train_pileup.npz     ← 训练集（Balanced 模式）
├── balanced_test_pileup.npz      ← 测试集（Balanced 模式）
└── figs/                         ← 可视化图（如果使用 --plot）
    ├── realistic_train_samples.png
    ├── realistic_train_targets_decomp.png
    ├── balanced_train_samples.png
    └── ...
```

---

## 技术细节

### 堆积合成算法

1. **时间间隔采样**：Δt ~ Exp(λ)
   - t₂ = Δt₂
   - t₃ = t₂ + Δt₃

2. **基线处理**：
   - 第 1 脉冲：不去基线，从 t=0 开始
   - 第 2、3 脉冲：去基线（前 B 点均值）+ 可选置零前缀

3. **叠加**：
   - 允许截窗（后续脉冲移位后超出窗口自然截断）
   - 无重采样

4. **Targets 构建**：
   - targets[:, 0, :] = 首脉冲原样
   - targets[:, 1, :] = 第 2 脉冲（去基线后，按 shift₂ 放入窗）
   - targets[:, 2, :] = 第 3 脉冲（去基线后，按 shift₃ 放入窗）

### Balanced 组合均衡

- 先按比例分配 K=2 与 K=3 样本数
- 在每个 K 内，把样本数尽量均分到所有组合类（不能整除则余数随机分配）
- 允许**有放回采样**单脉冲（保证任意组合都能凑齐）

---

## 故障排除

### 问题：`y_K` 为什么不全是 2 或 3？
在 Balanced 模式下，样本会同时包含 K=2 和 K=3（按 `--ratio-3` 比例分配）。

### 问题：为什么 Realistic 和 Balanced 的样本数一样？
默认行为是让堆积样本数 = 单脉冲总数。可用 `--n-pile` 显式指定。

### 问题：targets 全是 0？
检查 `--save-targets` 是否为 True（默认）；或查看 `targets_mask` 确认分量是否存在。

### 问题：λ 分布不均匀？
这是正常的，因为在生成过程中从 `--lambda-mhz` 中随机抽取。若要完全均匀，可在抽样时改用分层抽样。

---

## 依赖

- **numpy**：数组操作
- **h5py**：读取 .mat 文件
- **matplotlib**（可选）：可视化

```bash
pip install numpy h5py matplotlib
```

---

## 参考

- 单脉冲数据来源：`single_split_train.mat`, `single_split_test.mat`
- 堆积合成模块：`src/pileup/`
- 核心逻辑：`src/pileup/synth.py`
