# 快速使用指南

## 系统已完成！🎉

堆积脉冲数据集生成模块已全部实现。以下是快速使用步骤。

---

## 三个推荐命令

### 命令 1️⃣：生成数据集（Realistic 与 Balanced）

```bash
python scripts/01_make_piled_dataset.py \
  --train single_split_train.mat \
  --test single_split_test.mat \
  --outdir results/piled_pulse \
  --seed 42 \
  --mix-mode both
```

**预期输出**：
- `results/piled_pulse/realistic_train_pileup.npz` (14,370 样本)
- `results/piled_pulse/realistic_test_pileup.npz` (3,593 样本)
- `results/piled_pulse/balanced_train_pileup.npz` (14,370 样本)
- `results/piled_pulse/balanced_test_pileup.npz` (3,593 样本)

**预计运行时间**：30-60 秒（取决于 CPU）

---

### 命令 2️⃣：检查 Realistic 数据集

```bash
python scripts/02_check_piled_dataset.py \
  --npz results/piled_pulse/realistic_train_pileup.npz \
        results/piled_pulse/realistic_test_pileup.npz \
  --plot
```

**输出内容**：
- 样本数、K 分布、λ 分布统计
- Shifts 分布、组合标签分布
- 波形可视化图（5 条样本）
- Targets 分解对比图

---

### 命令 3️⃣：检查 Balanced 数据集

```bash
python scripts/02_check_piled_dataset.py \
  --npz results/piled_pulse/balanced_train_pileup.npz \
        results/piled_pulse/balanced_test_pileup.npz \
  --plot
```

**与 Realistic 的主要差异**：
- Balanced K 分布：严格 50/50（K=2 与 K=3）
- Balanced 组合分布：12 类均衡（vs. Realistic 随机混合）

---

## 数据用途

| 数据集 | 用途 | 特点 |
|--------|------|------|
| **balanced_train** | 模型预训练 | 样本多样化、组合均衡 |
| **balanced_test** | 内部验证 | 用于模型选择 |
| **realistic_train** | 微调 | 真实 γ/n 比例 |
| **realistic_test** | **最终评估** | 代表真实性能 |

### 推荐训练流程：
1. 用 `balanced_train` 预训练（充分学习各种组合特征）
2. 用 `realistic_train` 微调（适应真实分布）
3. 用 `realistic_test` 评估最终性能 ⭐

---

## 自定义参数

### 调整 λ 值档位（压力测试）
```bash
python scripts/01_make_piled_dataset.py \
  --lambda-mhz 0.5 1.0 2.0 5.0 10.0 \
  --n-pile 5000 \
  --seed 99
```

### 调整基线处理参数
```bash
python scripts/01_make_piled_dataset.py \
  --baseline-b 300 \
  --zero-prefix-b False  # 不置零前缀，只去基线
```

### 仅生成 Realistic 或 Balanced
```bash
# 仅 Realistic
python scripts/01_make_piled_dataset.py --mix-mode realistic

# 仅 Balanced
python scripts/01_make_piled_dataset.py --mix-mode balanced
```

---

## 文件目录

```
project_root/
├── README_pileup.md                    # 完整参数文档
├── IMPLEMENTATION_SUMMARY.md           # 实现细节
├── QUICK_START.md                      # 本文件
├── scripts/
│   ├── 01_make_piled_dataset.py       # 生成脚本
│   └── 02_check_piled_dataset.py      # 检查脚本
├── src/pileup/                         # 核心库
│   ├── __init__.py
│   ├── io_mat.py                       # MAT 读取
│   ├── synth.py                        # 合成核心
│   ├── sampling.py                     # 抽样策略
│   └── utils.py                        # 工具函数
└── results/piled_pulse/                # 输出目录
    ├── realistic_train_pileup.npz
    ├── realistic_test_pileup.npz
    ├── balanced_train_pileup.npz
    ├── balanced_test_pileup.npz
    └── figs/                           # 可视化图
        ├── realistic_train_samples.png
        ├── realistic_train_targets_decomp.png
        ├── balanced_train_samples.png
        └── ...
```

---

## 数据格式快速参考

每个 NPZ 文件包含：

```python
data = np.load('results/piled_pulse/realistic_train_pileup.npz')

# 合成波形
X = data['X']              # Shape: (N, 10002)

# 标签
y_K = data['y_K']          # 堆积重数: 2 或 3
comp_labels = data['comp_labels']  # Shape: (N, 3), 组合标签

# 位置信息
shifts_samples = data['shifts_samples']  # Shape: (N, 2), shift2 & shift3
lambda_hz = data['lambda_hz']           # Shape: (N,), 各样本的 λ

# 监督信息（可选用于分离任务）
targets = data['targets']           # Shape: (N, 3, 10002)
targets_mask = data['targets_mask'] # Shape: (N, 3), 分量存在性

# 元数据
fs_hz = data['fs_hz']               # 采样率: 500e6
L = data['L']                       # 波形长度: 10002
baseline_b = data['baseline_b']     # 基线点数: 200
seed = data['seed']                 # 随机种子
```

---

## 常见任务

### 加载数据用于训练
```python
import numpy as np

# 加载 Balanced 训练集（预训练）
data = np.load('results/piled_pulse/balanced_train_pileup.npz')
X_train = data['X']                    # (N_train, 10002)
comp_labels_train = data['comp_labels']  # (N_train, 3)
y_K_train = data['y_K']                  # (N_train,)

# 加载 Realistic 测试集（最终评估）
test_data = np.load('results/piled_pulse/realistic_test_pileup.npz')
X_test = test_data['X']
comp_labels_test = test_data['comp_labels']
```

### 检查组合分布
```python
from collections import Counter

# 统计组合类型
comp_tuples = [tuple(c) for c in comp_labels]
comp_counts = Counter(comp_tuples)
print(f"组合类型数: {len(comp_counts)}")
for comp, count in sorted(comp_counts.items()):
    print(f"  {comp}: {count}")
```

### 过滤 K=2 或 K=3
```python
# 仅获取二重堆积
mask_k2 = y_K == 2
X_k2 = X[mask_k2]
comp_k2 = comp_labels[mask_k2]

# 仅获取三重堆积
mask_k3 = y_K == 3
X_k3 = X[mask_k3]
```

---

## 故障排除

### 问题：脚本报错 "File not found"
**解决**：确保在项目根目录运行脚本，.mat 文件在当前目录

### 问题：NPZ 加载失败 "Bad CRC-32"
**解决**：删除 `results/piled_pulse/` 下的所有 .npz 文件重新生成

### 问题：生成太慢
**解决**：用 `--n-pile 1000` 减少样本数进行测试

### 问题：内存不足
**解决**：
- 减少 `--n-pile` 数值
- 关闭 `--save-targets`（不保存分离目标）

---

## 下一步

✅ 数据集已生成完毕，可以开始：

1. **模型设计**：基于 comp_labels 和 shifts 做分离
2. **训练管道**：
   - 预训练：balanced 数据
   - 微调：realistic 数据
   - 评估：realistic_test 数据
3. **性能优化**：调整基线参数或 λ 档位

---

**有问题？** 详见 `README_pileup.md`
