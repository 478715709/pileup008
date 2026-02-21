# 堆积脉冲数据构建 & 检查指南

## 📊 **三个核心概念**

### 1️⃣ 构建过程
```
单脉冲数据 → 指数到达采样 → 随机组合(K=2/3) 
  → 基线去除 → 增强(噪声/漂移) 
  → 可见性约束 → NPZ文件
```

### 2️⃣ 两种 Profile
| Profile | 可见点数 | 能量比 | 重采样限制 | 特点 |
|---------|---------|--------|---------|------|
| **main** | 600 | 15% | 50次 | 宽松，截断少 |
| **hard** | 200 | 5% | 3次 | 严格，截断多 |

**意义**：hard 是增难版本（脉冲露出来更少、更容易被截断）

### 3️⃣ 两种生成模式
- **realistic**：随机抽样，真实 γ/n 比例 → 用于微调+评估
- **balanced**：均衡分配，组合标签平均 → 用于预训练

---

## 🔍 **检查输出及其含义**

### 终端输出的关键指标

**K 分布**
```
K=2: 7,185 (50.0%)
K=3: 7,185 (50.0%)
```
→ K=2/3 的样本比例（balanced会均分，realistic随机）

**λ 分布**
```
λ=100kHz: 2,395 (16.7%)
λ=200kHz: 2,395 (16.7%)
...
```
→ 每个频率档的样本数（应均匀分布）

**截断统计（关键！）**
```
Component 1 截断: 50 (0.3%)     ← main
Component 1 截断: 2145 (75.1%)  ← hard
```
→ **hard 的截断率高很多** = 约束有效 ✅

**能量比统计（关键！）**
```
main: min=0.150, mean=0.425, max=0.998
hard: min=0.050, mean=0.182, max=0.654
```
→ **hard 的能量比显著更低** = 脉冲露出来更少 ✅

---

## 📈 **五张关键图解读**

| 图名 | 看哪里 | 说明 |
|------|--------|------|
| `realistic_train_samples.png` | 蓝线+虚线标记 | 5条波形的堆积效果，虚线是第2/3脉冲位置 |
| `realistic_train_targets_decomp.png` | 黑线vs彩色虚线 | 波形分离准不准，targets是否准确 |
| `comp_labels_comparison.png` | 两条折线 | main vs hard 的组合类型分布（应相似） |
| `main_hard_truncation_comparison.png` | 3个柱子 | **核心对比**：hard截断率应明显高于main |
| `main_hard_visibility_comparison.png` | 3个直方图 | **核心对比**：hard的能量比应明显左移（更低） |

**最重要的两张**：截断率和能量比对比图 = 验证hard是否真的更严格

---

## 🚀 **快速使用**

### 生成数据
```bash
# main=14370, hard=2874 (自动计算 20%)
python scripts/03_make_piled_dataset_v2.py --emit-hard --n-pile 14370 --hard-frac 0.2 --mix-mode both

# 或显式指定 hard 样本数
python scripts/03_make_piled_dataset_v2.py --emit-hard --n-pile 14370 --n-pile-hard 3000
```

### 检查数据并生成对比图
```bash
python scripts/04_check_piled_dataset_v2.py \
  --npz results/piled_pulse_v2/realistic_train_pileup_v2.npz \
        results/piled_pulse_v2_hard/realistic_train_pileup_v2_hard.npz \
  --plot --plot-dir results/piled_pulse_v2/figs
```

---

## 📋 **输出数据结构**

每个 NPZ 文件包含：
```python
X              # (N, 10002) 合成波形
y_K            # (N,) 堆积重数 2或3
comp_labels    # (N, 3) 组合标签 [γ=1/n=0/无=-1]
shifts_samples # (N, 2) 脉冲时间位置
lambda_hz      # (N,) 每个样本的λ值
targets        # (N, 3, 10002) 分离的3个分量
targets_mask   # (N, 3) 分量是否存在
truncated_flags    # (N, 3) 各分量是否被截断 ← hard会更多
visibility_metrics # (N, 3) 能量比 ← hard会更低
```

---

## ✅ **验收检查清单**

- ✅ main 截断率 < hard 截断率（说明约束有效）
- ✅ main 能量比 > hard 能量比（说明hard脉冲露出来更少）
- ✅ 组合标签分布相似（不同模式间的一致性）
- ✅ 波形分解图清晰（targets准确）

---

## 💡 **一句话总结**

hard profile 通过 **严格的可见性约束** 生成更难的数据，截断率更高、能量比更低 → 用来测试模型的鲁棒性
