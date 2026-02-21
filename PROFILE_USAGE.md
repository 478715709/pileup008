# Profile 功能说明

## 概述
03 和 04 脚本现已支持 **main** 和 **hard** 两种 profile，允许在不同可见性约束下生成和对比数据集。

---

## Profile 配置对比

### main (宽松约束)
- `min_visible_points`: 600
- `min_visible_energy_ratio`: 0.15
- `max_shift_resample`: 50
- **特性**：更容易通过可见性检查，截断率较低

### hard (严格约束)
- `min_visible_points`: 200 (宽松 ⬇️)
- `min_visible_energy_ratio`: 0.05 (宽松 ⬇️)
- `max_shift_resample`: 3 (严格 ⬆️)
- **特性**：更严格的重采样限制，截断率较高

---

## 脚本用法

### 03_make_piled_dataset_v2.py

#### 新增参数（v2.1）
- `--n-pile-hard`: hard profile 的样本数（默认 None，自动计算）
- `--hard-frac`: hard 样本数比例因子（默认 0.2），当 `--emit-hard` 且未指定 `--n-pile-hard` 时生效

#### 选项 1: 仅生成 main
```bash
python scripts/03_make_piled_dataset_v2.py --profile main
```
- 输出目录：`results/piled_pulse_v2/` （文件不带后缀）
- 文件例如：`realistic_train_pileup_v2.npz`

#### 选项 2: 仅生成 hard
```bash
python scripts/03_make_piled_dataset_v2.py --profile hard --outdir results/piled_pulse_v2_hard
```
- 输出目录：`results/piled_pulse_v2_hard/`
- 文件例如：`realistic_train_pileup_v2_hard.npz`

#### 选项 3: 同时生成 main 和 hard（推荐）
```bash
python scripts/03_make_piled_dataset_v2.py --emit-hard
```
- 输出目录1：`results/piled_pulse_v2/` （main，文件无后缀）
- 输出目录2：`results/piled_pulse_v2_hard/` （hard，文件带 `_hard` 后缀）
- **默认样本数配置**：hard 自动为 main 的 20% (`--hard-frac=0.2`)

#### 选项 4: 自定义 hard 样本数比例
```bash
python scripts/03_make_piled_dataset_v2.py --emit-hard --n-pile 14370 --hard-frac 0.2 --mix-mode both
```
- main 样本数：14,370
- hard 样本数：2,874（14370 * 0.2）
- 同时生成 realistic 和 balanced 两种模式

#### 选项 5: 显式指定 hard 样本数
```bash
python scripts/03_make_piled_dataset_v2.py --emit-hard --n-pile 14370 --n-pile-hard 3000
```
- main 样本数：14,370
- hard 样本数：3,000（显式指定，忽略 hard-frac）

#### 自定义参数覆盖
```bash
python scripts/03_make_piled_dataset_v2.py \
  --profile main \
  --n-pile 3000 \
  --amp-min 0.80 --amp-max 1.20 \
  --min-visible-points 800
```
- profile 提供基础配置，命令行参数可覆盖

---

### 04_check_piled_dataset_v2.py

#### 检查单个数据集
```bash
python scripts/04_check_piled_dataset_v2.py \
  --npz results/piled_pulse_v2/realistic_train_pileup_v2.npz
```

#### 检查多个数据集并对比
```bash
python scripts/04_check_piled_dataset_v2.py \
  --npz results/piled_pulse_v2/realistic_train_pileup_v2.npz \
        results/piled_pulse_v2_hard/realistic_train_pileup_v2_hard.npz \
  --plot --plot-dir results/piled_pulse_v2/figs
```

#### 对比 main 和 hard
检查脚本会自动识别文件中的 profile 标签（_hard 后缀或文件名），当同时输入 main 和 hard 数据集时，生成以下对比图：

1. **comp_labels_comparison.png** - 组合标签分布对比
2. **main_hard_truncation_comparison.png** - 截断率对比（3 个分量）
3. **main_hard_visibility_comparison.png** - 可见性能量比直方图对比（3 个分量）

---

## 输出文件命名规则

- **main**：`{mode}_{split}_pileup_v2.npz`
  - 例：`realistic_train_pileup_v2.npz`
  
- **hard**：`{mode}_{split}_pileup_v2_hard.npz`
  - 例：`realistic_train_pileup_v2_hard.npz`

---

## 核心改动总结

### 03_make_piled_dataset_v2.py
1. ✅ 新增 `PROFILE_CONFIGS` 字典预置 main/hard 参数
2. ✅ 新增 `--profile` 参数选择 profile
3. ✅ 新增 `--emit-hard` 参数同时生成两套
4. ✅ 新增 `--outdir` 和 `--hard-outdir` 分别设置输出目录
5. ✅ 修改 `generate_pileup_dataset_v2` 签名，新增 `profile_suffix` 参数
6. ✅ 输出文件名根据 profile 自动加后缀

### 04_check_piled_dataset_v2.py
1. ✅ `check_single_dataset_v2` 自动识别 profile 并标记
2. ✅ 所有输出（打印、标题等）加上 `(main)` 或 `(hard)` 标签
3. ✅ 新增 `compare_main_hard_metrics` 函数对比截断率和可见性指标
4. ✅ 新增对比图生成逻辑

---

## 不变的地方 ✅

- `synth_v2.py` - 无改动
- `augment.py` - 无改动
- `sampling.py` - 无改动
- v2 增强逻辑 - 完全保留

---

## 实验流程建议

### 方案 A: 固定比例生成（推荐）
适合快速生成不同难度的数据集，hard 自动为 main 的固定比例。

1. **生成数据（main=14370, hard=2874）**
   ```bash
   python scripts/03_make_piled_dataset_v2.py --emit-hard --n-pile 14370 --hard-frac 0.2 --mix-mode both
   ```

2. **验收检查**
   ```bash
   python scripts/04_check_piled_dataset_v2.py \
     --npz results/piled_pulse_v2/realistic_train_pileup_v2.npz \
           results/piled_pulse_v2_hard/realistic_train_pileup_v2_hard.npz \
     --plot --plot-dir results/piled_pulse_v2/figs
   ```

### 方案 B: 自定义样本数
适合需要精确控制 hard 样本数的场景。

1. **生成数据**
   ```bash
   python scripts/03_make_piled_dataset_v2.py --emit-hard --n-pile 14370 --n-pile-hard 5000 --mix-mode both
   ```

---

## 特别注意

- ⚠️ `--emit-hard` 会生成 **两套完整数据集**（包括 realistic + balanced 两种模式），耗时约翻倍
- ℹ️ hard profile 的 `max_shift_resample=3` 会导致更高的截断率，这是设计目的（更苛刻的数据）
- ℹ️ 所有命令行参数可以覆盖 profile 的基础配置，但 `--profile hard` 的标记会被保留在文件后缀中
- 🆕 `--hard-frac` 默认值 0.2 表示 hard 为 main 的 20%，适合大多数场景
- 🆕 如需精确控制 hard 样本数，使用 `--n-pile-hard` 显式指定（会忽略 `--hard-frac`）

---

## 版本历史

### v2.1 (2026-01-24)
- ✅ 新增 `--hard-frac` 参数，自动计算 hard 样本数
- ✅ 新增 `--n-pile-hard` 参数，支持显式指定 hard 样本数
- ✅ 优化输出信息，清楚显示样本数来源
- ✅ 更新使用示例和文档

### v2.0
- ✅ 初始 profile 系统（main/hard）
- ✅ 增强与可见性约束
- ✅ 对比分析功能
