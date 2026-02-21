# 项目现状与下一步（截至 2026-01-28）

## 1. 已完成的工作（从头到尾）

### A. 数据准备与基线检查
- **00_check_single_mat.py**
  - 功能：检查原始单脉冲 .mat 数据（gamma / neutron 数量、波形长度、基础可视化）。
  - 结果：确认单脉冲数据可用于后续合成。

### B. v1 堆积数据集（K=2/3）
- **01_make_piled_dataset.py**
  - 功能：从单脉冲合成堆积样本；支持 realistic / balanced 两种模式。
  - 结果：生成 4 个 v1 数据集。
- **02_check_piled_dataset.py**
  - 功能：检查 v1 NPZ 的统计分布与可视化（K 分布、λ 分布、targets 分解图等）。
  - 结果：产出多张可视化图，验证合成逻辑。

### C. v2 堆积数据集（main / hard profile）
- **03_make_piled_dataset_v2.py**
  - 功能：v2 合成脚本，支持 profile（main / hard）与 hard 比例控制。
  - 结果：生成 main + hard 两套 v2 数据集。
- **04_check_piled_dataset_v2.py**
  - 功能：v2 数据集检查与对比（main vs hard），输出截断率与可见性能量比对比图。
  - 结果：验证 hard 更严格、截断率更高。

### D. v2 单脉冲增强（K=1）新增能力
- **05_make_single_dataset_v2.py**
  - 功能：对原始单脉冲做增强（噪声/漂移/幅度），生成 K=1 NPZ，字段与 v2 对齐。
  - 结果：输出单脉冲增强数据集。
- **06_check_single_dataset_v2.py**
  - 功能：检查 K=1 数据集字段一致性，并绘制“多条随机波形叠加在一张图”。
  - 结果：输出 K=1 检查统计与可视化图（无方框乱码）。

---

## 2. 当前目录已产出的结果

### results/ 目录（已存在）
- **results/piled_pulse/**（v1）
  - realistic_train_pileup.npz
  - realistic_test_pileup.npz
  - balanced_train_pileup.npz
  - balanced_test_pileup.npz
  - figs/（v1 可视化图）

- **results/piled_pulse_v2/**（v2 main）
  - realistic_train_pileup_v2.npz
  - realistic_test_pileup_v2.npz
  - balanced_train_pileup_v2.npz
  - balanced_test_pileup_v2.npz
  - figs/（v2 可视化图）

- **results/piled_pulse_v2_hard/**（v2 hard）
  - realistic_train_pileup_v2_hard.npz
  - realistic_test_pileup_v2_hard.npz
  - balanced_train_pileup_v2_hard.npz
  - balanced_test_pileup_v2_hard.npz

- **results/single_pulse_v2/**（v2 单脉冲增强）
  - single_train_v2.npz
  - single_test_v2.npz
  - figs/（单脉冲叠加图）

---

## 3. 脚本功能简表（简短版）

- **00_check_single_mat.py**：检查原始单脉冲 .mat 数据。
- **01_make_piled_dataset.py**：v1 堆积数据集生成（K=2/3）。
- **02_check_piled_dataset.py**：v1 数据集统计 + 可视化。
- **03_make_piled_dataset_v2.py**：v2 生成（main/hard profile）。
- **04_check_piled_dataset_v2.py**：v2 检查 + main/hard 对比图。
- **05_make_single_dataset_v2.py**：v2 单脉冲增强（K=1）。
- **06_check_single_dataset_v2.py**：v2 单脉冲检查 + 单图叠加可视化。

---

## 4. 是否可以进入深度学习阶段？
**可以进入下一步。**

当前已具备：
- K=1 / K=2 / K=3 数据统一格式
- main/hard 难度对比数据
- realistic / balanced 模式可用于预训练 + 微调
- targets 与 targets_mask 可用于分离任务监督

---

## 5. 下一步深度学习目标（任务链路）

### 任务拆分（从易到难）
1. **堆积 / 非堆积判别**（二分类）
2. **K 判别**（K=1/2/3）
3. **组合类型判别**（comp_labels 多分类）
4. **分量分离（解混）**（回归：输出 3 个分量波形）
5. **单脉冲还原**（对分离结果做形状约束 / 重建一致性）

---

## 6. 建议的模型方向（含你提出的“Transformer + 聚类”）

### 候选模型家族（用于对比实验）
- **1D-CNN / ResNet1D**：强基线，速度快
- **Temporal CNN + Attention**：轻量但有注意力
- **Transformer Encoder (1D)**：长程依赖强
- **Conformer / Hybrid CNN-Transformer**：局部 + 全局兼顾

### 你提出的方案：Transformer + 聚类
**可行，适合作为最终方案候选**。

**依据：**
- 堆积波形含长时程衰减尾部，Transformer 擅长全局建模
- 聚类可用于发现无监督“形状族群”（如不同能量区、平顶饱和类、噪声类）
- 有利于分离任务中的“分量归因”与“结构先验”

---

## 7. 建议的下一步工作（最短路径）

1. **建立统一 DataLoader**（支持 K=1/2/3 与 targets）
2. **先做三项分类任务**
   - 堆积/非堆积
   - K=1/2/3
   - comp_labels 分类
3. **再做分离任务**
   - 预测 targets（3 通道）
   - 引入 mask 处理不存在分量
4. **评估指标**
   - 分类：Acc / F1 / Confusion
   - 分离：MSE / MAE / SNR / SI-SDR

---

## 8. 下一步准备情况结论
✅ 数据已经齐全
✅ K=1/K=2/K=3 同格式
✅ v2 main/hard 难度对比可用于鲁棒性评估

**可以直接进入深度学习建模阶段。**
