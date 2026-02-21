# 堆积脉冲数据集生成模块 - 实现总结

## 项目完成状态 ✅

所有代码已完成并通过基本功能测试。

---

## 核心模块文件清单

### src/pileup/ (核心库)
- **__init__.py** - 模块入口，导出所有公共接口
- **io_mat.py** - MAT 文件读取
  - 使用 expected_len=10002 明确判轴
  - 返回合并的波形池和标签
  
- **synth.py** - 堆积脉冲合成
  - `remove_baseline()` - 基线去除（前 B 点均值）
  - `synthesize_pileup_samples()` - 核心合成函数
    - 指数到达间隔（Exp(λ)）
    - 基线处理 + 可选前缀置零
    - 截窗支持
    - Targets 分量记录
  
- **sampling.py** - 抽样策略
  - `RealisticSampler` - 按原始比例随机
  - `BalancedSampler` - 按组合均衡
    - K=2: 4 类（γγ, γn, nγ, nn）
    - K=3: 8 类（所有组合）
    - 有放回采样支持
  
- **utils.py** - 工具函数
  - `save_pileup_dataset()` - NPZ 保存（带压缩）
  - `load_pileup_dataset()` - NPZ 加载
  - `print_dataset_stats()` - 统计信息打印
  - `analyze_comp_labels()` - 组合标签分析

### scripts/ (可执行脚本)
- **01_make_piled_dataset.py** - 数据集生成脚本
  - 参数化配置（λ, 样本数, K 值等）
  - Train/Test 分离生成
  - Realistic 与 Balanced 两种模式
  - 完整的统计信息输出
  
- **02_check_piled_dataset.py** - 数据集检查脚本
  - 单个或多个 NPZ 文件检查
  - K 分布、λ 分布、Shifts 分布统计
  - 组合标签均衡性分析
  - 可视化支持（波形 + Targets 分解）

### 文档
- **README_pileup.md** - 完整使用指南
  - 快速开始命令
  - 数据格式详解
  - 参数说明
  - 工作流示例
  - 故障排除

---

## 关键功能验证

### ✅ 数据加载
```
- 单脉冲 MAT 文件读取正确
- Gamma/Neutron 分离正确
- Shape 判轴方法可靠（L=10002）
```

### ✅ 堆积合成
```
- 指数间隔采样正确
- 基线处理（前 B 点均值去除）正确
- 可选前缀置零功能正常
- Targets 分量记录准确
```

### ✅ 抽样策略
```
- Realistic 模式：随机按比例抽样
- Balanced 模式：
  - K=2 均分到 4 类
  - K=3 均分到 8 类
  - 有放回采样工作正常
```

### ✅ 数据保存与加载
```
- NPZ 压缩保存正常
- 所有字段正确保存
- 加载恢复完整
```

---

## NPZ 文件结构

| 字段 | 形状 | 说明 |
|------|------|------|
| X | (N, L) | 合成波形 |
| y_is_pile | (N,) | 全 1 标记 |
| y_K | (N,) | 堆积重数（2 或 3） |
| comp_labels | (N, 3) | 分量标签（1=γ, 0=n, -1=∅） |
| shifts_samples | (N, 2) | [shift2, shift3] 样本点 |
| lambda_hz | (N,) | 每条样本的 λ (Hz) |
| targets | (N, 3, L) | 分量分离监督目标 |
| targets_mask | (N, 3) | 分量存在性掩码 |
| fs_hz | 标量 | 采样率（500e6） |
| L | 标量 | 波形长度（10002） |
| baseline_b | 标量 | 基线点数（200） |
| zero_prefix_b | 标量 | 前缀置零标记（0/1） |
| seed | 标量 | 随机种子 |

---

## 推荐使用流程

### 1️⃣ 生成数据集（完整规模）
```bash
python scripts/01_make_piled_dataset.py \
  --train single_split_train.mat \
  --test single_split_test.mat \
  --outdir results/piled_pulse \
  --seed 42 \
  --mix-mode both
```
**输出**: 4 个 NPZ 文件
- realistic_train_pileup.npz (14,370 样本)
- realistic_test_pileup.npz (3,593 样本)
- balanced_train_pileup.npz (14,370 样本)
- balanced_test_pileup.npz (3,593 样本)

### 2️⃣ 检查数据质量
```bash
python scripts/02_check_piled_dataset.py \
  --npz results/piled_pulse/realistic_train_pileup.npz \
        results/piled_pulse/balanced_train_pileup.npz \
  --plot
```
**输出**: 统计信息 + 可视化图片

### 3️⃣ 模型训练流程
```
a) 预训练：用 balanced_train_pileup.npz（样本多样，均衡）
b) 微调：用 realistic_train_pileup.npz（真实分布）
c) 评估：用 realistic_test_pileup.npz（最终评分）
```

---

## 命令行参数速查

### 01_make_piled_dataset.py
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --train | str | single_split_train.mat | 训练集路径 |
| --test | str | single_split_test.mat | 测试集路径 |
| --outdir | str | results/piled_pulse | 输出目录 |
| --lambda-mhz | float[] | [0.1, 0.2, ..., 3.0] | λ 值档位 |
| --pile-mult | int[] | [2, 3] | 堆积重数 |
| --ratio-3 | float | 0.5 | K=3 占比 |
| --n-pile | int | - | 样本数（默认=单脉冲总数） |
| --baseline-b | int | 200 | 基线点数 |
| --zero-prefix-b | bool | True | 前缀置零 |
| --seed | int | 42 | 随机种子 |
| --mix-mode | str | both | both/realistic/balanced |

### 02_check_piled_dataset.py
| 参数 | 类型 | 说明 |
|------|------|------|
| --npz | str[] | (必需) NPZ 文件列表 |
| --plot | flag | 生成可视化 |
| --plot-dir | str | 图片输出目录 |

---

## 测试验证

### 小规模测试（已通过）
```
n_pile=500, 所有模式
├── Realistic Train: 500 样本
│   ├── K=2: ~250, K=3: ~250 (随机)
│   ├── λ: 均匀分布在 6 档
│   └── Comp labels: 随机混合
├── Realistic Test: 500 样本
├── Balanced Train: 500 样本
│   ├── K=2: 250, K=3: 250 (严格均分)
│   └── Comp labels: 12 类均分（K=2 4类, K=3 8类）
└── Balanced Test: 500 样本

验证项:
✅ NPZ 保存正确
✅ NPZ 加载正确
✅ Comp labels 分布符合预期
✅ Shifts 分布合理
```

---

## 文件夹结构

```
d:\project_code\pythonProject\pileup004\
├── single_split_train.mat          # 原始单脉冲数据
├── single_split_test.mat
├── scripts/
│   ├── 00_check_single_mat.py      # 单脉冲检查（已有）
│   ├── 01_make_piled_dataset.py    # 堆积数据生成 ✨
│   └── 02_check_piled_dataset.py   # 堆积数据检查 ✨
├── src/
│   └── pileup/
│       ├── __init__.py             # 模块入口
│       ├── io_mat.py               # MAT 读取
│       ├── synth.py                # 合成核心
│       ├── sampling.py             # 抽样策略
│       └── utils.py                # 工具函数
├── results/
│   ├── single_pulse/               # 单脉冲结果（已有）
│   └── piled_pulse/                # 堆积脉冲结果 ✨
│       ├── realistic_train_pileup.npz
│       ├── realistic_test_pileup.npz
│       ├── balanced_train_pileup.npz
│       ├── balanced_test_pileup.npz
│       └── figs/                   # 可视化图片
├── README_pileup.md                # 完整指南 ✨
└── IMPLEMENTATION_SUMMARY.md       # 本文件
```

---

## 后续扩展建议

1. **并行化处理**：使用 multiprocessing 加速合成
2. **元数据记录**：保存生成配置到独立 JSON
3. **验证套件**：单元测试 + 集成测试
4. **模型集成**：直接加载 NPZ 到 PyTorch DataLoader
5. **性能优化**：Numba JIT 编译 synth 核心函数

---

## 常见问题

**Q: Balanced 和 Realistic 有什么区别？**
- Realistic：按 γ/n 真实比例随机抽取，更接近真实分布
- Balanced：强制每个组合类均等，预训练/数据增强用

**Q: 为什么 Realistic 的 K 分布不均？**
- 随机决定，符合预期。可用 Balanced 强制均分

**Q: Targets 是什么？**
- 分离监督信号：存储每条分量的去基线后波形（按 shift 放入窗）

**Q: 可以调整基线计算点数吗？**
- 可以，用 `--baseline-b N` 参数

---

## 许可与致谢

本模块基于核电脉冲识别项目需求开发。
核心算法：指数到达 + 基线处理 + Targets 构建

---

**最后更新**: 2026-01-22
**实现状态**: ✅ 完成
