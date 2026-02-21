# v2.1 更新日志

## 改动文件
- `scripts/03_make_piled_dataset_v2.py` ✅

## 新增功能

### 1. `--hard-frac` 参数
- **默认值**: 0.2 (20%)
- **作用**: 当 `--emit-hard=True` 且未指定 `--n-pile-hard` 时，自动计算 hard 样本数
- **公式**: `n_pile_hard = round(n_pile * hard_frac)`

### 2. `--n-pile-hard` 参数
- **默认值**: None
- **作用**: 显式指定 hard profile 的样本数
- **优先级**: 高于 `--hard-frac`（显式设置会忽略比例计算）

### 3. 增强的输出信息
当 `--emit-hard=True` 时，脚本会打印清晰的样本数配置：

```
======================================================================
Sample count configuration:
  Main  n_pile: 14,370
  Hard  n_pile: 2,874 (computed (--n-pile * --hard-frac = 14370 * 0.2 ≈ 2874))
======================================================================
```

或（当显式指定时）：

```
======================================================================
Sample count configuration:
  Main  n_pile: 14,370
  Hard  n_pile: 3,000 (explicitly set (--n-pile-hard=3000))
======================================================================
```

## 使用示例

### 推荐用法：自动计算 hard 样本数
```bash
python scripts/03_make_piled_dataset_v2.py \
  --emit-hard \
  --n-pile 14370 \
  --hard-frac 0.2 \
  --mix-mode both
```
**结果**: main=14370, hard=2874

### 精确控制：显式指定
```bash
python scripts/03_make_piled_dataset_v2.py \
  --emit-hard \
  --n-pile 14370 \
  --n-pile-hard 3000 \
  --mix-mode both
```
**结果**: main=14370, hard=3000

## 向后兼容性 ✅

所有既有功能保持不变：
- ✅ `--profile main` 单独生成 main
- ✅ `--profile hard` 单独生成 hard
- ✅ `--emit-hard` 同时生成两套（现在默认 hard=20% main）
- ✅ 所有增强参数、可见性约束参数不受影响
- ✅ 文件命名、目录结构保持一致

## 代码改动摘要

### parse_args()
1. 更新 `--n-pile` 帮助文本为 "main 的堆积样本数"
2. 新增 `--n-pile-hard` 参数
3. 新增 `--hard-frac` 参数
4. 更新 epilog 示例

### main()
1. 新增样本数计算逻辑（在读取数据后）：
   - 如果 `emit_hard=True`，根据 `n_pile_hard` 和 `hard_frac` 计算 hard 样本数
   - 打印样本数配置和来源信息
   
2. 在 profile 循环中：
   - 根据当前 profile 选择正确的样本数
   - 在输出中显示样本数
   - 传递给 `generate_pileup_dataset_v2()`

## 最小改动原则 ✅

- ✅ 不改动 `generate_pileup_dataset_v2()` 函数签名
- ✅ 不改动任何核心生成逻辑
- ✅ 不改动文件结构和命名规则
- ✅ 仅在 `main()` 函数中添加逻辑判断
- ✅ 向后兼容，不影响既有用法

## 测试验证 ✅

| 测试场景 | n-pile | hard-frac | n-pile-hard | 预期 hard | 实际 hard | 状态 |
|---------|--------|-----------|-------------|-----------|-----------|------|
| 默认比例 | 100 | 0.2 | None | 20 | 20 | ✅ |
| 自定义比例 | 100 | 0.3 | None | 30 | 30 | ✅ |
| 显式指定 | 100 | 0.2 | 25 | 25 | 25 | ✅ |
| 生产场景 | 14370 | 0.2 | None | 2874 | 2874 | ✅ |

## 文档更新 ✅

- ✅ `PROFILE_USAGE.md` - 添加新参数说明和示例
- ✅ `TEST_HARD_FRAC.md` - 创建详细测试指南
- ✅ 脚本 `--help` - 更新使用示例

---

**总结**: 以最小改动实现了灵活的 hard 样本数控制，既可自动计算（推荐），也可显式指定（精确控制）。✅
