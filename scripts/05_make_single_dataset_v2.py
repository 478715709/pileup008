#!/usr/bin/env python3
"""
【生成单脉冲增强数据集脚本 v2】
从单脉冲数据生成 K=1 的增强版本（字段格式与 pileup-v2 对齐）
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# 加入 src 目录以导入自定义模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pileup.io_mat import load_single_dataset
from src.pileup.synth_v2 import synthesize_pileup_samples_v2
from src.pileup.utils_v2 import save_pileup_dataset_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成单脉冲增强数据集 v2（K=1，字段格式与 pileup-v2 对齐）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 生成单脉冲增强数据集（使用默认参数）
  python scripts/05_make_single_dataset_v2.py
  
  # 自定义增强参数
  python scripts/05_make_single_dataset_v2.py \\
    --amp-min 0.80 --amp-max 1.20 \\
    --noise-std-rel 0.005 \\
    --outdir results/single_pulse_v2_custom
        """
    )
    
    # 基本参数
    parser.add_argument('--train', type=str, default='single_split_train.mat',
                       help='训练集 .mat 路径')
    parser.add_argument('--test', type=str, default='single_split_test.mat',
                       help='测试集 .mat 路径')
    parser.add_argument('--outdir', type=str, default='results/single_pulse_v2',
                       help='输出目录（默认 results/single_pulse_v2）')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # v2 增强参数（默认值与 main profile 一致）
    parser.add_argument('--amp-min', type=float, default=0.85,
                       help='幅度缩放下限（默认 0.85）')
    parser.add_argument('--amp-max', type=float, default=1.15,
                       help='幅度缩放上限（默认 1.15）')
    
    parser.add_argument('--noise-enable', type=lambda x: x.lower() == 'true',
                       default=True, help='是否启用噪声（默认 True）')
    parser.add_argument('--noise-std-rel', type=float, default=0.003,
                       help='噪声标准差相对比例（默认 0.003）')
    
    parser.add_argument('--drift-enable', type=lambda x: x.lower() == 'true',
                       default=True, help='是否启用基线漂移（默认 True）')
    parser.add_argument('--drift-slope-max-rel', type=float, default=1e-4,
                       help='漂移斜率最大值相对比例（默认 1e-4）')
    parser.add_argument('--drift-lf-std-rel', type=float, default=0.001,
                       help='低频漂移标准差相对比例（默认 0.001）')
    parser.add_argument('--drift-lf-window', type=int, default=800,
                       help='低频漂移移动平均窗口点数（默认 800）')
    
    return parser.parse_args()


def generate_single_dataset_v2(X_single, y_single, split_name, outdir, seed, aug_cfg):
    """生成单脉冲增强数据集 v2
    
    Args:
        X_single: (N, L) 单脉冲波形
        y_single: (N,) 标签 (0=中子, 1=伽马)
        split_name: 'train' 或 'test'
        outdir: 输出目录
        seed: 随机种子
        aug_cfg: 增强配置字典
    """
    rng = np.random.default_rng(seed)
    fs_hz = 500e6
    L = X_single.shape[1]
    n_samples = len(X_single)
    
    # K=1 参数
    k_values = np.array([1])
    lambda_hz_array = np.array([0.0])  # K=1 不需要泊松到达率
    
    # 构造 comp_labels: (N, 3) 形状，[y, -1, -1]
    comp_labels = np.full((n_samples, 3), -1, dtype=np.int8)
    comp_labels[:, 0] = y_single
    
    # 调用 synthesize_pileup_samples_v2（它已支持 K=1 路径）
    result = synthesize_pileup_samples_v2(
        X_single, y_single, n_samples, lambda_hz_array, k_values,
        comp_labels, rng, 
        baseline_b=200,  # 对 K=1 不起作用，但保持参数一致性
        zero_prefix_b=False,  # K=1 不需要前缀置零
        fs_hz=fs_hz,
        # 增强参数
        amp_min=aug_cfg['amp_min'],
        amp_max=aug_cfg['amp_max'],
        noise_enable=aug_cfg['noise_enable'],
        noise_std_rel=aug_cfg['noise_std_rel'],
        drift_enable=aug_cfg['drift_enable'],
        drift_slope_max_rel=aug_cfg['drift_slope_max_rel'],
        drift_lf_std_rel=aug_cfg['drift_lf_std_rel'],
        drift_lf_window=aug_cfg['drift_lf_window'],
        # K=1 不需要可见性约束，使用极宽松的默认值
        min_visible_points=1,
        min_visible_energy_ratio=0.0,
        max_shift_resample=1,
        fail_policy='flag',
    )
    
    # 整理输出
    X = result['X']
    y_K = result['y_K']  # 全 1
    comp_labels_out = result['comp_labels']
    shifts_samples = result['shifts_samples']  # 全 [-1, -1]
    lambda_hz_out = result['lambda_hz']  # 全 0.0
    targets = result['targets']
    targets_mask = result['targets_mask']  # [1, 0, 0]
    truncated_flags = result['truncated_flags']
    visibility_metrics = result['visibility_metrics']
    
    # 保存
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    filename = f"single_{split_name}_v2.npz"
    filepath = outdir / filename
    
    save_pileup_dataset_v2(
        filepath, X, y_K, comp_labels_out, shifts_samples, lambda_hz_out,
        targets, targets_mask, truncated_flags, visibility_metrics, aug_cfg,
        fs_hz, L, baseline_b=200, zero_prefix_b=False, seed=seed
    )
    
    # 统计
    n_gamma = np.sum(y_single == 1)
    n_neutron = np.sum(y_single == 0)
    
    print(f"\n{'='*70}")
    print(f"SINGLE - {split_name.upper()} (v2)")
    print(f"{'='*70}")
    print(f"总样本数: {n_samples:,}")
    print(f"  Gamma: {n_gamma:,} ({100*n_gamma/n_samples:.1f}%)")
    print(f"  Neutron: {n_neutron:,} ({100*n_neutron/n_samples:.1f}%)")
    print(f"  K=1: {n_samples:,} (100.0%)")
    print(f"  y_is_pile: 全 0 (单脉冲)")
    print(f"\n保存至: {filepath}")
    print(f"{'='*70}\n")
    
    return filepath


def main():
    args = parse_args()
    
    # 读取数据
    print(f"Reading train set: {args.train}")
    X_train, y_train = load_single_dataset(args.train)
    n_gamma_train = np.sum(y_train == 1)
    n_neutron_train = np.sum(y_train == 0)
    print(f"  Gamma={n_gamma_train:,}, Neutron={n_neutron_train:,}, Total={len(y_train):,}")
    
    print(f"Reading test set: {args.test}")
    X_test, y_test = load_single_dataset(args.test)
    n_gamma_test = np.sum(y_test == 1)
    n_neutron_test = np.sum(y_test == 0)
    print(f"  Gamma={n_gamma_test:,}, Neutron={n_neutron_test:,}, Total={len(y_test):,}")
    
    # 增强配置
    aug_cfg = {
        'amp_min': args.amp_min,
        'amp_max': args.amp_max,
        'noise_enable': args.noise_enable,
        'noise_std_rel': args.noise_std_rel,
        'drift_enable': args.drift_enable,
        'drift_slope_max_rel': args.drift_slope_max_rel,
        'drift_lf_std_rel': args.drift_lf_std_rel,
        'drift_lf_window': args.drift_lf_window,
    }
    
    print(f"\n{'='*70}")
    print(f"增强配置 (v2):")
    print(f"{'='*70}")
    for key, val in aug_cfg.items():
        print(f"  {key}: {val}")
    print(f"{'='*70}\n")
    
    # 生成数据集
    generate_single_dataset_v2(X_train, y_train, 'train', args.outdir, args.seed, aug_cfg)
    generate_single_dataset_v2(X_test, y_test, 'test', args.outdir, args.seed + 1, aug_cfg)
    
    print(f"\nOK: Single pulse v2 datasets generated successfully at {args.outdir}")


if __name__ == '__main__':
    main()
