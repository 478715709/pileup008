#!/usr/bin/env python3
"""
【生成堆积脉冲数据集脚本】
支持 Realistic 与 Balanced 两种模式，train/test 分别生成。
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# 加入 src 目录以导入自定义模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pileup.io_mat import load_single_dataset
from src.pileup.synth import synthesize_pileup_samples
from src.pileup.sampling import RealisticSampler, BalancedSampler
from src.pileup.utils import (
    save_pileup_dataset, get_tqdm, distribute_samples,
    print_dataset_stats, analyze_comp_labels
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成堆积脉冲数据集（Realistic 与 Balanced）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/01_make_piled_dataset.py \\
    --train single_split_train.mat \\
    --test single_split_test.mat \\
    --outdir results/piled_pulse \\
    --seed 42 \\
    --mix-mode both
        """
    )
    
    parser.add_argument('--train', type=str, default='single_split_train.mat',
                       help='训练集 .mat 路径')
    parser.add_argument('--test', type=str, default='single_split_test.mat',
                       help='测试集 .mat 路径')
    parser.add_argument('--outdir', type=str, default='results/piled_pulse',
                       help='输出目录')
    
    parser.add_argument('--lambda-mhz', type=float, nargs='+',
                       default=[0.1, 0.2, 0.4, 0.8, 1.5, 3.0],
                       help='λ 值数组 (MHz)')
    parser.add_argument('--pile-mult', type=int, nargs='+', default=[2, 3],
                       help='堆积重数 (2 和/或 3)')
    parser.add_argument('--ratio-3', type=float, default=0.5,
                       help='balanced 模式中 K=3 的占比')
    
    parser.add_argument('--n-pile-like-single', action='store_true', default=True,
                       help='堆积样本数 = 单脉冲总数 (默认启用)')
    parser.add_argument('--n-pile', type=int, default=None,
                       help='显式指定堆积样本数 (覆盖 --n-pile-like-single)')
    
    parser.add_argument('--baseline-b', type=int, default=200,
                       help='基线计算点数')
    parser.add_argument('--zero-prefix-b', type=bool, default=True,
                       help='是否将前 B 点置零')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save-targets', type=bool, default=True,
                       help='是否保存 targets（可选）')
    parser.add_argument('--progress', action='store_true', default=False,
                       help='显示进度条')
    
    parser.add_argument('--mix-mode', choices=['both', 'realistic', 'balanced'],
                       default='both', help='生成模式')
    
    return parser.parse_args()


def generate_pileup_dataset(
    X_single, y_single, split_name, n_samples, lambda_hz_array, k_values,
    outdir, mode='realistic', ratio_3=0.5, baseline_b=200, zero_prefix_b=True,
    seed=42, save_targets=True, progress=False
):
    """生成一份堆积脉冲数据集 (realistic 或 balanced)
    
    Args:
        X_single: (N, L) 单脉冲波形
        y_single: (N,) 标签
        split_name: 'train' 或 'test'
        n_samples: 堆积样本数
        lambda_hz_array: λ 值数组 (Hz)
        k_values: [2, 3] 等
        outdir: 输出目录
        mode: 'realistic' 或 'balanced'
        ratio_3: K=3 占比
        baseline_b: 基线点数
        zero_prefix_b: 置零前缀
        seed: 随机种子
        save_targets: 是否保存 targets
        progress: 是否显示进度
    """
    rng = np.random.default_rng(seed)
    fs_hz = 500e6
    L = X_single.shape[1]
    
    # 选择采样器
    if mode == 'realistic':
        sampler = RealisticSampler(X_single, y_single, rng)
        comp_labels = sampler.generate_comp_labels(n_samples, k_values, ratio_3)
    else:  # balanced
        sampler = BalancedSampler(X_single, y_single, rng)
        comp_labels = sampler.generate_comp_labels(n_samples, k_values, ratio_3)
    
    # 合成
    result = synthesize_pileup_samples(
        X_single, y_single, n_samples, lambda_hz_array, k_values,
        comp_labels, rng, baseline_b, zero_prefix_b, fs_hz
    )
    
    # 整理输出
    X = result['X']
    y_K = result['y_K']
    comp_labels_out = result['comp_labels']
    shifts_samples = result['shifts_samples']
    lambda_hz_out = result['lambda_hz']
    targets = result['targets'] if save_targets else None
    targets_mask = result['targets_mask']
    
    # 保存
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{mode}_{split_name}_pileup.npz"
    filepath = outdir / filename
    
    save_pileup_dataset(
        filepath, X, y_K, comp_labels_out, shifts_samples, lambda_hz_out,
        targets, targets_mask, fs_hz, L, baseline_b, zero_prefix_b, seed
    )
    
    # 统计
    n_k2 = np.sum(y_K == 2)
    n_k3 = np.sum(y_K == 3)
    
    lambda_counts = {}
    for lam in lambda_hz_out:
        lambda_counts[lam] = lambda_counts.get(lam, 0) + 1
    
    comp_counts = analyze_comp_labels(comp_labels_out)
    
    print_dataset_stats(mode, split_name, len(X), n_k2, n_k3, lambda_counts, comp_counts)
    
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
    
    # 决定堆积样本数
    if args.n_pile is not None:
        n_pile_train = args.n_pile
        n_pile_test = args.n_pile
    else:
        n_pile_train = len(y_train)
        n_pile_test = len(y_test)
    
    # 转换 λ 从 MHz 到 Hz
    lambda_hz_array = np.array(args.lambda_mhz) * 1e6
    k_values = np.array(args.pile_mult)
    
    # 生成数据集
    modes_to_run = []
    if args.mix_mode in ['both', 'realistic']:
        modes_to_run.append('realistic')
    if args.mix_mode in ['both', 'balanced']:
        modes_to_run.append('balanced')
    
    for mode in modes_to_run:
        print(f"\n{'='*70}")
        print(f"Generating {mode.upper()} mode dataset")
        print(f"{'='*70}\n")
        
        generate_pileup_dataset(
            X_train, y_train, 'train', n_pile_train, lambda_hz_array, k_values,
            args.outdir, mode, args.ratio_3, args.baseline_b, args.zero_prefix_b,
            args.seed, args.save_targets, args.progress
        )
        
        generate_pileup_dataset(
            X_test, y_test, 'test', n_pile_test, lambda_hz_array, k_values,
            args.outdir, mode, args.ratio_3, args.baseline_b, args.zero_prefix_b,
            args.seed + 1, args.save_targets, args.progress
        )
    
    print(f"\nOK: All datasets generated successfully at: {args.outdir}")


if __name__ == '__main__':
    main()
