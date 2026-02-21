#!/usr/bin/env python3
"""
【检查单脉冲增强数据集脚本 v2】
验证 K=1 单脉冲增强数据集的字段格式与统计
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pileup.utils_v2 import load_pileup_dataset_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查单脉冲增强 npz 数据集 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 验收并出图
  python scripts/06_check_single_dataset_v2.py \\
    --npz results/single_pulse_v2/single_train_v2.npz \\
          results/single_pulse_v2/single_test_v2.npz \\
    --plot --plot-dir results/single_pulse_v2/figs
        """
    )
    
    parser.add_argument('--npz', type=str, nargs='+', required=True,
                       help='npz 文件路径（可多个）')
    parser.add_argument('--plot', action='store_true', default=False,
                       help='生成可视化图')
    parser.add_argument('--plot-dir', type=str, default='results/single_pulse_v2/figs',
                       help='图片输出目录')
    
    return parser.parse_args()


def check_single_dataset_v2(filepath):
    """Check a single npz v2 file"""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return None
    
    data = load_pileup_dataset_v2(filepath)
    
    X = data['X']
    y_is_pile = data['y_is_pile']
    y_K = data['y_K']
    comp_labels = data['comp_labels']
    shifts_samples = data['shifts_samples']
    lambda_hz = data['lambda_hz']
    targets = data.get('targets')
    targets_mask = data['targets_mask']
    truncated_flags = data.get('truncated_flags')
    visibility_metrics = data.get('visibility_metrics')
    aug_cfg = data.get('aug_cfg')
    
    print(f"\n{'='*70}")
    print(f"Dataset: {filepath.name}")
    print(f"{'='*70}")
    
    # Basic stats
    n_samples = len(X)
    L = X.shape[1]
    print(f"Num samples: {n_samples:,}")
    print(f"Waveform length: {L}")
    
    # K stats
    print(f"\nK distribution:")
    unique_k, counts_k = np.unique(y_K, return_counts=True)
    for k, count in zip(unique_k, counts_k):
        print(f"  K={k}: {count:,} ({100*count/n_samples:.1f}%)")
    
    # y_is_pile check
    print(f"\ny_is_pile check:")
    n_pile = np.sum(y_is_pile == 1)
    n_single = np.sum(y_is_pile == 0)
    print(f"  Pileup (y_is_pile=1): {n_pile:,} ({100*n_pile/n_samples:.1f}%)")
    print(f"  Single (y_is_pile=0): {n_single:,} ({100*n_single/n_samples:.1f}%)")
    if n_single == n_samples:
        print(f"  ✓ All samples are single pulses (K=1)")
    
    # Class distribution (from comp_labels[:,0])
    y_class = comp_labels[:, 0]
    print(f"\nClass distribution:")
    n_gamma = np.sum(y_class == 1)
    n_neutron = np.sum(y_class == 0)
    print(f"  Gamma (y=1): {n_gamma:,} ({100*n_gamma/n_samples:.1f}%)")
    print(f"  Neutron (y=0): {n_neutron:,} ({100*n_neutron/n_samples:.1f}%)")
    
    # comp_labels check
    print(f"\ncomp_labels check:")
    print(f"  Shape: {comp_labels.shape}")
    print(f"  Sample comp_labels[0]: {comp_labels[0]}")
    if np.all(comp_labels[:, 1:] == -1):
        print(f"  ✓ comp_labels[:, 1:] all -1 (K=1)")
    
    # shifts_samples check
    print(f"\nshifts_samples check:")
    print(f"  Shape: {shifts_samples.shape}")
    print(f"  Sample shifts_samples[0]: {shifts_samples[0]}")
    if np.all(shifts_samples == -1):
        print(f"  ✓ shifts_samples all -1 (K=1)")
    
    # lambda_hz check
    print(f"\nlambda_hz check:")
    unique_lambda = np.unique(lambda_hz)
    print(f"  Unique values: {unique_lambda}")
    if np.all(lambda_hz == 0.0):
        print(f"  ✓ lambda_hz all 0.0 (K=1)")
    
    # targets_mask check
    print(f"\ntargets_mask check:")
    print(f"  Shape: {targets_mask.shape}")
    print(f"  Sample targets_mask[0]: {targets_mask[0]}")
    n_comp1 = np.sum(targets_mask[:, 0] == 1)
    n_comp2 = np.sum(targets_mask[:, 1] == 1)
    n_comp3 = np.sum(targets_mask[:, 2] == 1)
    print(f"  Component 1 present: {n_comp1:,} ({100*n_comp1/n_samples:.1f}%)")
    print(f"  Component 2 present: {n_comp2:,} ({100*n_comp2/n_samples:.1f}%)")
    print(f"  Component 3 present: {n_comp3:,} ({100*n_comp3/n_samples:.1f}%)")
    if n_comp1 == n_samples and n_comp2 == 0 and n_comp3 == 0:
        print(f"  ✓ Only Component 1 present (K=1)")
    
    # Augmentation config
    if aug_cfg is not None:
        print(f"\nAugmentation config (v2):")
        print(f"  {aug_cfg}")
    
    print()
    return data


def visualize_samples_v2(data, output_dir, name):
    """生成可视化图 v2 - 在一张图上画随机波形"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping visualization")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X = data['X']
    comp_labels = data['comp_labels']
    fs_hz = float(data['fs_hz'])
    
    # 随机抽 10 条波形，在一张图上画
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=min(10, len(X)), replace=False)
    
    # 颜色方案：Gamma 用蓝色系，Neutron 用红色系
    colors_gamma = plt.cm.Blues(np.linspace(0.5, 1.0, 10))
    colors_neutron = plt.cm.Reds(np.linspace(0.5, 1.0, 10))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    dt_us = 1e6 / fs_hz
    
    gamma_count = 0
    neutron_count = 0
    
    for i, idx in enumerate(indices):
        x = X[idx]
        time_axis = np.arange(len(x)) * dt_us
        y_class = comp_labels[idx, 0]
        
        if y_class == 1:
            color = colors_gamma[gamma_count]
            label = f'Gamma (y={idx})' if gamma_count == 0 else None
            gamma_count += 1
        else:
            color = colors_neutron[neutron_count]
            label = f'Neutron (y={idx})' if neutron_count == 0 else None
            neutron_count += 1
        
        ax.plot(time_axis, x, linewidth=1.5, color=color, alpha=0.7, label=label)
    
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Amplitude (a.u.)', fontsize=12)
    ax.set_title(f'Single Pulse Augmented Waveforms (K=1) - {name.upper()}\n(mixed gamma/neutron)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    fig.tight_layout()
    output_path = output_dir / f"{name}_samples.png"
    fig.savefig(output_path, dpi=100)
    plt.close()
    print(f"OK: Saved {output_path}")


def main():
    args = parse_args()
    
    datasets = {}
    for npz_path in args.npz:
        data = check_single_dataset_v2(npz_path)
        if data is not None:
            datasets[npz_path] = data
    
    if args.plot:
        print(f"\nGenerating plots...")
        for npz_path, data in datasets.items():
            name = Path(npz_path).stem
            visualize_samples_v2(data, args.plot_dir, name)
    
    print(f"\nOK: Check completed (v2)")


if __name__ == '__main__':
    main()
