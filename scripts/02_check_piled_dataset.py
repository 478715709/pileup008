#!/usr/bin/env python3
"""
【检查堆积脉冲数据集脚本】
支持加载与可视化 npz 数据集。
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pileup.utils import load_pileup_dataset, analyze_comp_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查堆积脉冲 npz 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/02_check_piled_dataset.py \\
    --npz results/piled_pulse/realistic_train_pileup.npz
  
  python scripts/02_check_piled_dataset.py \\
    --npz results/piled_pulse/realistic_train_pileup.npz \\
        results/piled_pulse/balanced_train_pileup.npz
        """
    )
    
    parser.add_argument('--npz', type=str, nargs='+', required=True,
                       help='npz 文件路径（可多个）')
    parser.add_argument('--plot', action='store_true', default=False,
                       help='生成可视化图')
    parser.add_argument('--plot-dir', type=str, default='results/piled_pulse/figs',
                       help='图片输出目录')
    
    return parser.parse_args()


def check_single_dataset(filepath):
    """Check a single npz file"""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return None
    
    data = load_pileup_dataset(filepath)
    
    X = data['X']
    y_K = data['y_K']
    comp_labels = data['comp_labels']
    shifts_samples = data['shifts_samples']
    lambda_hz = data['lambda_hz']
    targets_mask = data['targets_mask']
    
    print(f"\n{'='*70}")
    print(f"Dataset: {filepath.name}")
    print(f"{'='*70}")
    
    # Basic stats
    n_samples = len(X)
    L = X.shape[1]
    print(f"Num samples: {n_samples:,}")
    print(f"Waveform length: {L}")
    
    # K stats
    n_k2 = np.sum(y_K == 2)
    n_k3 = np.sum(y_K == 3)
    print(f"\nPileup multiplicity:")
    print(f"  K=2: {n_k2:,} ({100*n_k2/n_samples:.1f}%)")
    print(f"  K=3: {n_k3:,} ({100*n_k3/n_samples:.1f}%)")
    
    # Lambda stats
    unique_lambda = np.unique(lambda_hz)
    print(f"\nLambda distribution ({len(unique_lambda)} bins):")
    for lam in sorted(unique_lambda):
        count = np.sum(lambda_hz == lam)
        print(f"  L={lam:11.1f} Hz (MHz={lam/1e6:5.1f}): {count:6,} ({100*count/n_samples:5.1f}%)")
    
    # Shifts stats
    print(f"\nShift distribution:")
    valid_shifts = shifts_samples[shifts_samples >= 0]
    if len(valid_shifts) > 0:
        print(f"  min: {np.min(valid_shifts):,}, mean: {np.mean(valid_shifts):.0f}, max: {np.max(valid_shifts):,}")
    
    # Comp labels stats
    comp_counts = analyze_comp_labels(comp_labels)
    print(f"\nComposition label distribution ({len(comp_counts)} types):")
    for comp, count in sorted(comp_counts.items()):
        print(f"  {comp}: {count:6,} ({100*count/n_samples:5.1f}%)")
    
    # Targets mask stats
    print(f"\nTargets mask stats:")
    for k in range(3):
        count = np.sum(targets_mask[:, k] > 0)
        print(f"  Component {k+1}: {count:,} ({100*count/n_samples:.1f}%)")
    
    print()
    return data


def visualize_samples(data, output_dir, name):
    """生成可视化图"""
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
    shifts_samples = data['shifts_samples']
    targets = data.get('targets')
    fs_hz = float(data['fs_hz'])
    
    # 随机抽 5 条波形
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=min(5, len(X)), replace=False)
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    if len(indices) == 1:
        axes = [axes]
    
    dt_us = 1e6 / fs_hz
    
    for ax, idx in zip(axes, indices):
        x = X[idx]
        time_axis = np.arange(len(x)) * dt_us
        ax.plot(time_axis, x, linewidth=1, label='Composite waveform', color='blue')
        
        # Mark shifts
        shift2, shift3 = shifts_samples[idx]
        if shift2 >= 0:
            ax.axvline(x=shift2 * dt_us, color='orange', linestyle='--', alpha=0.5, label='shift2')
        if shift3 >= 0:
            ax.axvline(x=shift3 * dt_us, color='red', linestyle='--', alpha=0.5, label='shift3')
        
        comp = comp_labels[idx]
        ax.set_title(f"Sample {idx}: comp={tuple(comp)}")
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        if shift2 >= 0:
            ax.legend()
    
    fig.tight_layout()
    output_path = output_dir / f"{name}_samples.png"
    fig.savefig(output_path, dpi=100)
    plt.close()
    print(f"OK: Saved {output_path}")
    
    # If we have targets, plot one sample's targets decomposition
    if targets is not None and len(X) > 0:
        idx = indices[0]
        x = X[idx]
        target = targets[idx]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        time_axis = np.arange(len(x)) * dt_us
        
        # Original composite waveform
        ax.plot(time_axis, x, linewidth=2, label='Composite X', color='black', alpha=0.7)
        
        # Three components
        colors = ['blue', 'orange', 'red']
        labels = ['Component 1', 'Component 2', 'Component 3']
        for k in range(3):
            ax.plot(time_axis, target[k], linewidth=1.5, label=labels[k], 
                   color=colors[k], alpha=0.6, linestyle='--')
        
        # Sum of components
        target_sum = np.sum(target, axis=0)
        ax.plot(time_axis, target_sum, linewidth=2, label='Component sum', 
               color='green', alpha=0.6, linestyle=':')
        
        ax.set_title(f"Sample {idx} Targets Decomposition (comp={tuple(comp_labels[idx])})")
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        fig.tight_layout()
        output_path = output_dir / f"{name}_targets_decomp.png"
        fig.savefig(output_path, dpi=100)
        plt.close()
        print(f"OK: Saved {output_path}")


def compare_comp_labels(datasets_dict, output_dir):
    """对比多个数据集的组合标签分布"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping comparison")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for npz_path, data in datasets_dict.items():
        name = Path(npz_path).stem
        comp_labels = data['comp_labels']
        comp_counts = analyze_comp_labels(comp_labels)
        
        # 排序以确保顺序一致
        sorted_comps = sorted(comp_counts.keys())
        counts = [comp_counts[c] for c in sorted_comps]
        percentages = [100 * c / len(comp_labels) for c in counts]
        
        x_positions = np.arange(len(sorted_comps))
        ax.plot(x_positions, percentages, marker='o', linewidth=2, 
               markersize=8, label=name, alpha=0.7)
    
    # Set labels
    ax.set_xlabel('Composition Label', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Composition Label Distribution Comparison', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_comps, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    output_path = output_dir / "comp_labels_comparison.png"
    fig.savefig(output_path, dpi=100)
    plt.close()
    print(f"OK: Saved comparison plot {output_path}")


def main():
    args = parse_args()
    
    datasets = {}
    for npz_path in args.npz:
        data = check_single_dataset(npz_path)
        if data is not None:
            datasets[npz_path] = data
    
    if args.plot:
        print(f"\nGenerating plots...")
        for npz_path, data in datasets.items():
            name = Path(npz_path).stem
            visualize_samples(data, args.plot_dir, name)
        
        # 对比图（如果有多个数据集）
        if len(datasets) > 1:
            print(f"\nGenerating comparison plot...")
            compare_comp_labels(datasets, args.plot_dir)
    
    print(f"\nOK: Check completed")


if __name__ == '__main__':
    main()
