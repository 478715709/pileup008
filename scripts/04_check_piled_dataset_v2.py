#!/usr/bin/env python3
"""
【检查堆积脉冲数据集脚本 v2】
支持加载与可视化 npz 数据集 v2，包含增强与可见性统计。
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pileup.utils import analyze_comp_labels
from src.pileup.utils_v2 import load_pileup_dataset_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查堆积脉冲 npz 数据集 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 验收并出图
  python scripts/04_check_piled_dataset_v2.py \\
    --npz results/piled_pulse_v2/realistic_train_pileup_v2.npz \\
          results/piled_pulse_v2/balanced_train_pileup_v2.npz \\
    --plot --plot-dir results/piled_pulse_v2/figs
        """
    )
    
    parser.add_argument('--npz', type=str, nargs='+', required=True,
                       help='npz 文件路径（可多个）')
    parser.add_argument('--plot', action='store_true', default=False,
                       help='生成可视化图')
    parser.add_argument('--plot-dir', type=str, default='results/piled_pulse_v2/figs',
                       help='图片输出目录')
    
    return parser.parse_args()


def check_single_dataset_v2(filepath):
    """Check a single npz v2 file"""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return None
    
    data = load_pileup_dataset_v2(filepath)
    
    # 识别 profile (main 或 hard)
    filename = filepath.name
    if '_hard' in filename:
        profile_tag = '(hard)'
    else:
        profile_tag = '(main)'
    
    data['_profile_tag'] = profile_tag
    data['_filepath'] = str(filepath)
    
    X = data['X']
    y_K = data['y_K']
    comp_labels = data['comp_labels']
    shifts_samples = data['shifts_samples']
    lambda_hz = data['lambda_hz']
    targets_mask = data['targets_mask']
    truncated_flags = data.get('truncated_flags')
    visibility_metrics = data.get('visibility_metrics')
    aug_cfg = data.get('aug_cfg')
    
    print(f"\n{'='*70}")
    print(f"Dataset: {filepath.name} {profile_tag}")
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
    
    # v2 特有：截断统计
    if truncated_flags is not None:
        print(f"\nTruncated/Invisible stats (v2):")
        for k in range(3):
            count = np.sum(truncated_flags[:, k] > 0)
            print(f"  Component {k+1} truncated: {count:,} ({100*count/n_samples:.1f}%)")
    
    # v2 特有：可见能量统计
    if visibility_metrics is not None:
        print(f"\nVisibility energy ratio stats (v2):")
        for k in range(3):
            valid_mask = targets_mask[:, k] > 0
            if np.sum(valid_mask) > 0:
                ratios = visibility_metrics[valid_mask, k]
                print(f"  Component {k+1}: min={np.min(ratios):.3f}, "
                      f"mean={np.mean(ratios):.3f}, max={np.max(ratios):.3f}")
    
    # v2 特有：增强配置
    if aug_cfg is not None:
        print(f"\nAugmentation config (v2):")
        print(f"  {aug_cfg}")
    
    print()
    return data


def visualize_samples_v2(data, output_dir, name):
    """生成可视化图 v2"""
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
    profile_tag = data.get('_profile_tag', '')
    
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
        ax.set_title(f"Sample {idx}: comp={tuple(comp)} {profile_tag}")
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
        
        ax.set_title(f"Sample {idx} Targets Decomposition (comp={tuple(comp_labels[idx])}) {profile_tag}")
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        fig.tight_layout()
        output_path = output_dir / f"{name}_targets_decomp.png"
        fig.savefig(output_path, dpi=100)
        plt.close()
        print(f"OK: Saved {output_path}")


def compare_comp_labels_v2(datasets_dict, output_dir):
    """对比多个数据集的组合标签分布 v2"""
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
        profile_tag = data.get('_profile_tag', '')
        comp_labels = data['comp_labels']
        comp_counts = analyze_comp_labels(comp_labels)
        
        # 排序以确保顺序一致
        sorted_comps = sorted(comp_counts.keys())
        counts = [comp_counts[c] for c in sorted_comps]
        percentages = [100 * c / len(comp_labels) for c in counts]
        
        x_positions = np.arange(len(sorted_comps))
        ax.plot(x_positions, percentages, marker='o', linewidth=2, 
               markersize=8, label=f"{name} {profile_tag}", alpha=0.7)
    
    # Set labels
    ax.set_xlabel('Composition Label', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Composition Label Distribution Comparison (v2)', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_comps, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    output_path = output_dir / "comp_labels_comparison.png"
    fig.savefig(output_path, dpi=100)
    plt.close()
    print(f"OK: Saved comparison plot {output_path}")


def compare_main_hard_metrics(datasets_dict, output_dir):
    """对比 main 和 hard 的截断/可见性指标"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping metrics comparison")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集 main 和 hard 数据
    main_data = None
    hard_data = None
    
    for npz_path, data in datasets_dict.items():
        if '_hard' in npz_path or '(hard)' in data.get('_profile_tag', ''):
            hard_data = data
        else:
            main_data = data
    
    if main_data is None or hard_data is None:
        print("WARNING: Need both main and hard datasets for comparison")
        return
    
    # 对比截断率
    print(f"\n{'='*70}")
    print(f"Comparing truncation rates: main vs hard")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for k in range(3):
        main_trunc = main_data['truncated_flags'][:, k] if main_data.get('truncated_flags') is not None else None
        hard_trunc = hard_data['truncated_flags'][:, k] if hard_data.get('truncated_flags') is not None else None
        
        if main_trunc is not None and hard_trunc is not None:
            main_rate = 100 * np.sum(main_trunc > 0) / len(main_trunc)
            hard_rate = 100 * np.sum(hard_trunc > 0) / len(hard_trunc)
            
            print(f"  Component {k+1}:")
            print(f"    main: {main_rate:.1f}%")
            print(f"    hard: {hard_rate:.1f}%")
            
            ax = axes[k]
            labels = ['main', 'hard']
            rates = [main_rate, hard_rate]
            colors = ['blue', 'orange']
            ax.bar(labels, rates, color=colors, alpha=0.7)
            ax.set_ylabel('Truncation Rate (%)')
            ax.set_title(f'Component {k+1} Truncation Rate')
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    output_path = output_dir / "main_hard_truncation_comparison.png"
    fig.savefig(output_path, dpi=100)
    plt.close()
    print(f"OK: Saved {output_path}")
    
    # 对比可见性能量比
    print(f"\n{'='*70}")
    print(f"Comparing visibility energy ratios: main vs hard")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for k in range(3):
        main_vis = main_data.get('visibility_metrics')
        hard_vis = hard_data.get('visibility_metrics')
        main_mask = main_data['targets_mask'][:, k] > 0
        hard_mask = hard_data['targets_mask'][:, k] > 0
        
        if main_vis is not None and hard_vis is not None:
            main_ratios = main_vis[main_mask, k]
            hard_ratios = hard_vis[hard_mask, k]
            
            print(f"  Component {k+1}:")
            print(f"    main: min={np.min(main_ratios):.3f}, "
                  f"mean={np.mean(main_ratios):.3f}, max={np.max(main_ratios):.3f}")
            print(f"    hard: min={np.min(hard_ratios):.3f}, "
                  f"mean={np.mean(hard_ratios):.3f}, max={np.max(hard_ratios):.3f}")
            
            ax = axes[k]
            ax.hist(main_ratios, bins=30, alpha=0.5, label='main', color='blue')
            ax.hist(hard_ratios, bins=30, alpha=0.5, label='hard', color='orange')
            ax.set_xlabel('Energy Ratio')
            ax.set_ylabel('Count')
            ax.set_title(f'Component {k+1} Energy Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    output_path = output_dir / "main_hard_visibility_comparison.png"
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
        
        # 对比图（如果有多个数据集）
        if len(datasets) > 1:
            print(f"\nGenerating comparison plot...")
            compare_comp_labels_v2(datasets, args.plot_dir)
            
            # 如果包含 main 和 hard，对比截断/可见性指标
            has_main = any('_hard' not in p for p in datasets.keys())
            has_hard = any('_hard' in p for p in datasets.keys())
            if has_main and has_hard:
                print(f"\nGenerating main/hard metrics comparison...")
                compare_main_hard_metrics(datasets, args.plot_dir)
    
    print(f"\nOK: Check completed (v2)")


if __name__ == '__main__':
    main()
