"""工具函数 v2 - 扩展用于 v2 数据集"""
import numpy as np
from pathlib import Path


def save_pileup_dataset_v2(filepath, X, y_K, comp_labels, shifts_samples, lambda_hz, targets,
                           targets_mask, truncated_flags, visibility_metrics, aug_cfg,
                           fs_hz, L, baseline_b, zero_prefix_b, seed):
    """保存堆积脉冲数据集 v2 为 npz"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 将 aug_cfg 转为可序列化格式
    if isinstance(aug_cfg, dict):
        aug_cfg_str = str(aug_cfg)
    else:
        aug_cfg_str = aug_cfg
    
    # y_is_pile: K>1为1 (堆积), K=1为0 (单脉冲)
    y_is_pile = (y_K > 1).astype(np.int8)
    
    np.savez(
        filepath,
        X=X,
        y_is_pile=y_is_pile,
        y_K=y_K,
        comp_labels=comp_labels,
        shifts_samples=shifts_samples,
        lambda_hz=lambda_hz,
        targets=targets,
        targets_mask=targets_mask,
        truncated_flags=truncated_flags,
        visibility_metrics=visibility_metrics,
        aug_cfg=aug_cfg_str,
        fs_hz=np.float64(fs_hz),
        L=np.int32(L),
        baseline_b=np.int32(baseline_b),
        zero_prefix_b=np.int32(int(zero_prefix_b)),
        seed=np.int32(seed),
    )


def load_pileup_dataset_v2(filepath):
    """加载堆积脉冲数据集 v2"""
    data = np.load(filepath, allow_pickle=True)
    return {k: data[k] for k in data.files}


def print_dataset_stats_v2(mode, split, n_samples, n_k2, n_k3, lambda_counts, comp_counts,
                           truncated_stats):
    """打印数据集统计信息 v2（包含截断信息）"""
    print(f"\n{'='*70}")
    print(f"{mode.upper()} - {split.upper()} (v2)")
    print(f"{'='*70}")
    print(f"总样本数: {n_samples:,}")
    print(f"  K=2: {n_k2:,} ({100*n_k2/n_samples:.1f}%)")
    print(f"  K=3: {n_k3:,} ({100*n_k3/n_samples:.1f}%)")
    
    print(f"\nλ 分布 (Hz):")
    for lambda_hz, count in sorted(lambda_counts.items()):
        print(f"  λ={lambda_hz:7.1f}: {count:6,} ({100*count/n_samples:5.1f}%)")
    
    print(f"\n组合标签分布:")
    for comp, count in sorted(comp_counts.items()):
        print(f"  {comp}: {count:6,} ({100*count/n_samples:5.1f}%)")
    
    print(f"\n截断/不可见统计:")
    print(f"  Component 1 截断: {truncated_stats[0]:6,} ({100*truncated_stats[0]/n_samples:5.1f}%)")
    print(f"  Component 2 截断: {truncated_stats[1]:6,} ({100*truncated_stats[1]/n_samples:5.1f}%)")
    print(f"  Component 3 截断: {truncated_stats[2]:6,} ({100*truncated_stats[2]/n_samples:5.1f}%)")
    
    print(f"{'='*70}\n")
