"""工具函数"""
import numpy as np
from pathlib import Path


def save_pileup_dataset(filepath, X, y_K, comp_labels, shifts_samples, lambda_hz, targets,
                        targets_mask, fs_hz, L, baseline_b, zero_prefix_b, seed):
    """保存堆积脉冲数据集为 npz (不压缩以加快速度)"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        filepath,
        X=X,
        y_is_pile=np.ones(len(X), dtype=np.int32),
        y_K=y_K,
        comp_labels=comp_labels,
        shifts_samples=shifts_samples,
        lambda_hz=lambda_hz,
        targets=targets,
        targets_mask=targets_mask,
        fs_hz=np.float64(fs_hz),
        L=np.int32(L),
        baseline_b=np.int32(baseline_b),
        zero_prefix_b=np.int32(int(zero_prefix_b)),
        seed=np.int32(seed),
    )


def load_pileup_dataset(filepath):
    """加载堆积脉冲数据集"""
    data = np.load(filepath)
    return {k: data[k] for k in data.files}


def get_tqdm():
    """导入 tqdm 或 fallback 到虚拟实现"""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        # Fallback: 虚拟进度条
        class FakeTqdm:
            def __init__(self, iterable, **kwargs):
                self.iterable = iterable
            
            def __iter__(self):
                return iter(self.iterable)
        
        return FakeTqdm


def distribute_samples(total, num_groups):
    """将 total 样本尽量均匀分配到 num_groups 组
    
    Returns:
        [count0, count1, ...] 长度为 num_groups
    """
    base = total // num_groups
    remainder = total % num_groups
    result = [base + (1 if i < remainder else 0) for i in range(num_groups)]
    return result


def print_dataset_stats(mode, split, n_samples, n_k2, n_k3, lambda_counts, comp_counts):
    """打印数据集统计信息"""
    print(f"\n{'='*70}")
    print(f"{mode.upper()} - {split.upper()}")
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
    print(f"{'='*70}\n")


def analyze_comp_labels(comp_labels):
    """统计组合标签分布"""
    counts = {}
    for comp in comp_labels:
        # 转换为元组用于dict key
        key = tuple(comp)
        counts[key] = counts.get(key, 0) + 1
    return counts
