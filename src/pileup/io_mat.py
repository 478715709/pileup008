"""读取.mat文件工具"""
import numpy as np
from pathlib import Path


def _ensure_pulse_first(arr: np.ndarray, expected_len: int) -> np.ndarray:
    """确保形状为 [num_pulses, num_samples]，用已知波形长度判轴"""
    if arr.ndim != 2:
        raise ValueError(f"Unexpected waveform ndim={arr.ndim}")
    
    h, w = arr.shape
    
    if w == expected_len:
        return arr
    elif h == expected_len:
        return arr.T
    else:
        raise ValueError(
            f"无法判断轴方向！数组形状 {arr.shape}，期望长度 {expected_len}。\n"
            f"  若 shape[1]=={expected_len} → 已是 [N, L]\n"
            f"  若 shape[0]=={expected_len} → 需转置为 [N, L]\n"
            f"  都不匹配 → 请检查数据或显式指定配置"
        )


def load_single_dataset(mat_path: str | Path, expected_len: int = 10002):
    """从.mat文件加载单脉冲数据集 - 返回合并的波形和标签池
    
    Args:
        mat_path: .mat文件路径
        expected_len: 期望的波形长度
    
    Returns:
        X_single: (N_total, L) 合并的波形（gamma + neutron）
        y_single: (N_total,) 标签（1=gamma, 0=neutron）
    """
    import h5py
    
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")
    
    with h5py.File(mat_path, 'r') as f:
        gamma = _ensure_pulse_first(np.array(f['single_G']), expected_len)
        neutron = _ensure_pulse_first(np.array(f['single_N']), expected_len)
    
    # 合并波形和标签
    X_single = np.vstack([gamma, neutron])
    y_single = np.concatenate([
        np.ones(len(gamma), dtype=np.int32),    # gamma = 1
        np.zeros(len(neutron), dtype=np.int32)  # neutron = 0
    ])
    
    return X_single, y_single
