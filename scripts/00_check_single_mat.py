#!/usr/bin/env python3
"""
【单脉冲库检查脚本】inspect single-pulse train/test .mat files

功能：快速读取 train / test 的单脉冲库，输出样本数量与随机波形示例
    - train / test 中伽马、中子的数量与总数
    - 波形长度
    - 随机绘制训练集若干条单脉冲波形（混合 γ / n）
    - 【新增】用已知波形长度判轴、加载后断言+抽查峰值、配置项管理

默认数据路径：
    - 根目录 single_split_train.mat
    - 根目录 single_split_test.mat

用法示例：
    python scripts/00_check_single_mat.py
    python scripts/00_check_single_mat.py --train single_split_train.mat --test single_split_test.mat --plot-count 5
"""

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

# 配置字体（优先 DejaVu 以覆盖 µ 符号，再回退中文黑体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局配置项（方便后续修改采样率或窗口长度）
# 【做法3】把 shape 信息写入 meta / 配置
# ============================================================================
CONFIG = {
    "expected_waveform_length": 10002,  # 已知脉冲样本数
    "sampling_rate_mhz": 500.0,         # 采样率 (MHz)
    "peak_search_window_us": (1.5, 4.0),  # 预期峰值位置时间窗口 (µs)
    "num_samples_to_check": 10,         # 抽查脉冲数
}


def _ensure_pulse_first(arr: np.ndarray, expected_len: int) -> np.ndarray:
    """Ensure shape is [num_pulses, num_samples].
    
    【做法1】用"已知波形长度"判轴
    
    Args:
        arr: Input array (2D)
        expected_len: Expected waveform length (明确波形长度用于判轴)
    
    Returns:
        Array with shape [num_pulses, num_samples]
    
    Raises:
        ValueError: 如果无法确定正确的轴方向
    """
    if arr.ndim != 2:
        raise ValueError(f"Unexpected waveform ndim={arr.ndim}")
    
    h, w = arr.shape
    
    # 若 shape[1]==expected_len → 已是 [num_pulses, expected_len]
    if w == expected_len:
        logger.debug(f"Array shape {arr.shape}: 已是 [N, L={expected_len}]")
        return arr
    # 若 shape[0]==expected_len → 是 [expected_len, num_pulses]，需要转置
    elif h == expected_len:
        logger.debug(f"Array shape {arr.shape}: 转置为 [N, L={expected_len}]")
        return arr.T
    else:
        raise ValueError(
            f"无法判断轴方向！数组形状 {arr.shape}，期望长度 {expected_len}。\n"
            f"  若 shape[1]=={expected_len} → 已是 [N, L]\n"
            f"  若 shape[0]=={expected_len} → 需转置为 [N, L]\n"
            f"  都不匹配 → 请检查数据或显式指定配置"
        )


def _validate_and_check_peaks(label: str, waveforms: np.ndarray, expected_len: int, 
                              sampling_rate_mhz: float, peak_window_us: tuple, num_check: int = 10):
    """【做法2】加载后立刻断言 + 抽查峰值位置
    
    Args:
        label: 标签（'gamma' 或 'neutron'）
        waveforms: 波形数组，shape=[num_pulses, num_samples]
        expected_len: 期望的波形长度
        sampling_rate_mhz: 采样率 (MHz)
        peak_window_us: 预期峰值时间窗口 (t_min, t_max) 单位µs
        num_check: 抽查的脉冲数
    """
    # 断言形状
    assert waveforms.ndim == 2, f"{label}: 波形应为2D数组"
    assert waveforms.shape[1] == expected_len, \
        f"{label}: 期望长度 {expected_len}，实际 {waveforms.shape[1]}"
    
    logger.info(f"{label}: 形状验证通过 {waveforms.shape}")
    
    if waveforms.shape[0] == 0:
        logger.warning(f"{label}: 无样本，跳过峰值检查")
        return
    
    # 抽查峰值位置
    rng = np.random.default_rng(seed=42)  # 固定种子保证可复现
    check_indices = rng.choice(waveforms.shape[0], size=min(num_check, waveforms.shape[0]), replace=False)
    
    # 将时间窗口转换为样本索引范围
    t_min_us, t_max_us = peak_window_us
    idx_min = int(t_min_us * sampling_rate_mhz)
    idx_max = int(t_max_us * sampling_rate_mhz)
    
    peak_positions = []
    peaks_in_window = 0
    
    for idx in check_indices:
        wave = waveforms[idx]
        peak_idx = np.argmax(np.abs(wave))  # 绝对值最大值位置
        peak_time_us = peak_idx / sampling_rate_mhz
        peak_positions.append(peak_time_us)
        
        in_window = idx_min <= peak_idx <= idx_max
        if in_window:
            peaks_in_window += 1
            status = "✓"
        else:
            status = "✗"
        logger.debug(f"  {label}[{idx}]: 峰值 @ {peak_idx:5d} 样本 = {peak_time_us:.3f} µs {status}")
    
    ratio = peaks_in_window / len(check_indices) if check_indices.size > 0 else 0
    logger.info(f"{label}: {peaks_in_window}/{len(check_indices)} 峰值在预期窗口 "
                f"[{t_min_us:.2f}, {t_max_us:.2f}] µs (比例 {ratio*100:.1f}%)")
    
    if ratio < 0.7:
        logger.warning(f"{label}: 警告！只有 {ratio*100:.1f}% 的峰值在预期窗口内，"
                      f"可能是轴方向错误或数据异常")
    
    return {
        "peak_positions_us": peak_positions,
        "peaks_in_window": peaks_in_window,
        "total_checked": len(check_indices),
    }


def load_single_dataset(mat_path: Path, config: dict = None):
    """Load single pulse dataset (gamma / neutron).
    
    Args:
        mat_path: 路径到 .mat 文件
        config: 配置字典，包含 'expected_waveform_length' 等参数
    
    Returns:
        (gamma, neutron, meta) 元组
    """
    import h5py
    
    if config is None:
        config = CONFIG
    
    expected_len = config["expected_waveform_length"]
    sampling_rate = config["sampling_rate_mhz"]
    peak_window = config["peak_search_window_us"]
    num_check = config["num_samples_to_check"]

    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")

    with h5py.File(mat_path, "r") as f:
        # 【做法1】用"已知波形长度"判轴
        gamma = _ensure_pulse_first(np.array(f["single_G"]), expected_len)
        neutron = _ensure_pulse_first(np.array(f["single_N"]), expected_len)
        
        meta = {
            "gamma_P": np.array(f["single_G_P"]) if "single_G_P" in f else None,
            "gamma_E": np.array(f["single_G_E"]) if "single_G_E" in f else None,
            "neutron_P": np.array(f["single_N_P"]) if "single_N_P" in f else None,
            "neutron_E": np.array(f["single_N_E"]) if "single_N_E" in f else None,
            # 【做法3】把 shape 信息写入 meta/配置
            "config": config,
            "gamma_shape": gamma.shape,
            "neutron_shape": neutron.shape,
        }
    
    # 【做法2】加载后立刻断言 + 抽查峰值位置
    logger.info(f"验证伽马射线波形...")
    gamma_check = _validate_and_check_peaks(
        "gamma", gamma, expected_len, sampling_rate, peak_window, num_check
    )
    meta["gamma_validation"] = gamma_check
    
    logger.info(f"验证中子波形...")
    neutron_check = _validate_and_check_peaks(
        "neutron", neutron, expected_len, sampling_rate, peak_window, num_check
    )
    meta["neutron_validation"] = neutron_check

    return gamma, neutron, meta


def describe_dataset(name: str, gamma: np.ndarray, neutron: np.ndarray) -> dict:
    n_gamma = gamma.shape[0]
    n_neutron = neutron.shape[0]
    total = n_gamma + n_neutron
    length = gamma.shape[1] if n_gamma else neutron.shape[1]
    return {
        "name": name,
        "n_gamma": n_gamma,
        "n_neutron": n_neutron,
        "total": total,
        "length": length,
    }


def plot_random_training_waveforms(train_gamma: np.ndarray, train_neutron: np.ndarray, output_dir: Path, k: int = 5):
    rng = np.random.default_rng()
    waves = []
    labels = []

    if train_gamma.size:
        gamma_idx = rng.choice(train_gamma.shape[0], size=min(k, train_gamma.shape[0]), replace=False)
        waves.extend(train_gamma[gamma_idx])
        labels.extend([f"γ {i}" for i in gamma_idx])

    if len(waves) < k and train_neutron.size:
        need = k - len(waves)
        neutron_idx = rng.choice(train_neutron.shape[0], size=min(need, train_neutron.shape[0]), replace=False)
        waves.extend(train_neutron[neutron_idx])
        labels.extend([f"n {i}" for i in neutron_idx])

    waves = waves[:k]
    labels = labels[:k]

    if not waves:
        logger.warning("训练集中没有可用波形用于绘图")
        return None

    length = waves[0].shape[0]
    time_axis = np.arange(length) / 500.0  # 500 MHz -> µs

    colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#0EA5E9"]
    plt.figure(figsize=(14, 8))
    for idx, (wave, label) in enumerate(zip(waves, labels)):
        plt.plot(time_axis, wave, linewidth=1.2, alpha=0.9, color=colors[idx % len(colors)], label=label)

    plt.title("随机训练单脉冲波形 (混合 γ / n)", fontsize=14, fontweight="bold")
    plt.xlabel("时间 (µs)", fontsize=12)
    plt.ylabel("幅度 (a.u.)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_random_waveforms.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="读取单脉冲 train / test 数据并绘制训练波形示例")
    parser.add_argument("--train", type=str, default=str(repo_root / "single_split_train.mat"), help="训练集 .mat 路径")
    parser.add_argument("--test", type=str, default=str(repo_root / "single_split_test.mat"), help="测试集 .mat 路径")
    parser.add_argument("--plot-count", type=int, default=5, help="随机绘制的训练波形条数")
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)

    logger.info(f"读取 train: {train_path}")
    train_gamma, train_neutron, train_meta = load_single_dataset(train_path)
    logger.info(f"读取 test : {test_path}")
    test_gamma, test_neutron, test_meta = load_single_dataset(test_path)

    train_info = describe_dataset("train", train_gamma, train_neutron)
    test_info = describe_dataset("test", test_gamma, test_neutron)

    total_gamma = train_info["n_gamma"] + test_info["n_gamma"]
    total_neutron = train_info["n_neutron"] + test_info["n_neutron"]
    total_all = total_gamma + total_neutron

    print("\n" + "=" * 70)
    print("单脉冲库数据量")
    print("=" * 70)
    print(f"train: γ={train_info['n_gamma']:,}  n={train_info['n_neutron']:,}  total={train_info['total']:,}")
    print(f"test : γ={test_info['n_gamma']:,}  n={test_info['n_neutron']:,}  total={test_info['total']:,}")
    print("-" * 70)
    print(f"总体: γ={total_gamma:,}  n={total_neutron:,}  total={total_all:,}")
    print(f"波形长度: {train_info['length']} samples @500 MHz -> {train_info['length']/500:.2f} µs")
    print("=" * 70)
    
    # 【新增】输出验证信息
    print("\n" + "=" * 70)
    print("波形验证结果")
    print("=" * 70)
    if train_meta.get("gamma_validation"):
        gv = train_meta["gamma_validation"]
        print(f"训练伽马: {gv['peaks_in_window']}/{gv['total_checked']} 峰值在预期窗口内")
    if train_meta.get("neutron_validation"):
        nv = train_meta["neutron_validation"]
        print(f"训练中子: {nv['peaks_in_window']}/{nv['total_checked']} 峰值在预期窗口内")
    print("=" * 70 + "\n")

    output_dir = repo_root / "results" / "single_pulse"
    fig_path = plot_random_training_waveforms(train_gamma, train_neutron, output_dir, k=args.plot_count)

    if fig_path:
        print(f"已保存随机训练波形示例: {fig_path}")
    else:
        print("训练集中无可用波形，未生成图像。")


if __name__ == "__main__":
    main()
