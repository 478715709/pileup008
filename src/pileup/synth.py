"""堆积脉冲合成核心逻辑"""
import numpy as np


def remove_baseline(waveform: np.ndarray, baseline_b: int = 200) -> np.ndarray:
    """基线去除：用前 B 点均值作为基线"""
    baseline = np.mean(waveform[:baseline_b])
    return waveform - baseline


def synthesize_pileup_samples(
    X_single,  # (N_single, L) 单脉冲池
    y_single,  # (N_single,) 标签：1=gamma, 0=neutron
    n_samples,  # 要生成多少条堆积波形
    lambda_hz_array,  # 要遍历的 λ 值数组 (Hz)
    k_values,  # (2, 3) 或其他
    comp_labels_array,  # (n_samples, 3) 组合标签 或 None(则随机决定)
    rng,  # numpy.random.Generator
    baseline_b=200,
    zero_prefix_b=True,
    fs_hz=500e6,
    allow_truncate=True,
):
    """合成堆积脉冲样本
    
    Args:
        X_single: (N_single, L) 单脉冲波形池
        y_single: (N_single,) 标签池
        n_samples: 要生成的样本数
        lambda_hz_array: 各 λ 值数组
        k_values: [2, 3] 等
        comp_labels_array: (n_samples, 3) 指定每条样本的组合标签
                          或 None 则随机决定
        rng: np.random.Generator
        baseline_b: 基线计算点数
        zero_prefix_b: 是否将前 B 点置零
        fs_hz: 采样率 (Hz)
        allow_truncate: 是否允许截窗
    
    Returns:
        dict: {
            'X': (n_samples, L) 合成波形,
            'y_K': (n_samples,) 2 或 3,
            'comp_labels': (n_samples, 3) 组合标签,
            'shifts_samples': (n_samples, 2) shift2, shift3,
            'lambda_hz': (n_samples,) 每条样本的 λ,
            'targets': (n_samples, 3, L) or None,
            'targets_mask': (n_samples, 3),
        }
    """
    L = X_single.shape[1]
    dt_us_per_sample = 1e6 / fs_hz
    
    X_out = np.zeros((n_samples, L), dtype=np.float64)
    y_K_out = np.zeros(n_samples, dtype=np.int32)
    comp_labels_out = np.zeros((n_samples, 3), dtype=np.int32) - 1
    shifts_out = np.zeros((n_samples, 2), dtype=np.int32) - 1
    lambda_hz_out = np.zeros(n_samples, dtype=np.float64)
    targets_out = np.zeros((n_samples, 3, L), dtype=np.float64)
    targets_mask_out = np.zeros((n_samples, 3), dtype=np.int32)
    
    for i in range(n_samples):
        # 1. 决定 K 值（2 或 3）
        if comp_labels_array is not None:
            comp = comp_labels_array[i]
            K = np.sum(comp >= 0)
        else:
            K = rng.choice(k_values)
            comp = np.array([rng.choice([0, 1]) for _ in range(K)], dtype=np.int32)
            comp = np.concatenate([comp, np.full(3 - K, -1, dtype=np.int32)])
        
        y_K_out[i] = K
        comp_labels_out[i] = comp
        targets_mask_out[i, :K] = 1
        
        # 2. 选择 λ
        lambda_hz = rng.choice(lambda_hz_array)
        lambda_hz_out[i] = lambda_hz
        lambda_per_sec = lambda_hz
        
        # 3. 抽样脉冲（允许有放回）
        pulses = []
        for k in range(K):
            idx = rng.choice(len(y_single))
            pulses.append((X_single[idx], y_single[idx]))
        
        # 4. 时间间隔采样（Exp(λ)）
        shifts = [-1, -1]  # shift2, shift3
        
        # 首脉冲在 0，直接叠加（不去基线）
        x_out = pulses[0][0].copy()
        targets_out[i, 0] = pulses[0][0]  # 首脉冲原样
        
        if K >= 2:
            # 第二脉冲的时间间隔（指数分布）
            if lambda_per_sec > 0:
                # rng.exponential 的尺度参数以“秒”为单位，这里换算为“微秒”以与 dt_us_per_sample 一致
                dt2_us = rng.exponential(1.0 / lambda_per_sec) * 1e6
                shift2 = int(np.round(dt2_us / dt_us_per_sample))
            else:
                shift2 = L  # 无穷远
            
            shifts[0] = min(shift2, L - 1)
            
            # 去基线 + 可选置零前缀
            pulse2 = pulses[1][0].copy()
            pulse2 = remove_baseline(pulse2, baseline_b)
            if zero_prefix_b:
                pulse2[:baseline_b] = 0
            
            # 移位叠加（允许截窗）
            if shift2 < L:
                if shift2 + len(pulse2) <= L:
                    x_out[shift2:shift2 + len(pulse2)] += pulse2
                else:
                    # 截断
                    remain = L - shift2
                    x_out[shift2:] += pulse2[:remain]
            
            # 记录 targets（去基线后的脉冲，按shift放入）
            targets_out[i, 1, :] = 0
            if shift2 < L:
                if shift2 + len(pulse2) <= L:
                    targets_out[i, 1, shift2:shift2 + len(pulse2)] = pulse2
                else:
                    targets_out[i, 1, shift2:] = pulse2[:L - shift2]
        
        if K == 3:
            # 第三脉冲：t3 = t2 + Δt3
            if lambda_per_sec > 0:
                # 同上：将“秒”换算为“微秒”
                dt3_us = rng.exponential(1.0 / lambda_per_sec) * 1e6
            else:
                dt3_us = 0
            
            shift3 = shifts[0] + int(np.round(dt3_us / dt_us_per_sample))
            shifts[1] = min(shift3, L - 1)
            
            pulse3 = pulses[2][0].copy()
            pulse3 = remove_baseline(pulse3, baseline_b)
            if zero_prefix_b:
                pulse3[:baseline_b] = 0
            
            if shift3 < L:
                if shift3 + len(pulse3) <= L:
                    x_out[shift3:shift3 + len(pulse3)] += pulse3
                else:
                    remain = L - shift3
                    x_out[shift3:] += pulse3[:remain]
            
            targets_out[i, 2, :] = 0
            if shift3 < L:
                if shift3 + len(pulse3) <= L:
                    targets_out[i, 2, shift3:shift3 + len(pulse3)] = pulse3
                else:
                    targets_out[i, 2, shift3:] = pulse3[:L - shift3]
        
        X_out[i] = x_out
        shifts_out[i] = shifts
    
    return {
        'X': X_out,
        'y_K': y_K_out,
        'comp_labels': comp_labels_out,
        'shifts_samples': shifts_out,
        'lambda_hz': lambda_hz_out,
        'targets': targets_out,
        'targets_mask': targets_mask_out,
    }
