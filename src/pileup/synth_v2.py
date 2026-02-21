"""堆积脉冲合成核心逻辑 v2 - 增加增强与可见性约束"""
import numpy as np
from .augment import (
    augment_component, augment_composite, 
    resample_shift_until_visible, compute_visible_energy_ratio
)


def remove_baseline(waveform: np.ndarray, baseline_b: int = 200) -> np.ndarray:
    """基线去除：用前 B 点均值作为基线"""
    baseline = np.mean(waveform[:baseline_b])
    return waveform - baseline


def synthesize_pileup_samples_v2(
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
    # 增强参数
    amp_min=0.85,
    amp_max=1.15,
    noise_enable=True,
    noise_std_rel=0.003,
    drift_enable=True,
    drift_slope_max_rel=1e-4,
    drift_lf_std_rel=0.001,
    drift_lf_window=800,
    # 可见性约束参数
    min_visible_points=600,
    min_visible_energy_ratio=0.15,
    max_shift_resample=50,
    fail_policy='flag',  # 'flag' or 'drop'
):
    """合成堆积脉冲样本 v2 - 带增强与可见性约束
    
    Args:
        X_single: (N_single, L) 单脉冲波形池
        y_single: (N_single,) 标签池
        n_samples: 要生成的样本数
        lambda_hz_array: 各 λ 值数组
        k_values: [2, 3] 等
        comp_labels_array: (n_samples, 3) 指定每条样本的组合标签或 None
        rng: np.random.Generator
        baseline_b: 基线计算点数
        zero_prefix_b: 是否将前 B 点置零
        fs_hz: 采样率 (Hz)
        amp_min, amp_max: 幅度缩放范围
        noise_enable: 是否启用噪声
        noise_std_rel: 噪声标准差相对比例
        drift_enable: 是否启用基线漂移
        drift_slope_max_rel: 漂移斜率最大值相对比例
        drift_lf_std_rel: 低频漂移标准差相对比例
        drift_lf_window: 低频漂移移动平均窗口
        min_visible_points: 最小可见点数
        min_visible_energy_ratio: 最小可见能量比
        max_shift_resample: 最大重采样次数
        fail_policy: 'flag' 或 'drop'
    
    Returns:
        dict: {
            'X': (n_samples, L) 合成波形,
            'y_K': (n_samples,) 2 或 3,
            'comp_labels': (n_samples, 3) 组合标签,
            'shifts_samples': (n_samples, 2) shift2, shift3,
            'lambda_hz': (n_samples,) 每条样本的 λ,
            'targets': (n_samples, 3, L),
            'targets_mask': (n_samples, 3),
            'truncated_flags': (n_samples, 3) 截断标记,
            'visibility_metrics': (n_samples, 3) 可见能量比,
        }
    """
    L = X_single.shape[1]
    dt_us_per_sample = 1e6 / fs_hz
    
    # 预分配数组
    X_out = []
    y_K_out = []
    comp_labels_out = []
    shifts_out = []
    lambda_hz_out = []
    targets_out = []
    targets_mask_out = []
    truncated_flags_out = []
    visibility_metrics_out = []
    
    generated = 0
    attempts = 0
    max_total_attempts = n_samples * 10  # 防止无限循环
    
    while generated < n_samples and attempts < max_total_attempts:
        attempts += 1
        
        # 1. 决定 K 值（2 或 3）
        if comp_labels_array is not None and generated < len(comp_labels_array):
            comp = comp_labels_array[generated].copy()
            K = np.sum(comp >= 0)
        else:
            K = rng.choice(k_values)
            comp = np.array([rng.choice([0, 1]) for _ in range(K)], dtype=np.int32)
            comp = np.concatenate([comp, np.full(3 - K, -1, dtype=np.int32)])
        
        # 2. 选择 λ
        lambda_hz = rng.choice(lambda_hz_array)
        lambda_per_sec = lambda_hz
        
        # 3. 抽样脉冲（允许有放回）
        pulses_raw = []
        for k in range(K):
            idx = rng.choice(len(y_single))
            pulses_raw.append(X_single[idx].copy())
        
        # 4. 对每个分量应用幅度增强
        pulses = []
        for pulse_raw in pulses_raw:
            pulse_aug = augment_component(pulse_raw, rng, amp_min, amp_max)
            pulses.append(pulse_aug)
        
        # 5. 首脉冲在 0（不去基线，不增强）
        x_out = pulses[0].copy()
        targets = [pulses[0].copy()]
        shifts = [-1, -1]
        truncated_flags = [0, 0, 0]
        visibility_metrics = [1.0, 0.0, 0.0]  # 首脉冲完全可见
        
        # 6. 第二脉冲（带可见性约束）
        if K >= 2:
            pulse2 = pulses[1].copy()
            pulse2 = remove_baseline(pulse2, baseline_b)
            if zero_prefix_b:
                pulse2[:baseline_b] = 0
            
            # 采样 shift2 并检查可见性
            shift2, is_truncated2, energy_ratio2 = resample_shift_until_visible(
                lambda_per_sec, rng, fs_hz, L, pulse2,
                base_shift=0,
                min_visible_points=min_visible_points,
                min_visible_energy_ratio=min_visible_energy_ratio,
                max_resample=max_shift_resample
            )
            
            # 根据 fail_policy 处理截断情况
            if is_truncated2 and fail_policy == 'drop':
                continue  # 重新生成整条样本
            
            shifts[0] = min(shift2, L - 1)
            truncated_flags[1] = int(is_truncated2)
            visibility_metrics[1] = energy_ratio2
            
            # 叠加第二脉冲
            if shift2 < L:
                if shift2 + len(pulse2) <= L:
                    x_out[shift2:shift2 + len(pulse2)] += pulse2
                else:
                    remain = L - shift2
                    x_out[shift2:] += pulse2[:remain]
            
            # 记录 target
            target2 = np.zeros(L, dtype=np.float64)
            if shift2 < L:
                if shift2 + len(pulse2) <= L:
                    target2[shift2:shift2 + len(pulse2)] = pulse2
                else:
                    target2[shift2:] = pulse2[:L - shift2]
            targets.append(target2)
        
        # 7. 第三脉冲（带可见性约束）
        if K == 3:
            pulse3 = pulses[2].copy()
            pulse3 = remove_baseline(pulse3, baseline_b)
            if zero_prefix_b:
                pulse3[:baseline_b] = 0
            
            # 采样 shift3 并检查可见性
            shift3, is_truncated3, energy_ratio3 = resample_shift_until_visible(
                lambda_per_sec, rng, fs_hz, L, pulse3,
                base_shift=shifts[0],
                min_visible_points=min_visible_points,
                min_visible_energy_ratio=min_visible_energy_ratio,
                max_resample=max_shift_resample
            )
            
            # 根据 fail_policy 处理截断情况
            if is_truncated3 and fail_policy == 'drop':
                continue  # 重新生成整条样本
            
            shifts[1] = min(shift3, L - 1)
            truncated_flags[2] = int(is_truncated3)
            visibility_metrics[2] = energy_ratio3
            
            # 叠加第三脉冲
            if shift3 < L:
                if shift3 + len(pulse3) <= L:
                    x_out[shift3:shift3 + len(pulse3)] += pulse3
                else:
                    remain = L - shift3
                    x_out[shift3:] += pulse3[:remain]
            
            # 记录 target
            target3 = np.zeros(L, dtype=np.float64)
            if shift3 < L:
                if shift3 + len(pulse3) <= L:
                    target3[shift3:shift3 + len(pulse3)] = pulse3
                else:
                    target3[shift3:] = pulse3[:L - shift3]
            targets.append(target3)
        
        # 8. 对合成波形应用噪声与漂移
        x_out = augment_composite(
            x_out, rng, fs_hz,
            noise_enable=noise_enable,
            noise_std_rel=noise_std_rel,
            drift_enable=drift_enable,
            drift_slope_max_rel=drift_slope_max_rel,
            drift_lf_std_rel=drift_lf_std_rel,
            drift_lf_window=drift_lf_window
        )
        
        # 9. 补齐 targets 到 3 个
        while len(targets) < 3:
            targets.append(np.zeros(L, dtype=np.float64))
        
        targets_array = np.stack(targets, axis=0)  # (3, L)
        targets_mask = np.zeros(3, dtype=np.int32)
        targets_mask[:K] = 1
        
        # 10. 保存结果
        X_out.append(x_out)
        y_K_out.append(K)
        comp_labels_out.append(comp)
        shifts_out.append(shifts)
        lambda_hz_out.append(lambda_hz)
        targets_out.append(targets_array)
        targets_mask_out.append(targets_mask)
        truncated_flags_out.append(truncated_flags)
        visibility_metrics_out.append(visibility_metrics)
        
        generated += 1
    
    if generated < n_samples:
        print(f"WARNING: Only generated {generated}/{n_samples} samples after {attempts} attempts")
    
    # 转换为 numpy 数组
    return {
        'X': np.array(X_out, dtype=np.float64),
        'y_K': np.array(y_K_out, dtype=np.int32),
        'comp_labels': np.array(comp_labels_out, dtype=np.int32),
        'shifts_samples': np.array(shifts_out, dtype=np.int32),
        'lambda_hz': np.array(lambda_hz_out, dtype=np.float64),
        'targets': np.array(targets_out, dtype=np.float64),
        'targets_mask': np.array(targets_mask_out, dtype=np.int32),
        'truncated_flags': np.array(truncated_flags_out, dtype=np.int8),
        'visibility_metrics': np.array(visibility_metrics_out, dtype=np.float32),
    }
