"""数据增强模块 - 用于生成更真实的堆积脉冲波形"""
import numpy as np


def augment_component(x, rng, amp_min=0.85, amp_max=1.15):
    """对单个分量波形进行幅度缩放增强
    
    Args:
        x: (L,) 单个分量波形
        rng: numpy.random.Generator
        amp_min: 幅度缩放下限
        amp_max: 幅度缩放上限
    
    Returns:
        增强后的波形 (L,)
    """
    scale = rng.uniform(amp_min, amp_max)
    return x * scale


def augment_composite(x, rng, fs_hz=500e6, 
                     noise_enable=True, noise_std_rel=0.003,
                     drift_enable=True, drift_slope_max_rel=1e-4, 
                     drift_lf_std_rel=0.001, drift_lf_window=800):
    """对合成后的波形添加噪声与基线漂移
    
    Args:
        x: (L,) 合成波形
        rng: numpy.random.Generator
        fs_hz: 采样率 (Hz)
        noise_enable: 是否启用噪声
        noise_std_rel: 噪声标准差相对于波形最大幅值的比例
        drift_enable: 是否启用基线漂移
        drift_slope_max_rel: 线性漂移斜率最大值（相对每微秒）
        drift_lf_std_rel: 低频漂移标准差相对比例
        drift_lf_window: 低频漂移的移动平均窗口（样本点数）
    
    Returns:
        增强后的波形 (L,)
    """
    x_aug = x.copy()
    L = len(x)
    x_max = np.max(np.abs(x))
    
    # 添加白噪声
    if noise_enable and x_max > 0:
        noise_std = noise_std_rel * x_max
        noise = rng.normal(0, noise_std, size=L)
        x_aug += noise
    
    # 添加基线漂移
    if drift_enable and x_max > 0:
        # 线性漂移
        dt_us = 1e6 / fs_hz  # 每点时间（微秒）
        time_us = np.arange(L) * dt_us
        time_center = time_us[L // 2]
        slope_max = drift_slope_max_rel * x_max / (time_us[-1] - time_us[0] if L > 1 else 1.0)
        slope = rng.uniform(-slope_max, slope_max)
        drift_linear = slope * (time_us - time_center)
        
        # 低频随机漂移（移动平均白噪声）
        if drift_lf_window > 0 and L > drift_lf_window:
            lf_noise_std = drift_lf_std_rel * x_max
            lf_noise_raw = rng.normal(0, lf_noise_std, size=L)
            # 简单移动平均作为低通滤波
            drift_lf = np.convolve(lf_noise_raw, 
                                   np.ones(drift_lf_window) / drift_lf_window, 
                                   mode='same')
        else:
            drift_lf = 0
        
        x_aug += drift_linear + drift_lf
    
    return x_aug


def compute_visible_energy_ratio(pulse, shift, L):
    """计算脉冲在窗口内可见部分的能量比
    
    Args:
        pulse: (L_pulse,) 脉冲波形
        shift: int 起始位置
        L: int 窗口总长度
    
    Returns:
        float 可见能量占总能量的比例 [0, 1]
    """
    total_energy = np.sum(pulse ** 2)
    if total_energy == 0:
        return 1.0  # 零能量脉冲视为完全可见
    
    if shift >= L:
        return 0.0  # 完全在窗口外
    
    # 计算可见部分
    visible_len = min(len(pulse), L - shift)
    if visible_len <= 0:
        return 0.0
    
    visible_pulse = pulse[:visible_len]
    visible_energy = np.sum(visible_pulse ** 2)
    
    return visible_energy / total_energy


def check_visibility(pulse, shift, L, min_visible_points=600, min_visible_energy_ratio=0.15):
    """检查脉冲是否满足可见性约束
    
    Args:
        pulse: (L_pulse,) 脉冲波形
        shift: int 起始位置
        L: int 窗口总长度
        min_visible_points: 最小可见点数
        min_visible_energy_ratio: 最小可见能量比
    
    Returns:
        (is_visible: bool, energy_ratio: float)
    """
    if shift >= L:
        return False, 0.0
    
    # 点数约束
    visible_points = min(len(pulse), L - shift)
    if visible_points < min_visible_points:
        return False, 0.0
    
    # 能量约束
    energy_ratio = compute_visible_energy_ratio(pulse, shift, L)
    if energy_ratio < min_visible_energy_ratio:
        return False, energy_ratio
    
    return True, energy_ratio


def resample_shift_until_visible(lambda_per_sec, rng, fs_hz, L, pulse, 
                                 base_shift=0,
                                 min_visible_points=600, 
                                 min_visible_energy_ratio=0.15,
                                 max_resample=50):
    """重采样 shift 直到满足可见性约束
    
    Args:
        lambda_per_sec: float 到达率 (Hz)
        rng: numpy.random.Generator
        fs_hz: float 采样率
        L: int 窗口长度
        pulse: (L_pulse,) 脉冲波形
        base_shift: int 基准位置（对于 shift3，这是 shift2）
        min_visible_points: 最小可见点数
        min_visible_energy_ratio: 最小可见能量比
        max_resample: 最大重采样次数
    
    Returns:
        (shift: int, is_truncated: bool, energy_ratio: float)
    """
    dt_us_per_sample = 1e6 / fs_hz
    
    for attempt in range(max_resample):
        # 采样时间间隔
        if lambda_per_sec > 0:
            dt_us = rng.exponential(1.0 / lambda_per_sec) * 1e6
            shift = base_shift + int(np.round(dt_us / dt_us_per_sample))
        else:
            shift = L  # 无穷远
        
        # 检查可见性
        is_visible, energy_ratio = check_visibility(
            pulse, shift, L, min_visible_points, min_visible_energy_ratio
        )
        
        if is_visible:
            return shift, False, energy_ratio
    
    # 达到最大重采样次数，返回最后一次的结果并标记为截断
    return shift, True, energy_ratio
