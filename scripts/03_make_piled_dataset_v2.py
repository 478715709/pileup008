#!/usr/bin/env python3
"""
【生成堆积脉冲数据集脚本 v2】
支持 Realistic 与 Balanced 两种模式，增加增强与可见性约束。
支持 main 与 hard 两种 profile。
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# 加入 src 目录以导入自定义模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pileup.io_mat import load_single_dataset
from src.pileup.synth_v2 import synthesize_pileup_samples_v2
from src.pileup.sampling import RealisticSampler, BalancedSampler
from src.pileup.utils import analyze_comp_labels
from src.pileup.utils_v2 import save_pileup_dataset_v2, print_dataset_stats_v2


# 预置 profile 配置
PROFILE_CONFIGS = {
    'main': {
        'amp_min': 0.85,
        'amp_max': 1.15,
        'noise_enable': True,
        'noise_std_rel': 0.003,
        'drift_enable': True,
        'drift_slope_max_rel': 1e-4,
        'drift_lf_std_rel': 0.001,
        'drift_lf_window': 800,
        'min_visible_points': 600,
        'min_visible_energy_ratio': 0.15,
        'max_shift_resample': 50,
        'fail_policy': 'flag',
    },
    'hard': {
        'amp_min': 0.85,
        'amp_max': 1.15,
        'noise_enable': True,
        'noise_std_rel': 0.003,
        'drift_enable': True,
        'drift_slope_max_rel': 1e-4,
        'drift_lf_std_rel': 0.001,
        'drift_lf_window': 800,
        'min_visible_points': 200,         # 宽松：从 600 到 200
        'min_visible_energy_ratio': 0.05,  # 宽松：从 0.15 到 0.05
        'max_shift_resample': 3,           # 紧缩：从 50 到 3（更多截断）
        'fail_policy': 'flag',
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成堆积脉冲数据集 v2（增强 + 可见性约束）- 支持 main/hard profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 只生成 main
  python scripts/03_make_piled_dataset_v2.py --profile main
  
  # 只生成 hard
  python scripts/03_make_piled_dataset_v2.py --profile hard --outdir results/piled_pulse_v2_hard
  
  # 一次同时生成两套 (main + hard)，hard 自动为 main 的 20%
  python scripts/03_make_piled_dataset_v2.py --emit-hard --n-pile 14370 --hard-frac 0.2 --mix-mode both
  
  # 显式指定 hard 样本数
  python scripts/03_make_piled_dataset_v2.py --emit-hard --n-pile 14370 --n-pile-hard 3000
  
  # 自定义增强参数（覆盖 profile）
  python scripts/03_make_piled_dataset_v2.py \\
    --profile main \\
    --n-pile 2000 \\
    --amp-min 0.80 --amp-max 1.20 \\
    --min-visible-points 800
        """
    )
    
    # Profile 与输出目录参数
    parser.add_argument('--profile', choices=['main', 'hard'], default='main',
                       help='预置参数 profile：main（宽松约束）或 hard（严格约束）')
    parser.add_argument('--emit-hard', action='store_true', default=False,
                       help='同时生成 main 和 hard 两套数据集')
    parser.add_argument('--outdir', type=str, default='results/piled_pulse_v2',
                       help='main profile 输出目录（默认 results/piled_pulse_v2）')
    parser.add_argument('--hard-outdir', type=str, default='results/piled_pulse_v2_hard',
                       help='hard profile 输出目录（默认 results/piled_pulse_v2_hard）')
    
    # 基本参数
    parser.add_argument('--train', type=str, default='single_split_train.mat',
                       help='训练集 .mat 路径')
    parser.add_argument('--test', type=str, default='single_split_test.mat',
                       help='测试集 .mat 路径')
    
    parser.add_argument('--lambda-mhz', type=float, nargs='+',
                       default=[0.1, 0.2, 0.4, 0.8, 1.5, 3.0],
                       help='λ 值数组 (MHz)')
    parser.add_argument('--pile-mult', type=int, nargs='+', default=[2, 3],
                       help='堆积重数 (2 和/或 3)')
    parser.add_argument('--ratio-3', type=float, default=0.5,
                       help='balanced 模式中 K=3 的占比')
    
    parser.add_argument('--n-pile', type=int, default=2000,
                       help='main 的堆积样本数（默认 2000）')
    parser.add_argument('--n-pile-hard', type=int, default=None,
                       help='hard 的堆积样本数（默认 None，自动根据 hard-frac 计算）')
    parser.add_argument('--hard-frac', type=float, default=0.2,
                       help='当 emit-hard 且未指定 n-pile-hard 时，hard 样本数 = round(n-pile * hard-frac)（默认 0.2）')
    
    parser.add_argument('--baseline-b', type=int, default=200,
                       help='基线计算点数')
    parser.add_argument('--zero-prefix-b', type=lambda x: x.lower() == 'true', 
                       default=True, help='是否将前 B 点置零')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--mix-mode', choices=['both', 'realistic', 'balanced'],
                       default='both', help='生成模式')
    
    # v2 增强参数
    parser.add_argument('--amp-min', type=float, default=0.85,
                       help='幅度缩放下限（默认 0.85）')
    parser.add_argument('--amp-max', type=float, default=1.15,
                       help='幅度缩放上限（默认 1.15）')
    
    parser.add_argument('--noise-enable', type=lambda x: x.lower() == 'true',
                       default=True, help='是否启用噪声（默认 True）')
    parser.add_argument('--noise-std-rel', type=float, default=0.003,
                       help='噪声标准差相对比例（默认 0.003）')
    
    parser.add_argument('--drift-enable', type=lambda x: x.lower() == 'true',
                       default=True, help='是否启用基线漂移（默认 True）')
    parser.add_argument('--drift-slope-max-rel', type=float, default=1e-4,
                       help='漂移斜率最大值相对比例（默认 1e-4）')
    parser.add_argument('--drift-lf-std-rel', type=float, default=0.001,
                       help='低频漂移标准差相对比例（默认 0.001）')
    parser.add_argument('--drift-lf-window', type=int, default=800,
                       help='低频漂移移动平均窗口点数（默认 800）')
    
    # v2 可见性约束参数
    parser.add_argument('--min-visible-points', type=int, default=600,
                       help='最小可见点数（默认 600）')
    parser.add_argument('--min-visible-energy-ratio', type=float, default=0.15,
                       help='最小可见能量比（默认 0.15）')
    parser.add_argument('--max-shift-resample', type=int, default=50,
                       help='最大重采样次数（默认 50）')
    parser.add_argument('--fail-policy', choices=['flag', 'drop'], default='flag',
                       help='截断处理策略：flag=标记 | drop=丢弃（默认 flag）')
    
    return parser.parse_args()


def generate_pileup_dataset_v2(
    X_single, y_single, split_name, n_samples, lambda_hz_array, k_values,
    outdir, mode='realistic', ratio_3=0.5, baseline_b=200, zero_prefix_b=True,
    seed=42, aug_cfg=None, profile_suffix=''
):
    """生成一份堆积脉冲数据集 v2 (realistic 或 balanced)
    
    Args:
        X_single: (N, L) 单脉冲波形
        y_single: (N,) 标签
        split_name: 'train' 或 'test'
        n_samples: 堆积样本数
        lambda_hz_array: λ 值数组 (Hz)
        k_values: [2, 3] 等
        outdir: 输出目录
        mode: 'realistic' 或 'balanced'
        ratio_3: K=3 占比
        baseline_b: 基线点数
        zero_prefix_b: 置零前缀
        seed: 随机种子
        aug_cfg: 增强配置字典
        profile_suffix: 文件后缀（如 '' 或 '_hard'）
    """
    rng = np.random.default_rng(seed)
    fs_hz = 500e6
    L = X_single.shape[1]
    
    # 选择采样器
    if mode == 'realistic':
        sampler = RealisticSampler(X_single, y_single, rng)
        comp_labels = sampler.generate_comp_labels(n_samples, k_values, ratio_3)
    else:  # balanced
        sampler = BalancedSampler(X_single, y_single, rng)
        comp_labels = sampler.generate_comp_labels(n_samples, k_values, ratio_3)
    
    # 合成 v2
    result = synthesize_pileup_samples_v2(
        X_single, y_single, n_samples, lambda_hz_array, k_values,
        comp_labels, rng, baseline_b, zero_prefix_b, fs_hz,
        # 增强参数
        amp_min=aug_cfg['amp_min'],
        amp_max=aug_cfg['amp_max'],
        noise_enable=aug_cfg['noise_enable'],
        noise_std_rel=aug_cfg['noise_std_rel'],
        drift_enable=aug_cfg['drift_enable'],
        drift_slope_max_rel=aug_cfg['drift_slope_max_rel'],
        drift_lf_std_rel=aug_cfg['drift_lf_std_rel'],
        drift_lf_window=aug_cfg['drift_lf_window'],
        # 可见性约束参数
        min_visible_points=aug_cfg['min_visible_points'],
        min_visible_energy_ratio=aug_cfg['min_visible_energy_ratio'],
        max_shift_resample=aug_cfg['max_shift_resample'],
        fail_policy=aug_cfg['fail_policy'],
    )
    
    # 整理输出
    X = result['X']
    y_K = result['y_K']
    comp_labels_out = result['comp_labels']
    shifts_samples = result['shifts_samples']
    lambda_hz_out = result['lambda_hz']
    targets = result['targets']
    targets_mask = result['targets_mask']
    truncated_flags = result['truncated_flags']
    visibility_metrics = result['visibility_metrics']
    
    # 保存
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{mode}_{split_name}_pileup_v2{profile_suffix}.npz"
    filepath = outdir / filename
    
    save_pileup_dataset_v2(
        filepath, X, y_K, comp_labels_out, shifts_samples, lambda_hz_out,
        targets, targets_mask, truncated_flags, visibility_metrics, aug_cfg,
        fs_hz, L, baseline_b, zero_prefix_b, seed
    )
    
    # 统计
    n_k2 = np.sum(y_K == 2)
    n_k3 = np.sum(y_K == 3)
    
    lambda_counts = {}
    for lam in lambda_hz_out:
        lambda_counts[lam] = lambda_counts.get(lam, 0) + 1
    
    comp_counts = analyze_comp_labels(comp_labels_out)
    
    truncated_stats = [
        np.sum(truncated_flags[:, 0] > 0),
        np.sum(truncated_flags[:, 1] > 0),
        np.sum(truncated_flags[:, 2] > 0),
    ]
    
    print_dataset_stats_v2(mode, split_name, len(X), n_k2, n_k3, 
                          lambda_counts, comp_counts, truncated_stats)
    
    return filepath


def main():
    args = parse_args()
    
    # 读取数据
    print(f"Reading train set: {args.train}")
    X_train, y_train = load_single_dataset(args.train)
    n_gamma_train = np.sum(y_train == 1)
    n_neutron_train = np.sum(y_train == 0)
    print(f"  Gamma={n_gamma_train:,}, Neutron={n_neutron_train:,}, Total={len(y_train):,}")
    
    print(f"Reading test set: {args.test}")
    X_test, y_test = load_single_dataset(args.test)
    n_gamma_test = np.sum(y_test == 1)
    n_neutron_test = np.sum(y_test == 0)
    print(f"  Gamma={n_gamma_test:,}, Neutron={n_neutron_test:,}, Total={len(y_test):,}")
    
    # 堆积样本数
    n_pile_train = args.n_pile
    n_pile_test = args.n_pile
    
    # 计算 hard 样本数
    if args.emit_hard:
        if args.n_pile_hard is not None:
            n_pile_train_hard = args.n_pile_hard
            n_pile_test_hard = args.n_pile_hard
            hard_source = f"explicitly set (--n-pile-hard={args.n_pile_hard})"
        else:
            n_pile_train_hard = round(args.n_pile * args.hard_frac)
            n_pile_test_hard = round(args.n_pile * args.hard_frac)
            hard_source = f"computed (--n-pile * --hard-frac = {args.n_pile} * {args.hard_frac} ≈ {n_pile_train_hard})"
        
        print(f"\n{'='*70}")
        print(f"Sample count configuration:")
        print(f"  Main  n_pile: {n_pile_train:,}")
        print(f"  Hard  n_pile: {n_pile_train_hard:,} ({hard_source})")
        print(f"{'='*70}\n")
    
    # 转换 λ 从 MHz 到 Hz
    lambda_hz_array = np.array(args.lambda_mhz) * 1e6
    k_values = np.array(args.pile_mult)
    
    # 决定要生成的 profile
    profiles_to_run = []
    if args.emit_hard:
        profiles_to_run = ['main', 'hard']
    else:
        profiles_to_run = [args.profile]
    
    # 生成模式
    modes_to_run = []
    if args.mix_mode in ['both', 'realistic']:
        modes_to_run.append('realistic')
    if args.mix_mode in ['both', 'balanced']:
        modes_to_run.append('balanced')
    
    # 对每个 profile 生成数据
    for profile in profiles_to_run:
        # 获取 profile 配置或使用命令行参数覆盖
        if profile in PROFILE_CONFIGS:
            aug_cfg = PROFILE_CONFIGS[profile].copy()
            # 允许命令行参数覆盖（如果显式指定了）
            aug_cfg['amp_min'] = args.amp_min
            aug_cfg['amp_max'] = args.amp_max
            aug_cfg['noise_enable'] = args.noise_enable
            aug_cfg['noise_std_rel'] = args.noise_std_rel
            aug_cfg['drift_enable'] = args.drift_enable
            aug_cfg['drift_slope_max_rel'] = args.drift_slope_max_rel
            aug_cfg['drift_lf_std_rel'] = args.drift_lf_std_rel
            aug_cfg['drift_lf_window'] = args.drift_lf_window
            # 不覆盖 min_visible_points/energy_ratio/max_resample（除非显式指定）
            if args.min_visible_points != 600:  # 检测非默认值
                aug_cfg['min_visible_points'] = args.min_visible_points
            if args.min_visible_energy_ratio != 0.15:
                aug_cfg['min_visible_energy_ratio'] = args.min_visible_energy_ratio
            if args.max_shift_resample != 50:
                aug_cfg['max_shift_resample'] = args.max_shift_resample
        else:
            aug_cfg = {
                'amp_min': args.amp_min,
                'amp_max': args.amp_max,
                'noise_enable': args.noise_enable,
                'noise_std_rel': args.noise_std_rel,
                'drift_enable': args.drift_enable,
                'drift_slope_max_rel': args.drift_slope_max_rel,
                'drift_lf_std_rel': args.drift_lf_std_rel,
                'drift_lf_window': args.drift_lf_window,
                'min_visible_points': args.min_visible_points,
                'min_visible_energy_ratio': args.min_visible_energy_ratio,
                'max_shift_resample': args.max_shift_resample,
                'fail_policy': args.fail_policy,
            }
        
        # 选择输出目录和样本数
        if profile == 'hard':
            outdir = args.hard_outdir
            profile_suffix = '_hard'
            current_n_pile_train = n_pile_train_hard if args.emit_hard else n_pile_train
            current_n_pile_test = n_pile_test_hard if args.emit_hard else n_pile_test
        else:
            outdir = args.outdir
            profile_suffix = ''
            current_n_pile_train = n_pile_train
            current_n_pile_test = n_pile_test
        
        print(f"\n{'='*70}")
        print(f"Profile: {profile.upper()}")
        print(f"Output directory: {outdir}")
        print(f"Samples: train={current_n_pile_train:,}, test={current_n_pile_test:,}")
        print(f"{'='*70}")
        print(f"增强配置:")
        print(f"  min_visible_points: {aug_cfg['min_visible_points']}")
        print(f"  min_visible_energy_ratio: {aug_cfg['min_visible_energy_ratio']}")
        print(f"  max_shift_resample: {aug_cfg['max_shift_resample']}")
        print(f"  fail_policy: {aug_cfg['fail_policy']}")
        print(f"{'='*70}\n")
        
        # 生成数据集
        for mode in modes_to_run:
            print(f"\n{'='*70}")
            print(f"Generating {mode.upper()} mode dataset (v2 - {profile})")
            print(f"{'='*70}\n")
            
            generate_pileup_dataset_v2(
                X_train, y_train, 'train', current_n_pile_train, lambda_hz_array, k_values,
                outdir, mode, args.ratio_3, args.baseline_b, args.zero_prefix_b,
                args.seed, aug_cfg, profile_suffix
            )
            
            generate_pileup_dataset_v2(
                X_test, y_test, 'test', current_n_pile_test, lambda_hz_array, k_values,
                outdir, mode, args.ratio_3, args.baseline_b, args.zero_prefix_b,
                args.seed + 1, aug_cfg, profile_suffix
            )
    
    # 生成完毕提示
    if args.emit_hard:
        print(f"\nOK: v2 datasets generated successfully")
        print(f"  main: {args.outdir}")
        print(f"  hard: {args.hard_outdir}")
    else:
        print(f"\nOK: v2 dataset generated successfully at {outdir}")



if __name__ == '__main__':
    main()
