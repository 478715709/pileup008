"""Realistic 与 Balanced 抽样策略"""
import numpy as np


class RealisticSampler:
    """按原始比例抽样"""
    
    def __init__(self, X_single, y_single, rng):
        """
        Args:
            X_single: (N, L) 单脉冲池
            y_single: (N,) 标签：1=gamma, 0=neutron
            rng: np.random.Generator
        """
        self.X_single = X_single
        self.y_single = y_single
        self.rng = rng
        self.gamma_mask = (y_single == 1)
        self.neutron_mask = (y_single == 0)
    
    def generate_comp_labels(self, n_samples, k_values, ratio_3=0.5):
        """生成 comp_labels (不用，realistic 随机决定)"""
        return None


class BalancedSampler:
    """按组合均衡抽样"""
    
    # K=2 的 4 类
    K2_COMBOS = [
        np.array([1, 1, -1], dtype=np.int32),  # [γ, γ]
        np.array([1, 0, -1], dtype=np.int32),  # [γ, n]
        np.array([0, 1, -1], dtype=np.int32),  # [n, γ]
        np.array([0, 0, -1], dtype=np.int32),  # [n, n]
    ]
    
    # K=3 的 8 类
    K3_COMBOS = [
        np.array([1, 1, 1], dtype=np.int32),  # [γ, γ, γ]
        np.array([1, 1, 0], dtype=np.int32),  # [γ, γ, n]
        np.array([1, 0, 1], dtype=np.int32),  # [γ, n, γ]
        np.array([0, 1, 1], dtype=np.int32),  # [n, γ, γ]
        np.array([1, 0, 0], dtype=np.int32),  # [γ, n, n]
        np.array([0, 1, 0], dtype=np.int32),  # [n, γ, n]
        np.array([0, 0, 1], dtype=np.int32),  # [n, n, γ]
        np.array([0, 0, 0], dtype=np.int32),  # [n, n, n]
    ]
    
    def __init__(self, X_single, y_single, rng):
        """
        Args:
            X_single: (N, L)
            y_single: (N,)
            rng: np.random.Generator
        """
        self.X_single = X_single
        self.y_single = y_single
        self.rng = rng
        self.gamma_mask = (y_single == 1)
        self.neutron_mask = (y_single == 0)
    
    def generate_comp_labels(self, n_samples, k_values, ratio_3=0.5):
        """生成 comp_labels，按组合均衡分配
        
        Args:
            n_samples: 总样本数
            k_values: [2, 3] 等
            ratio_3: K=3 的占比（0.5 = 50%）
        
        Returns:
            (n_samples, 3) 的 comp_labels 数组
        """
        comp_labels = np.zeros((n_samples, 3), dtype=np.int32) - 1
        
        # 按 K=2 与 K=3 分配
        n_k3 = int(n_samples * ratio_3)
        n_k2 = n_samples - n_k3
        
        indices = self.rng.permutation(n_samples)
        k2_idx = indices[:n_k2]
        k3_idx = indices[n_k2:]
        
        # K=2：均分到 4 类
        n_per_class_k2 = n_k2 // len(self.K2_COMBOS)
        remainder_k2 = n_k2 % len(self.K2_COMBOS)
        
        pos = 0
        for class_id, combo in enumerate(self.K2_COMBOS):
            count = n_per_class_k2 + (1 if class_id < remainder_k2 else 0)
            comp_labels[k2_idx[pos:pos + count]] = combo
            pos += count
        
        # K=3：均分到 8 类
        n_per_class_k3 = n_k3 // len(self.K3_COMBOS)
        remainder_k3 = n_k3 % len(self.K3_COMBOS)
        
        pos = 0
        for class_id, combo in enumerate(self.K3_COMBOS):
            count = n_per_class_k3 + (1 if class_id < remainder_k3 else 0)
            comp_labels[k3_idx[pos:pos + count]] = combo
            pos += count
        
        return comp_labels
