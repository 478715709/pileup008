import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ====================== 1. 数据合成模块 ======================
class PileupDataset(Dataset):
    """
    模拟堆积脉冲数据集
    假设已有一个纯净单脉冲库，包含中子（n）和伽马（g）的波形。
    这里用随机生成的指数衰减脉冲代替真实数据。
    """
    def __init__(self, num_samples=10000, max_peaks=3, pulse_length=256):
        self.num_samples = num_samples
        self.max_peaks = max_peaks
        self.pulse_length = pulse_length
        # 预先存储纯净单脉冲的“模板”，实际应用中应从真实数据加载
        self.neutron_template = self._generate_pulse_template('n')
        self.gamma_template = self._generate_pulse_template('g')
        # 模拟不同时间间隔的分布
        self.dt_range = (10, 150)  # 脉冲间间隔范围 (ns) 假设采样率1ns/点

    def _generate_pulse_template(self, particle_type):
        """
        生成模拟的单脉冲模板（双指数衰减）
        中子慢成分多，伽马快成分多
        """
        t = np.arange(self.pulse_length)
        if particle_type == 'n':
            # 中子：慢衰减为主
            tau1, tau2 = 80, 300   # 快、慢时间常数
            amp1, amp2 = 0.3, 0.7
        else:
            # 伽马：快衰减为主
            tau1, tau2 = 30, 150
            amp1, amp2 = 0.8, 0.2
        pulse = amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2)
        pulse = pulse / pulse.max()  # 归一化
        return torch.tensor(pulse, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机决定本次事件的脉冲数 k (1,2,3)
        k = np.random.choice([1,2,3], p=[0.4,0.35,0.25])
        # 随机决定每个脉冲的类型 (0=gamma, 1=neutron)
        types = np.random.choice([0,1], size=k)
        # 随机生成每个脉冲的幅度（能量），可服从均匀分布
        amps = np.random.uniform(0.5, 1.5, size=k)
        # 随机生成脉冲起始时间，确保不超出总长度且不重叠太严重
        if k == 1:
            starts = [np.random.randint(20, 80)]
        else:
            # 保证脉冲间间隔在设定范围内，且最后一个脉冲不超出窗口
            starts = [np.random.randint(20, 50)]
            for i in range(1, k):
                last = starts[-1]
                dt = np.random.randint(self.dt_range[0], self.dt_range[1])
                new_start = last + dt
                # 确保新脉冲仍在窗口内，且留出尾部
                if new_start + 120 < self.pulse_length:
                    starts.append(new_start)
                else:
                    # 若超出，则用前一脉冲的稍后位置重新尝试（简单处理：若超出则设为最大允许）
                    new_start = self.pulse_length - 120 - np.random.randint(10,30)
                    starts.append(new_start)
        starts = np.array(starts, dtype=np.float32)

        # 合成波形
        waveform = torch.zeros(self.pulse_length)
        for i in range(k):
            pulse = self.neutron_template if types[i]==1 else self.gamma_template
            # 平移并叠加，幅度缩放
            pos = int(starts[i])
            length = min(self.pulse_length - pos, self.pulse_length)
            waveform[pos:pos+length] += amps[i] * pulse[:length]
        # 添加噪声
        noise = torch.randn_like(waveform) * 0.05
        waveform += noise
        # 归一化
        waveform = (waveform - waveform.mean()) / waveform.std()

        # 构建标签
        # label_k: 脉冲数 (1,2,3) -> 转换为0,1,2
        label_k = torch.tensor(k-1, dtype=torch.long)
        # label_types: 长度固定为max_peaks，填充-100忽略
        label_types = torch.full((self.max_peaks,), -100, dtype=torch.long)
        label_types[:k] = torch.tensor(types, dtype=torch.long)
        # label_offsets: 每个脉冲的峰值位置（相对于波形起始）
        label_offsets = torch.full((self.max_peaks,), -1, dtype=torch.float32)
        label_offsets[:k] = torch.tensor(starts, dtype=torch.float32) + 30  # 假设峰值在起始后30ns

        return waveform, label_k, label_types, label_offsets

# ====================== 2. 模型定义 ======================
class TransformerPileupModel(nn.Module):
    """
    基于Transformer的多任务模型
    - 输入: 波形 (batch, seq_len)
    - 输出:
        - count_logits: (batch, 3) 堆积数分类
        - type_logits: (batch, max_peaks, 2) 每个查询的类型概率
        - offset_pred: (batch, max_peaks) 峰值位置回归
    """
    def __init__(self, seq_len=256, d_model=128, nhead=8, num_layers=4, max_peaks=3):
        super().__init__()
        self.seq_len = seq_len
        self.max_peaks = max_peaks
        self.d_model = d_model

        # 输入投影：将1维波形映射到d_model维
        self.input_proj = nn.Linear(1, d_model)

        # 可学习的 positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 可学习的 query (用于提取每个脉冲的信息)
        self.query_embed = nn.Embedding(max_peaks, d_model)

        # 输出头
        # 计数头：利用全局特征（CLS token或平均池化）
        self.count_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 1,2,3类
        )

        # 分类头：每个query预测一个脉冲的类型 (2类)
        self.type_head = nn.Linear(d_model, 2)

        # 偏移回归头：每个query预测峰值位置
        self.offset_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        batch_size = x.shape[0]

        # 增加通道维 -> (batch, seq_len, 1) -> 投影
        x = x.unsqueeze(-1)
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # 加位置编码
        x = x + self.pos_encoder

        # Transformer Encoder
        memory = self.encoder(x)  # (batch, seq_len, d_model)

        # 计数头：使用全局平均池化
        global_feat = memory.mean(dim=1)  # (batch, d_model)
        count_logits = self.count_head(global_feat)  # (batch, 3)

        # 准备 queries
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, max_peaks, d_model)

        # 通过简单的注意力将 queries 与 memory 交互（这里简化：直接相加后送入分类头）
        # 实际更复杂的交互可考虑 cross-attention，为简化代码直接相加
        feat = queries + global_feat.unsqueeze(1)  # (batch, max_peaks, d_model)

        type_logits = self.type_head(feat)  # (batch, max_peaks, 2)
        offset_pred = self.offset_head(feat).squeeze(-1)  # (batch, max_peaks)

        return count_logits, type_logits, offset_pred

# ====================== 3. 损失函数 ======================
def pileup_loss(count_logits, type_logits, offset_pred,
                label_k, label_types, label_offsets, max_peaks=3):
    """
    count_logits: (batch, 3)
    type_logits: (batch, max_peaks, 2)
    offset_pred: (batch, max_peaks)
    label_k: (batch,) 实际脉冲数-1
    label_types: (batch, max_peaks) 包含-100填充
    label_offsets: (batch, max_peaks) 包含-1填充
    """
    # 1. 计数损失
    loss_count = F.cross_entropy(count_logits, label_k)

    # 2. 类型损失（仅计算有效脉冲）
    mask = label_types != -100
    loss_type = F.cross_entropy(type_logits[mask], label_types[mask], ignore_index=-100)

    # 3. 偏移损失（仅计算有效脉冲）
    mask_off = label_offsets != -1
    loss_offset = F.mse_loss(offset_pred[mask_off], label_offsets[mask_off])

    # 总损失（可调节权重）
    total_loss = loss_count + loss_type + loss_offset
    return total_loss, loss_count, loss_type, loss_offset

# ====================== 4. 训练与评估 ======================
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        waveforms, label_k, label_types, label_offsets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        count_logits, type_logits, offset_pred = model(waveforms)
        loss, _, _, _ = pileup_loss(count_logits, type_logits, offset_pred,
                                    label_k, label_types, label_offsets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_correct_count = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            waveforms, label_k, label_types, label_offsets = [x.to(device) for x in batch]
            count_logits, type_logits, offset_pred = model(waveforms)
            # 计数准确率
            pred_count = count_logits.argmax(dim=1)
            correct = (pred_count == label_k).sum().item()
            total_correct_count += correct
            total_samples += label_k.size(0)

            # 类型准确率（仅当计数正确时计算有效脉冲的类型）
            # 这里简单统计所有有效脉冲的类型准确率（不考虑顺序匹配）
            mask = label_types != -100
            if mask.sum() > 0:
                pred_types = type_logits[mask].argmax(dim=1)
                correct_types = (pred_types == label_types[mask]).sum().item()
                # 这里可以累积类型准确计数，但需要按batch加权
                # 为简化，返回计数准确率作为代表
    return total_correct_count / total_samples

# ====================== 5. 主程序 ======================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    batch_size = 32
    epochs = 20
    seq_len = 256
    max_peaks = 3

    # 数据集
    dataset = PileupDataset(num_samples=10000, max_peaks=max_peaks, pulse_length=seq_len)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 模型
    model = TransformerPileupModel(seq_len=seq_len, d_model=128, nhead=8, num_layers=4, max_peaks=max_peaks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Count Accuracy: {acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "transformer_pileup.pth")