#!/usr/bin/env python3
"""
Step 3/4/5 训练流程复现脚本：
- Step 3: 用 balanced_train 预训练
- Step 4: 用 realistic_train 微调
- Step 5: 在 realistic_test 上评估

任务定义（简单可复现 baseline）：
1) y_K 二分类：K=2 vs K=3
2) comp_labels 多标签分类：3 个分量位分别预测 gamma(1)/neutron(0)，忽略 -1
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Metrics:
    loss: float
    k_acc: float
    comp_acc: float


class PileupNpzDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.y_k = data["y_K"].astype(np.int64)
        self.comp = data["comp_labels"].astype(np.int64)

        if self.X.ndim != 2:
            raise ValueError(f"X shape must be (N, L), got {self.X.shape}")
        if self.comp.shape[1] != 3:
            raise ValueError(f"comp_labels second dim must be 3, got {self.comp.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        x = (x - x.mean()) / (x.std() + 1e-6)

        y_k = int(self.y_k[idx])
        if y_k not in (2, 3):
            raise ValueError(f"y_K must be 2 or 3, got {y_k}")
        y_k_bin = torch.tensor(y_k - 2, dtype=torch.long)

        comp = torch.from_numpy(self.comp[idx])
        comp_mask = (comp != -1)
        comp_target = comp.clamp(min=0).to(torch.float32)

        return x, y_k_bin, comp_target, comp_mask


class ConvBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.k_head = nn.Linear(64, 2)
        self.comp_head = nn.Linear(64, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        feat = self.backbone(x)
        feat = self.head(feat)
        return self.k_head(feat), self.comp_head(feat)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ConvEnhanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            ResidualConvBlock(32, kernel_size=7, dropout=0.05),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(64, kernel_size=5, dropout=0.10),
            ResidualConvBlock(64, kernel_size=5, dropout=0.10),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            ResidualConvBlock(128, kernel_size=3, dropout=0.10),
            ResidualConvBlock(128, kernel_size=3, dropout=0.10),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.k_head = nn.Linear(64, 2)
        self.comp_head = nn.Linear(64, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        pooled_avg = self.avg_pool(x)
        pooled_max = self.max_pool(x)
        feat = torch.cat([pooled_avg, pooled_max], dim=1)

        feat = self.head(feat)
        return self.k_head(feat), self.comp_head(feat)


def build_model(model_name: str) -> nn.Module:
    if model_name == "baseline":
        return ConvBaseline()
    if model_name == "enhanced":
        return ConvEnhanced()
    raise ValueError(f"Unknown model: {model_name}")


def build_loader(npz_path: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = PileupNpzDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def compute_losses(k_logits, comp_logits, y_k, comp_target, comp_mask, comp_weight: float):
    loss_k = nn.functional.cross_entropy(k_logits, y_k)

    valid_comp_logits = comp_logits[comp_mask]
    valid_comp_target = comp_target[comp_mask]
    if valid_comp_logits.numel() == 0:
        loss_comp = torch.tensor(0.0, device=k_logits.device)
    else:
        loss_comp = nn.functional.binary_cross_entropy_with_logits(valid_comp_logits, valid_comp_target)

    total = loss_k + comp_weight * loss_comp
    return total, loss_k, loss_comp


def run_epoch(model, loader, optimizer, device, comp_weight: float, train: bool, grad_clip: float = 1.0) -> Metrics:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    total_k_correct = 0

    total_comp_correct = 0
    total_comp_count = 0

    for x, y_k, comp_target, comp_mask in loader:
        x = x.to(device)
        y_k = y_k.to(device)
        comp_target = comp_target.to(device)
        comp_mask = comp_mask.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            k_logits, comp_logits = model(x)
            loss, _, _ = compute_losses(k_logits, comp_logits, y_k, comp_target, comp_mask, comp_weight)

            if train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        batch_size = x.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        pred_k = k_logits.argmax(dim=1)
        total_k_correct += (pred_k == y_k).sum().item()

        comp_prob = torch.sigmoid(comp_logits)
        pred_comp = (comp_prob >= 0.5)
        valid_comp = comp_mask
        total_comp_correct += (pred_comp[valid_comp] == comp_target[valid_comp].bool()).sum().item()
        total_comp_count += valid_comp.sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    k_acc = total_k_correct / max(total_samples, 1)
    comp_acc = total_comp_correct / max(total_comp_count, 1)
    return Metrics(loss=avg_loss, k_acc=k_acc, comp_acc=comp_acc)


def save_checkpoint(path: Path, model: nn.Module, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, path)


def load_checkpoint(path: str | Path, model: nn.Module, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt.get("config", {})


def train_stage(
    stage_name: str,
    train_npz: str,
    val_npz: str,
    out_ckpt: Path,
    init_ckpt: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    comp_weight: float,
    device,
    seed: int,
    model_name: str,
    weight_decay: float,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(model_name).to(device)
    if init_ckpt:
        _ = load_checkpoint(init_ckpt, model, device)

    train_loader = build_loader(train_npz, batch_size=batch_size, shuffle=True)
    val_loader = build_loader(val_npz, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_loss = float("inf")
    best_metrics = None

    print(f"\n[{stage_name}] train={train_npz}")
    print(f"[{stage_name}] val={val_npz}")

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, comp_weight, train=True)
        va = run_epoch(model, val_loader, optimizer, device, comp_weight, train=False)
        scheduler.step()

        print(
            f"[{stage_name}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={tr.loss:.4f} train_k_acc={tr.k_acc:.4f} train_comp_acc={tr.comp_acc:.4f} | "
            f"val_loss={va.loss:.4f} val_k_acc={va.k_acc:.4f} val_comp_acc={va.comp_acc:.4f}"
        )

        if va.loss < best_val_loss:
            best_val_loss = va.loss
            best_metrics = va
            save_checkpoint(
                out_ckpt,
                model,
                {
                    "stage": stage_name,
                    "model": model_name,
                    "train_npz": train_npz,
                    "val_npz": val_npz,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "comp_weight": comp_weight,
                    "weight_decay": weight_decay,
                    "seed": seed,
                },
            )

    if best_metrics is None:
        raise RuntimeError(f"No checkpoint saved for stage {stage_name}")

    print(
        f"[{stage_name}] best_val_loss={best_val_loss:.4f}, "
        f"best_val_k_acc={best_metrics.k_acc:.4f}, best_val_comp_acc={best_metrics.comp_acc:.4f}"
    )


def evaluate_stage(test_npz: str, ckpt_path: str, batch_size: int, comp_weight: float, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})
    model_name = ckpt_cfg.get("model", "baseline")
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model"])

    test_loader = build_loader(test_npz, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    m = run_epoch(model, test_loader, optimizer, device, comp_weight, train=False)

    result = {
        "test_npz": test_npz,
        "checkpoint": ckpt_path,
        "test_loss": m.loss,
        "test_k_acc": m.k_acc,
        "test_comp_acc": m.comp_acc,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Step 3/4/5 训练流程脚本")
    parser.add_argument("--mode", choices=["pretrain", "finetune", "eval", "all"], default="all")

    parser.add_argument("--balanced-train", type=str, default="results/piled_pulse/balanced_train_pileup.npz")
    parser.add_argument("--balanced-test", type=str, default="results/piled_pulse/balanced_test_pileup.npz")
    parser.add_argument("--realistic-train", type=str, default="results/piled_pulse/realistic_train_pileup.npz")
    parser.add_argument("--realistic-test", type=str, default="results/piled_pulse/realistic_test_pileup.npz")

    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--comp-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--model", type=str, choices=["baseline", "enhanced"], default="baseline")

    parser.add_argument("--outdir", type=str, default="results/models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    outdir = Path(args.outdir)
    pretrain_ckpt = outdir / "pretrain_balanced.pt"
    finetune_ckpt = outdir / "finetune_realistic.pt"

    print(f"Using device: {device}")

    if args.mode in ("pretrain", "all"):
        train_stage(
            stage_name="pretrain",
            train_npz=args.balanced_train,
            val_npz=args.balanced_test,
            out_ckpt=pretrain_ckpt,
            init_ckpt=None,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.pretrain_lr,
            comp_weight=args.comp_weight,
            device=device,
            seed=args.seed,
            model_name=args.model,
            weight_decay=args.weight_decay,
        )

    if args.mode in ("finetune", "all"):
        init_ckpt = str(pretrain_ckpt) if args.mode == "all" else str(pretrain_ckpt)
        if not Path(init_ckpt).exists():
            raise FileNotFoundError(
                f"Pretrain checkpoint not found: {init_ckpt}. "
                f"Run --mode pretrain first or use --mode all."
            )
        train_stage(
            stage_name="finetune",
            train_npz=args.realistic_train,
            val_npz=args.realistic_test,
            out_ckpt=finetune_ckpt,
            init_ckpt=init_ckpt,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.finetune_lr,
            comp_weight=args.comp_weight,
            device=device,
            seed=args.seed,
            model_name=args.model,
            weight_decay=args.weight_decay,
        )

    if args.mode in ("eval", "all"):
        ckpt = finetune_ckpt if finetune_ckpt.exists() else pretrain_ckpt
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint found under {outdir}. "
                f"Run pretrain/finetune first or use --mode all."
            )
        result = evaluate_stage(
            test_npz=args.realistic_test,
            ckpt_path=str(ckpt),
            batch_size=args.batch_size,
            comp_weight=args.comp_weight,
            device=device,
        )

        print("\n[eval] Final metrics:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        metrics_path = outdir / "eval_realistic_test.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
