"""
Main training script for the ViT emotion recognition model.
Reads train/val JSON splits, runs two-phase training (frozen head then full fine-tuning),
and saves the best checkpoint based on validation accuracy.

How to run:
    python scripts/train_vit.py
"""

import json
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_vit import JsonImageDataset
from models.vit_baseline import build_vit



class FocalLoss(nn.Module):		# replaces CrossEntropy -- downweights easy samples so the model focuses on hard ones
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# mixup / cutmix stuff

def do_mixup(x, y, alpha=0.4):		# blends two images together with a random ratio from Beta distribution
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def do_cutmix(x, y, alpha=1.0):		# cuts a random rectangle from one image and pastes it onto another
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    rw, rh = int(W * cut_ratio), int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - rw // 2, 0)
    y1 = max(cy - rh // 2, 0)
    x2 = min(cx + rw // 2, W)
    y2 = min(cy + rh // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # recalc lambda based on actual box size since clipping at edges changes it
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed_x, y, y[index], lam


def mixed_loss(criterion, logits, y_a, y_b, lam):		# weighted loss for mixed samples: lam * loss_A + (1-lam) * loss_B
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def set_seed(seed: int = 42):		# lock all random number generators for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_device():		# returns cuda if gpu is available, cpu otherwise
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def calc_acc(logits: torch.Tensor, y: torch.Tensor) -> float:		# quick batch accuracy check
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def get_class_weights(samples, class_to_idx):		# inverse frequency weights so rare classes get higher penalty in loss
    counts = {k: 0 for k in class_to_idx.keys()}
    for s in samples:
        counts[s["label"]] += 1
    total = sum(counts.values())
    weights = []
    for lab in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x]):
        c = counts[lab]
        weights.append(total / max(1, c))
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean()
    return w, counts


def make_sampler(samples, class_to_idx):		# oversamples minority classes so each epoch sees balanced emotion counts
    _, counts = get_class_weights(samples, class_to_idx)
    total = sum(counts.values())
    class_weight = {k: total / max(1, v) for k, v in counts.items()}
    sample_weights = [class_weight[s["label"]] for s in samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(samples),
        replacement=True,
    )


def dump_logs(path: Path, logs):		# writes training log to json after every epoch so we don't lose progress
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def train_epoch(model, loader, criterion, optimizer, scaler, device, amp_on, use_mix=True):		# one full pass over training data with optional mixup/cutmix
    model.train()
    tot_loss = 0.0
    tot_acc  = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        # coin flip between mixup and cutmix (only in phase 2)
        mixed = False
        if use_mix and random.random() < 0.5:
            if random.random() < 0.5:
                x, y_a, y_b, lam = do_mixup(x, y, alpha=0.4)
            else:
                x, y_a, y_b, lam = do_cutmix(x, y, alpha=1.0)
            mixed = True

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x)
            if mixed:
                loss = mixed_loss(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item()
        # accuracy isn't exact when mixing but close enough for tracking
        tot_acc += calc_acc(logits, y)

    n = max(1, len(loader))
    return tot_loss / n, tot_acc / n


@torch.no_grad()
def eval_model(model, loader, criterion, device, amp_on):		# runs validation without updating weights
    model.eval()
    tot_loss = 0.0
    tot_acc  = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x)
            loss   = criterion(logits, y)

        tot_loss += loss.item()
        tot_acc  += calc_acc(logits, y)

    n = max(1, len(loader))
    return tot_loss / n, tot_acc / n


def main():
    set_seed(42)
    device = get_device()
    print("Device:", device)

    train_json = "data/train.json"
    val_json   = "data/val.json"

    # training augmentations -- heavy on purpose to fight overfitting
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.05)
        ], p=0.7),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    # validation just gets resized and normalized, no augmentation
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_ds = JsonImageDataset(train_json, transform=train_tf, class_to_idx=None)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    val_ds = JsonImageDataset(val_json, transform=val_tf, class_to_idx=class_to_idx)

    print("Classes:", class_to_idx)
    print("Train samples:", len(train_ds), "| Val samples:", len(val_ds))

    batch_size  = 64
    num_workers = 4
    pin_memory  = (device.type == "cuda")

    sampler = make_sampler(train_ds.samples, class_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), prefetch_factor=2,
    )

    model = build_vit(
        num_classes=len(class_to_idx),
        pretrained=True,
        model_name="vit_base_patch16_224",
        drop_rate=0.2,
        drop_path_rate=0.2,
        freeze_backbone=False,
    ).to(device)

    w, counts = get_class_weights(train_ds.samples, class_to_idx)
    class_weights = w.to(device)
    print("Train label counts:", counts)

    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)

    # separate learning rates -- backbone gets a smaller lr since it already has good features from imagenet
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params     = [p for n, p in model.named_parameters() if "head" in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params,     "lr": 1e-4},
    ], weight_decay=0.05)

    print(f"Backbone params : {sum(p.numel() for p in backbone_params)/1e6:.1f}M  lr=1e-5")
    print(f"Head params     : {sum(p.numel() for p in head_params)/1e6:.1f}M  lr=1e-4")

    epochs        = 30
    freeze_epochs = 3   # keep backbone frozen for the first 3 epochs
    warmup_epochs = 3   # ramp up lr over 3 epochs (aligned with freeze period now)
    patience      = 8

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    amp_on = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    out_dir  = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "training_log.json"

    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    # phase 1: freeze backbone, only train the classification head
    print(f"\n[Phase 1] Head only for {freeze_epochs} epochs")
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    best_val_acc = 0.0
    best_epoch   = 0
    no_improve   = 0
    logs         = []

    for epoch in range(1, epochs + 1):

        # phase 2: unfreeze everything, start mixup/cutmix
        if epoch == freeze_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            print(f"\n[Phase 2] Full fine-tuning + mixup/cutmix\n")

        use_mix = (epoch > freeze_epochs)

        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_on, use_mix
        )
        val_loss, val_acc = eval_model(
            model, val_loader, criterion, device, amp_on
        )

        scheduler.step()

        dt      = time.time() - t0
        lr_head = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {lr_head:.2e} | time {dt:.1f}s"
        )

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
            "lr": lr_head,
        })
        dump_logs(log_path, logs)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        }, out_dir / "vit_last.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
            }, out_dir / "vit_best.pth")
            print(f"  -> New best saved: outputs/vit_best.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {patience} epochs with no improvement.")
                break

    print(f"\nBest val acc : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"Training log : {log_path}")


if __name__ == "__main__":
    main()
