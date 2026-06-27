"""
Evaluates a trained ViT checkpoint on a test set.
Computes accuracy, top-k, confusion matrix, per-class metrics, and logs every prediction.

How to run:
    python scripts/eval_vit.py
    python scripts/eval_vit.py --json data/test.json --ckpt outputs/vit_best.pth
    python scripts/eval_vit.py --no_tta

For manual test set:
    python scripts/eval_vit.py --json data/manual_test/manual_test.json --ckpt outputs/vit_best.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_vit import JsonImageDataset
from models.vit_baseline import build_vit


def get_device():		# gpu if available, otherwise cpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PickCrop:		# picks one crop from FiveCrop output -- class instead of lambda so it works on windows multiprocessing
    def __init__(self, index):
        self.index = index

    def __call__(self, crops):
        return crops[self.index]


def build_tta_transforms():		# 4 views of the same face: original, flipped, top-left crop, bottom-right crop
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))

    original = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm,
    ])

    flipped = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        norm,
    ])

    crop_left = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.FiveCrop(224),
        PickCrop(0),
        transforms.ToTensor(),
        norm,
    ])

    crop_right = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.FiveCrop(224),
        PickCrop(3),
        transforms.ToTensor(),
        norm,
    ])

    return [original, flipped, crop_left, crop_right]


def build_standard_transform():		# basic resize + normalize, no augmentation
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def topk_accuracy(y_true, y_probs, k):		# checks if correct label is in the model's top k predictions
    correct = 0
    for true_label, probs in zip(y_true, y_probs):
        topk_labels = sorted(range(len(probs)), key=lambda i: -probs[i])[:k]
        if true_label in topk_labels:
            correct += 1
    return correct / max(1, len(y_true))


def confidence_stats(y_true, y_pred, y_probs):		# compares average confidence when the model is right vs wrong
    correct_confs = []
    wrong_confs = []

    for true, pred, probs in zip(y_true, y_pred, y_probs):
        conf = max(probs)
        if true == pred:
            correct_confs.append(conf)
        else:
            wrong_confs.append(conf)

    avg_correct = sum(correct_confs) / max(1, len(correct_confs))
    avg_wrong = sum(wrong_confs) / max(1, len(wrong_confs))

    return {
        "avg_confidence_correct": round(avg_correct, 4),
        "avg_confidence_wrong": round(avg_wrong, 4),
        "num_correct": len(correct_confs),
        "num_wrong": len(wrong_confs),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="data/test.json",
                        help="Path to evaluation JSON file")
    parser.add_argument("--ckpt", type=str, default="outputs/vit_best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    test_json = args.json
    ckpt_path = Path(args.ckpt)
    use_tta = not args.no_tta

    if not Path(test_json).exists():
        raise FileNotFoundError(f"JSON file not found: {test_json}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # load checkpoint and rebuild model
    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = ckpt["idx_to_class"]
    epoch = ckpt.get("epoch", "unknown")

    print("Loaded checkpoint:", str(ckpt_path))
    print("Checkpoint epoch:", epoch)
    print("Classes:", class_to_idx)
    print("TTA:", "on (4 views)" if use_tta else "off")

    model = build_vit(
        num_classes=len(class_to_idx),
        pretrained=False,
        model_name="vit_base_patch16_224",
        drop_rate=0.0,
        drop_path_rate=0.0,
        freeze_backbone=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    amp_enabled = (device.type == "cuda")

    with open(test_json, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    if use_tta:
        # run each of the 4 views, stack up probabilities, average at the end
        tta_transforms = build_tta_transforms()
        num_views = len(tta_transforms)
        all_probs = None
        y_true = None

        for view_idx, tf in enumerate(tta_transforms):
            ds = JsonImageDataset(test_json, transform=tf, class_to_idx=class_to_idx)

            batch_size = 64 if device.type == "cuda" else 32
            num_workers = 4 if device.type == "cuda" else 0

            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )

            view_probs = []
            view_true = []

            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, dtype=torch.long, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(x)

                probs = F.softmax(logits, dim=1)
                view_probs.append(probs.cpu())
                view_true.extend(y.cpu().tolist())

            view_probs = torch.cat(view_probs, dim=0)

            if all_probs is None:
                all_probs = view_probs
                y_true = view_true
            else:
                all_probs += view_probs

            print(f"  TTA view {view_idx + 1}/{num_views} done")

        all_probs = all_probs / num_views
        y_pred = all_probs.argmax(dim=1).tolist()
        y_probs = all_probs.tolist()

    else:
        # single pass, no tta
        tf = build_standard_transform()
        ds = JsonImageDataset(test_json, transform=tf, class_to_idx=class_to_idx)

        batch_size = 64 if device.type == "cuda" else 32
        num_workers = 4 if device.type == "cuda" else 0

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        y_true, y_pred, y_probs = [], [], []

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs.cpu().tolist())

    # compute metrics
    labels_sorted = list(range(len(class_to_idx)))
    target_names = [idx_to_class[i] for i in labels_sorted]

    acc = accuracy_score(y_true, y_pred)
    top2_acc = topk_accuracy(y_true, y_probs, k=2)
    top3_acc = topk_accuracy(y_true, y_probs, k=3)
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    conf_stats = confidence_stats(y_true, y_pred, y_probs)

    print(f"\nEval JSON: {test_json}")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Top-2 Accuracy: {top2_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"\nConfidence -- correct: {conf_stats['avg_confidence_correct']:.4f}"
          f"  |  wrong: {conf_stats['avg_confidence_wrong']:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):\n", cm)
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # save report json
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    is_manual = "manual_test" in test_json.replace("\\", "/")
    report_name = "manual_test_report.json" if is_manual else "test_report.json"
    pred_log_name = "manual_test_predictions.json" if is_manual else "test_predictions.json"

    report = {
        "eval_json": test_json,
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "tta": use_tta,
        "accuracy": acc,
        "top2_accuracy": top2_acc,
        "top3_accuracy": top3_acc,
        "confidence_stats": conf_stats,
        "classes": class_to_idx,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names, digits=4, output_dict=True
        ),
    }

    with open(out_dir / report_name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: outputs/{report_name}")

    # per-frame prediction log -- handy for finding what the model gets wrong
    predictions = []
    for i in range(len(y_true)):
        true_label = idx_to_class[y_true[i]]
        pred_label = idx_to_class[y_pred[i]]
        conf = max(y_probs[i])
        probs_map = {idx_to_class[j]: round(y_probs[i][j], 4)
                     for j in range(len(y_probs[i]))}

        predictions.append({
            "path": raw_samples[i]["path"],
            "true": true_label,
            "pred": pred_label,
            "correct": true_label == pred_label,
            "confidence": round(conf, 4),
            "probs": probs_map,
        })

    with open(out_dir / pred_log_name, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved: outputs/{pred_log_name}")

    # show the most confident wrong predictions -- worth investigating
    wrong = [p for p in predictions if not p["correct"]]
    wrong.sort(key=lambda p: -p["confidence"])

    if wrong:
        print(f"\nTop 10 most confident mistakes:")
        for p in wrong[:10]:
            print(f"  {p['confidence']*100:5.1f}%  pred={p['pred']:8s}"
                  f"  true={p['true']:8s}  {p['path']}")


if __name__ == "__main__":
    main()
