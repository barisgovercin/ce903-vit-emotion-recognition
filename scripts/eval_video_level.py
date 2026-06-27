"""
Aggregates frame predictions into video-level results.
For each video, combines all frame outputs via majority voting and mean probability.

How to run:
    python scripts/eval_video_level.py

For manual test set:
    python scripts/eval_video_level.py --json data/manual_test/manual_test.json --ckpt outputs/vit_best.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_vit import JsonImageDataset
from models.vit_baseline import build_vit


def get_device():		# gpu if available, otherwise cpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PickCrop:		# picks one crop from FiveCrop -- class instead of lambda for windows compatibility
    def __init__(self, index):
        self.index = index

    def __call__(self, crops):
        return crops[self.index]


def build_tta_transforms():		# same 4 views as eval_vit: original, flipped, two corner crops
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))

    return [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm,
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            norm,
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.FiveCrop(224),
            PickCrop(0),
            transforms.ToTensor(),
            norm,
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.FiveCrop(224),
            PickCrop(3),
            transforms.ToTensor(),
            norm,
        ]),
    ]


def build_standard_transform():		# plain resize + normalize
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def get_frame_probs(model, json_path, tf, class_to_idx, device):		# runs all frames through model, returns softmax probs
    ds = JsonImageDataset(json_path, transform=tf, class_to_idx=class_to_idx)

    batch_size = 64 if device.type == "cuda" else 32
    num_workers = 4 if device.type == "cuda" else 0
    amp_enabled = (device.type == "cuda")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    all_probs = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(x)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str,
                        default="data/manual_test/manual_test.json",
                        help="Path to test JSON with video field")
    parser.add_argument("--ckpt", type=str,
                        default="outputs/vit_best.pth",
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
        raise FileNotFoundError(f"JSON not found: {test_json}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = ckpt["idx_to_class"]

    print("Loaded:", str(ckpt_path))
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

    with open(test_json, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    print(f"Total frames: {len(raw_samples)}")

    # get per-frame probabilities
    if use_tta:
        tta_transforms = build_tta_transforms()
        all_probs = None
        for view_idx, tf in enumerate(tta_transforms):
            view_probs = get_frame_probs(model, test_json, tf, class_to_idx, device)
            if all_probs is None:
                all_probs = view_probs
            else:
                all_probs += view_probs
            print(f"  TTA view {view_idx + 1}/{len(tta_transforms)} done")
        all_probs = all_probs / len(tta_transforms)
    else:
        tf = build_standard_transform()
        all_probs = get_frame_probs(model, test_json, tf, class_to_idx, device)

    # group frames by their source video
    video_frames = defaultdict(list)
    for i, sample in enumerate(raw_samples):
        video_name = sample["video"]
        true_label = sample["label"]
        probs = all_probs[i].tolist()
        pred_idx = all_probs[i].argmax().item()
        pred_label = idx_to_class[pred_idx]

        video_frames[video_name].append({
            "true": true_label,
            "pred": pred_label,
            "probs": probs,
        })

    # collapse each video's frames into one prediction
    results = []
    for video_name, frames in video_frames.items():
        true_label = frames[0]["true"]
        num_frames = len(frames)

        # majority voting: most frequent prediction wins
        pred_counts = Counter(f["pred"] for f in frames)
        majority_pred = pred_counts.most_common(1)[0][0]

        # mean probability: average softmax across frames
        mean_probs = [0.0] * len(class_to_idx)
        for f in frames:
            for j in range(len(mean_probs)):
                mean_probs[j] += f["probs"][j]
        mean_probs = [p / num_frames for p in mean_probs]
        mean_pred_idx = mean_probs.index(max(mean_probs))
        mean_pred = idx_to_class[mean_pred_idx]
        mean_conf = max(mean_probs)

        results.append({
            "video": video_name,
            "true": true_label,
            "frames": num_frames,
            "majority_pred": majority_pred,
            "majority_correct": majority_pred == true_label,
            "mean_pred": mean_pred,
            "mean_correct": mean_pred == true_label,
            "mean_confidence": round(mean_conf, 4),
            "frame_predictions": dict(pred_counts),
        })

    total_videos = len(results)
    majority_correct = sum(1 for r in results if r["majority_correct"])
    mean_correct = sum(1 for r in results if r["mean_correct"])

    # per-emotion breakdown
    emotion_stats = defaultdict(lambda: {"total": 0, "majority_correct": 0, "mean_correct": 0})
    for r in results:
        e = r["true"]
        emotion_stats[e]["total"] += 1
        if r["majority_correct"]:
            emotion_stats[e]["majority_correct"] += 1
        if r["mean_correct"]:
            emotion_stats[e]["mean_correct"] += 1

    print(f"\n{'='*60}")
    print(f"VIDEO-LEVEL RESULTS")
    print(f"{'='*60}")
    print(f"Total videos: {total_videos}")
    print(f"Majority voting:      {majority_correct}/{total_videos} correct ({majority_correct/total_videos*100:.1f}%)")
    print(f"Mean probability:     {mean_correct}/{total_videos} correct ({mean_correct/total_videos*100:.1f}%)")

    print(f"\nPer-emotion breakdown:")
    print(f"  {'Emotion':10s} {'Total':>6s} {'Majority':>10s} {'Mean Prob':>10s}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10}")
    for emotion in sorted(emotion_stats.keys()):
        s = emotion_stats[emotion]
        print(f"  {emotion:10s} {s['total']:>6d} "
              f"{s['majority_correct']:>4d}/{s['total']:<4d}  "
              f"{s['mean_correct']:>4d}/{s['total']:<4d}")

    # show wrong videos for debugging
    wrong_majority = [r for r in results if not r["majority_correct"]]
    if wrong_majority:
        print(f"\nWrong predictions (majority voting):")
        for r in sorted(wrong_majority, key=lambda x: x["true"]):
            print(f"  {r['video']:50s}  true={r['true']:8s}  pred={r['majority_pred']:8s}  "
                  f"frames: {r['frame_predictions']}")

    # save report
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    report = {
        "eval_json": test_json,
        "checkpoint": str(ckpt_path),
        "tta": use_tta,
        "total_videos": total_videos,
        "majority_correct": majority_correct,
        "majority_accuracy": round(majority_correct / total_videos, 4),
        "mean_correct": mean_correct,
        "mean_accuracy": round(mean_correct / total_videos, 4),
        "per_emotion": dict(emotion_stats),
        "video_results": results,
    }

    report_name = "video_level_report.json"
    with open(out_dir / report_name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: outputs/{report_name}")


if __name__ == "__main__":
    main()
