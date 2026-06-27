"""
Generates all result plots from the saved JSON reports.
Reads training log, test report, manual test report, and video-level report.

How to run:
    python scripts/plot_results.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


def load_json(path):		# read a json file and return its contents
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, path):		# saves figure as png at 150 dpi
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _style(ax, xlabel=None, ylabel=None, title=None, grid_axis="both"):		# common axis styling
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(axis=grid_axis, linestyle="--", alpha=0.5)


def plot_confusion_matrix(cm, class_names, title, out_path):		# row-normalized heatmap with raw counts annotated
    cm = np.array(cm)
    n = len(class_names)

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums.astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = range(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")

    for (i, j), pct in np.ndenumerate(cm_norm):
        ax.text(j, i, f"{cm[i,j]}\n({pct*100:.0f}%)",
                ha="center", va="center", fontsize=9,
                color="white" if pct > 0.55 else "black")

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_class_metrics(report, class_names, title, out_path):		# grouped bar chart: precision, recall, f1 per class
    rows = []
    for cls in class_names:
        r = report.get(cls, {})
        rows.append((r.get("precision", 0), r.get("recall", 0), r.get("f1-score", 0)))

    precision, recall, f1 = zip(*rows)
    x = np.arange(len(class_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.bar(x - w, precision, w, label="Precision", color="#4C72B0")
    br = ax.bar(x,     recall,    w, label="Recall",    color="#DD8452")
    bf = ax.bar(x + w, f1,        w, label="F1-Score",  color="#55A868")

    for bars in (bp, br, bf):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    _style(ax, ylabel="Score", title=title, grid_axis="y")

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_training_curves(log, out_path):		# loss and accuracy curves side by side
    epochs = [e["epoch"] for e in log]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, [e["train_loss"] for e in log], marker="o", label="Train", color="#4C72B0")
    ax1.plot(epochs, [e["val_loss"]   for e in log], marker="o", label="Val",   color="#DD8452")
    _style(ax1, xlabel="Epoch", ylabel="Loss", title="Loss")

    ax2.plot(epochs, [e["train_acc"] for e in log], marker="o", label="Train", color="#4C72B0")
    ax2.plot(epochs, [e["val_acc"]   for e in log], marker="o", label="Val",   color="#DD8452")
    _style(ax2, xlabel="Epoch", ylabel="Accuracy", title="Accuracy")

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_video_level(report, out_path):		# per-emotion bar chart comparing majority vote vs mean probability
    per_emotion = report["per_emotion"]
    classes = sorted(per_emotion.keys())

    totals    = [per_emotion[c]["total"]            for c in classes]
    maj_corr  = [per_emotion[c]["majority_correct"] for c in classes]
    mean_corr = [per_emotion[c]["mean_correct"]     for c in classes]

    x = np.arange(len(classes))
    w = 0.30

    fig, ax = plt.subplots(figsize=(9, 5))
    bm = ax.bar(x - w/2, [m/t for m, t in zip(maj_corr,  totals)], w, label="Majority Voting",  color="#4C72B0")
    bp = ax.bar(x + w/2, [m/t for m, t in zip(mean_corr, totals)], w, label="Mean Probability", color="#DD8452")

    for bars, counts in [(bm, maj_corr), (bp, mean_corr)]:
        for bar, c, t in zip(bars, counts, totals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}\n({c}/{t})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    maj  = report["majority_accuracy"]
    mean = report["mean_accuracy"]
    _style(ax, ylabel="Accuracy",
           title=f"Video-Level Accuracy — Manual Test  (Majority: {maj:.4f}  |  Mean Prob: {mean:.4f})",
           grid_axis="y")

    fig.tight_layout()
    save_fig(fig, out_path)


def _plot_report(report, label, out_dir):		# generates confusion matrix + per-class metrics for a given report
    class_names = sorted(report["classes"].keys())
    acc  = report["accuracy"]
    slug = label.lower().replace(" ", "_")

    plot_confusion_matrix(
        report["confusion_matrix"], class_names,
        title=f"Confusion Matrix — {label}  (Acc: {acc:.4f})",
        out_path=out_dir / f"{slug}_confusion_matrix.png",
    )
    plot_class_metrics(
        report["classification_report"], class_names,
        title=f"Per-Class Metrics — {label}  (Acc: {acc:.4f})",
        out_path=out_dir / f"{slug}_class_metrics.png",
    )


def main():
    out_dir = Path("outputs/plots")

    log_path = Path("outputs/training_log.json")
    if log_path.exists():
        plot_training_curves(load_json(log_path), out_dir / "training_curves.png")
    else:
        print(f"[WARN] Not found: {log_path}")

    for path, label in [
        (Path("outputs/test_report.json"),        "Test Set"),
        (Path("outputs/manual_test_report.json"), "Manual Test"),
    ]:
        if path.exists():
            _plot_report(load_json(path), label, out_dir)
        else:
            print(f"[WARN] Not found: {path}")

    video_path = Path("outputs/video_level_report.json")
    if video_path.exists():
        plot_video_level(load_json(video_path), out_dir / "video_level_accuracy.png")
    else:
        print(f"[WARN] Not found: {video_path}")

    print("\nDone. Plots in outputs/plots/")


if __name__ == "__main__":
    main()
