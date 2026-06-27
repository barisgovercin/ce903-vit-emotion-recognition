"""
Sorts raw RAVDESS videos into emotion-labelled folders.
Parses the 7-field filename to get emotion and actor, then copies each video
into data/raw_videos/<label>/ with the actor id prefixed to avoid name collisions.

How to run:
    python scripts/prepare_ravdess.py
    python scripts/prepare_ravdess.py --src data/ravdess_raw --dst data/raw_videos
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import Counter

# ravdess filename: 01-01-03-02-02-02-01.mp4
# fields: modality-vocal-emotion-intensity-statement-repetition-actor

EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    # skipping: 02=calm, 07=disgust, 08=surprised
}

VALID_MODALITIES = {"01", "02"}		# 01=audiovisual, 02=video-only (skip 03=audio-only)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_ravdess_filename(name: str):		# splits a ravdess filename into its 7 fields, returns None if format is wrong
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    if any(len(x) != 2 for x in parts):
        return None

    modality, vocal, emotion, intensity, statement, repetition, actor = parts
    return {
        "modality": modality,
        "vocal": vocal,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Organize RAVDESS videos into emotion-labeled folders for the face extraction pipeline."
    )
    parser.add_argument(
        "--src", type=str, default="data/ravdess_raw",
        help="Source folder containing Actor_XX subdirectories (default: data/ravdess_raw)",
    )
    parser.add_argument(
        "--dst", type=str, default="data/raw_videos",
        help="Destination root folder where label subfolders will be created (default: data/raw_videos)",
    )
    parser.add_argument("--video_only", action="store_true",
        help="Only include video-only modality (02). By default both AV (01) and video-only (02) are included.",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files to destination (default behavior).")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying (saves disk space).")

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise SystemExit(f"[ERROR] Source path does not exist: {src_root}")

    for label in sorted(set(EMOTION_MAP.values())):
        ensure_dir(dst_root / label)

    mp4_files = list(src_root.rglob("*.mp4"))
    if not mp4_files:
        raise SystemExit(f"[ERROR] No .mp4 files found under: {src_root}")

    allowed_modalities = VALID_MODALITIES
    op = "move" if args.move else "copy"

    counts = Counter()
    skipped = Counter()

    for fpath in mp4_files:
        meta = parse_ravdess_filename(fpath.name)
        if meta is None:
            skipped["bad_name"] += 1
            continue

        if meta["modality"] not in allowed_modalities:
            skipped[f"modality_{meta['modality']}"] += 1
            continue

        label = EMOTION_MAP.get(meta["emotion"])
        if label is None:
            skipped[f"emotion_{meta['emotion']}"] += 1
            continue

        # prefix with actor id so filenames don't collide across actors
        out_name = f"actor{meta['actor']}_{fpath.name}"
        out_path = dst_root / label / out_name

        if out_path.exists():
            skipped["already_exists"] += 1
            continue

        if op == "move":
            shutil.move(str(fpath), str(out_path))
        else:
            shutil.copy2(str(fpath), str(out_path))

        counts[label] += 1

    print("\n=== RAVDESS PREP SUMMARY ===")
    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Modalities:  {sorted(list(allowed_modalities))}  (01=AV, 02=video-only)")
    print(f"Operation:   {op}")
    print("\nFiles per class:")
    for k in ["neutral", "happy", "sad", "angry", "fearful"]:
        print(f"  {k:8s}: {counts[k]}")
    print("\nSkipped:")
    for k, v in skipped.most_common():
        print(f"  {k:16s}: {v}")
    print("\nDone. Raw videos are ready under data/raw_videos/<label>/")


if __name__ == "__main__":
    main()
