"""
Builds a JSON manifest for the CREMA-D test faces so eval_vit.py can read them.

How to run:
    python scripts/make_crema_test_json.py
"""

import os
import json
from collections import Counter


def make_crema_test_json(
    processed_root="data/manual_test/crema_processed_faces",
    out_path="data/manual_test/crema_test.json",
):		# walks crema_processed_faces/ and writes a json list that eval_vit.py expects
    items = []

    for label in sorted(os.listdir(processed_root)):
        label_dir = os.path.join(processed_root, label)
        if not os.path.isdir(label_dir):
            continue

        for video_folder in os.listdir(label_dir):
            vdir = os.path.join(label_dir, video_folder)
            if not os.path.isdir(vdir):
                continue

            for fname in os.listdir(vdir):
                if not fname.lower().endswith(".jpg"):
                    continue

                fpath = os.path.join(vdir, fname)
                if not os.path.exists(fpath):
                    continue

                items.append({"path": fpath, "label": label, "video": video_folder})

    if not items:
        raise RuntimeError(f"No images found under: {processed_root}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    counts = Counter(it["label"] for it in items)
    print("=== CREMA-D TEST JSON SUMMARY ===")
    print(f"Total frames: {len(items)}")
    for label, count in sorted(counts.items()):
        print(f"  {label:8s}: {count}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_crema_test_json()
