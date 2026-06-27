"""
Builds a JSON manifest from the manual test face images so eval_vit.py can read them.

How to run:
    python scripts/make_manual_test_json.py
"""

import os
import json


def make_manual_test_json(
    processed_root="data/manual_test/processed_faces",
    out_path="data/manual_test/manual_test.json",
):		# walks processed_faces/ and writes a json list of {path, label, video} entries
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

    from collections import Counter
    counts = Counter(it["label"] for it in items)
    print("=== MANUAL TEST JSON SUMMARY ===")
    print(f"Total frames: {len(items)}")
    for label, count in sorted(counts.items()):
        print(f"  {label:8s}: {count}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_manual_test_json()
