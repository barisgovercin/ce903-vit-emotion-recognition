"""
Creates actor-disjoint train/val/test splits from processed face images.
No actor appears in more than one split, which prevents identity leakage.

How to run:
    python scripts/make_split.py
"""

import os
import json
import random
from collections import defaultdict


def collect_images(processed_root="data/processed_faces"):		# walks processed_faces/ and collects all jpeg paths with their labels
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

    return items


items = collect_images()
if not items:
    raise RuntimeError("No processed images found. Run scripts/extract_faces.py first.")

# group frames by video
by_video = defaultdict(list)
for it in items:
    by_video[it["video"]].append(it)

# group videos by actor so we can split at actor level
# folder format: "actor12_02-01-06-01-02-02-12" -> actor = "actor12"
actor_to_videos = defaultdict(list)
for v in by_video:
    actor_id = v.split("_")[0]
    actor_to_videos[actor_id].append(v)

actors = sorted(actor_to_videos.keys())
random.seed(42)
random.shuffle(actors)

n_a = len(actors)
n_train_a = int(0.70 * n_a)
n_val_a   = int(0.20 * n_a)

train_actors = set(actors[:n_train_a])
val_actors   = set(actors[n_train_a:n_train_a + n_val_a])
test_actors  = set(actors[n_train_a + n_val_a:])

train_videos = set(v for a in train_actors for v in actor_to_videos[a])
val_videos   = set(v for a in val_actors   for v in actor_to_videos[a])
test_videos  = set(v for a in test_actors  for v in actor_to_videos[a])

train_items = [it for v in train_videos for it in by_video[v]]
val_items   = [it for v in val_videos   for it in by_video[v]]
test_items  = [it for v in test_videos  for it in by_video[v]]

os.makedirs("data", exist_ok=True)
with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(train_items, f, indent=2)
with open("data/val.json", "w", encoding="utf-8") as f:
    json.dump(val_items, f, indent=2)
with open("data/test.json", "w", encoding="utf-8") as f:
    json.dump(test_items, f, indent=2)

print("=== ACTOR-LEVEL SPLIT SUMMARY ===")
print(f"actors total : {n_a} | train: {len(train_actors)} | val: {len(val_actors)} | test: {len(test_actors)}")
print(f"train actors : {sorted(train_actors)}")
print(f"val actors   : {sorted(val_actors)}")
print(f"test actors  : {sorted(test_actors)}")
print(f"videos total : {len(by_video)}")
print(f"videos       : train={len(train_videos)} | val={len(val_videos)} | test={len(test_videos)}")
print(f"frames       : train={len(train_items)} | val={len(val_items)} | test={len(test_items)}")
print("Saved: data/train.json, data/val.json, data/test.json")
