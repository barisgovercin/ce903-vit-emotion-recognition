"""
Extracts face crops from CREMA-D videos for cross-dataset testing.
Output goes to crema_processed_faces/ to keep it separate from RAVDESS data.

How to run:
    python scripts/crema_extract_faces.py
"""

import os
import cv2
from pathlib import Path
from collections import Counter
from tqdm import tqdm

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

CREMA_EMOTION_MAP = {		# maps crema-d emotion codes to our 5-class labels
    "ANG": "angry",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".flv", ".mkv"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_crema_filename(name):		# parses crema-d filename like '1003_IEO_ANG_HI.mp4' into actor/emotion/intensity
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    actor = parts[0]
    emotion = parts[2]
    intensity = parts[3]

    return {"actor": actor, "emotion": emotion, "intensity": intensity}


def extract_faces_from_video(video_path, out_dir, stride=5, min_size=64):		# haar cascade detection, largest face, 224x224 jpeg
    ensure_dir(out_dir)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg"), face)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    src_root = Path("data/manual_test/crema")
    out_root = Path("data/manual_test/crema_processed_faces")

    if not src_root.exists():
        raise RuntimeError(f"Source folder not found: {src_root}")

    video_files = [
        f for f in src_root.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        raise RuntimeError(f"No video files found in {src_root}")

    # only keep videos whose emotion maps to one of our 5 classes
    videos = []
    skipped = Counter()

    for fpath in video_files:
        meta = parse_crema_filename(fpath.name)
        if meta is None:
            skipped["bad_name"] += 1
            continue
        label = CREMA_EMOTION_MAP.get(meta["emotion"])
        if label is None:
            skipped[f"emotion_{meta['emotion']}"] += 1
            continue
        videos.append((fpath, label, meta))

    print(f"Found {len(videos)} CREMA-D videos with matching emotions")

    if skipped:
        print("Skipped:")
        for reason, count in skipped.most_common():
            print(f"  {reason}: {count}")

    emotion_counts = Counter(v[1] for v in videos)
    for label, count in sorted(emotion_counts.items()):
        print(f"  {label:8s}: {count} video(s)")

    already_done = 0
    for fpath, label, meta in tqdm(videos, desc="Extracting CREMA-D faces"):
        folder_name = f"crema_{fpath.stem}"
        out_dir = out_root / label / folder_name

        if out_dir.exists() and any(f.endswith(".jpg") for f in os.listdir(out_dir)):
            already_done += 1
            continue

        saved = extract_faces_from_video(str(fpath), str(out_dir))
        if saved == 0:
            print(f"[WARN] No faces saved for {fpath.name}")

    if already_done:
        print(f"[INFO] {already_done} video(s) already processed, skipped.")

    print("\n=== CREMA-D FACE EXTRACTION SUMMARY ===")
    for label in sorted(CREMA_EMOTION_MAP.values()):
        label_dir = out_root / label
        if label_dir.exists():
            folders = [d for d in os.listdir(label_dir) if (label_dir / d).is_dir()]
            total = sum(
                len([f for f in os.listdir(label_dir / d) if f.endswith(".jpg")])
                for d in folders
            )
            print(f"  {label:8s}: {total} frames ({len(folders)} videos)")
        else:
            print(f"  {label:8s}: 0 frames")

    print(f"\nDone. Faces saved under {out_root}")
    print("Next steps:")
    print("  python scripts/make_crema_test_json.py")
    print("  python scripts/eval_vit.py --json data/manual_test/crema_test.json --ckpt outputs/vit_best.pth")


if __name__ == "__main__":
    main()
