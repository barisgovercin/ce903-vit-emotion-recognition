"""
Extracts face crops from the manual test actors (Actor_03, Actor_10, Actor_18).
Covers all 5 emotion categories, not just a single one.

How to run:
    python scripts/extract_manual_faces.py
"""

import os
import cv2
from pathlib import Path
from collections import Counter
from tqdm import tqdm

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
}

VALID_MODALITIES = {"01", "02"}		# 01=audiovisual, 02=video-only


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_ravdess_filename(name):		# breaks a ravdess filename into its component fields
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    if any(len(x) != 2 for x in parts):
        return None
    modality, vocal, emotion, intensity, statement, repetition, actor = parts
    return {"modality": modality, "emotion": emotion, "actor": actor}


def extract_faces_from_video(video_path, out_dir, stride=5, min_size=64):		# haar cascade face detection, keeps largest face, saves 224x224
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
    src_root = Path("data/manual_test/ravdess_raw")
    out_root = Path("data/manual_test/processed_faces")

    if not src_root.exists():
        raise RuntimeError(f"Source folder not found: {src_root}")

    mp4_files = list(src_root.rglob("*.mp4"))

    # filter to our 5 emotions and video modalities only
    videos = []
    for fpath in mp4_files:
        meta = parse_ravdess_filename(fpath.name)
        if meta is None:
            continue
        if meta["modality"] not in VALID_MODALITIES:
            continue
        label = EMOTION_MAP.get(meta["emotion"])
        if label is None:
            continue
        videos.append((fpath, label, meta))

    print(f"Found {len(videos)} videos across all emotions")

    counts = Counter(v[1] for v in videos)
    for label, count in sorted(counts.items()):
        print(f"  {label:8s}: {count} video(s)")

    skipped = 0
    for fpath, label, meta in tqdm(videos, desc="Extracting faces"):
        folder_name = f"actor{meta['actor']}_{fpath.stem}"
        out_dir = out_root / label / folder_name

        if out_dir.exists() and any(f.endswith(".jpg") for f in os.listdir(out_dir)):
            skipped += 1
            continue

        saved = extract_faces_from_video(str(fpath), str(out_dir))
        if saved == 0:
            print(f"[WARN] No faces saved for {fpath.name}")

    if skipped:
        print(f"[INFO] {skipped} video(s) already processed, skipped.")

    # show how many frames we got per emotion
    print("\n=== MANUAL TEST FACE EXTRACTION SUMMARY ===")
    for label in sorted(EMOTION_MAP.values()):
        label_dir = out_root / label
        if label_dir.exists():
            total = sum(
                len([f for f in os.listdir(label_dir / d) if f.endswith(".jpg")])
                for d in os.listdir(label_dir)
                if (label_dir / d).is_dir()
            )
            print(f"  {label:8s}: {total} frames")
        else:
            print(f"  {label:8s}: 0 frames (no folder)")

    print(f"\nDone. Faces saved under {out_root}")


if __name__ == "__main__":
    main()
