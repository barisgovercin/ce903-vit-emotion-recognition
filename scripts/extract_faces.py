"""
Extracts face crops from RAVDESS training videos using Haar Cascade.
Saves 224x224 JPEG faces under data/processed_faces/<label>/<video>/.

How to run:
    python scripts/extract_faces.py
"""

import os
import cv2
from tqdm import tqdm

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def iter_videos(root_dir: str):		# walks data/raw_videos/<label>/ and yields (label, video_path) pairs
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                yield label, os.path.join(label_dir, fname)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_faces_from_video(video_path: str, out_dir: str, stride: int = 5, min_size: int = 64):		# detects largest face every N frames, saves as 224x224 jpeg
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
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size),
        )

        # pick the biggest face (there's only one person per video)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, face)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    raw_root = os.path.join("data", "raw_videos")
    out_root = os.path.join("data", "processed_faces")
    ensure_dir(out_root)

    videos = list(iter_videos(raw_root))
    if not videos:
        print("No videos found. Put videos under: data/raw_videos/<label>/*.mp4")
        return

    skipped = 0
    for label, vpath in tqdm(videos, desc="Processing videos"):
        vname = os.path.splitext(os.path.basename(vpath))[0]
        out_dir = os.path.join(out_root, label, vname)

        # skip if already processed
        if os.path.isdir(out_dir) and any(f.endswith(".jpg") for f in os.listdir(out_dir)):
            skipped += 1
            continue

        ensure_dir(out_dir)
        saved = extract_faces_from_video(vpath, out_dir, stride=5)

        if saved == 0:
            print(f"[WARN] No faces saved for {vpath}")

    if skipped:
        print(f"[INFO] {skipped} video already processed, skipped.")

    print("Done. Faces saved under data/processed_faces/")


if __name__ == "__main__":
    main()
