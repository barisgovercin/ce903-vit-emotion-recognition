"""
Real-time emotion prediction from webcam, iPhone stream, or a single image.
Loads the trained ViT checkpoint and runs Haar Cascade face detection + inference.

How to run:
    Single image:  python scripts/infer.py --image face.jpg
    Webcam:        python scripts/infer.py --webcam
    Webcam (id):   python scripts/infer.py --webcam --cam_id 1
    iPhone stream: python scripts/infer.py --webcam --iphone http://10.130.234.49:8080
"""

import argparse
import os
import sys
import urllib.request
import threading
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vit_baseline import build_vit

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

EMOTION_COLOR = {		# bgr colors for each emotion label
    "angry":   (0,   0,   255),
    "fearful": (0,   140, 255),
    "happy":   (0,   200, 0),
    "neutral": (200, 200, 200),
    "sad":     (255, 80,  80),
}

INFER_TF = transforms.Compose([		# same preprocessing as training: resize, tensor, imagenet normalize
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


class IPhoneStreamReader:		# reads mjpeg stream from iphone camera app over http
    def __init__(self, base_url: str):
        self.stream_url = base_url.rstrip("/") + "/stream"
        self._frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"iPhone stream: {self.stream_url}")

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _loop(self):
        while self._running:
            try:
                stream = urllib.request.urlopen(self.stream_url, timeout=5)
                data = b""
                while self._running:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    data += chunk
                    latest_frame = None
                    search_start = 0
                    while True:
                        start = data.find(b'\xff\xd8', search_start)
                        if start == -1:
                            break
                        end = data.find(b'\xff\xd9', start + 2)
                        if end == -1:
                            break
                        jpg = data[start:end+2]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            latest_frame = frame
                        search_start = end + 2
                    if latest_frame is not None:
                        with self._lock:
                            self._frame = latest_frame
                    if search_start > 0:
                        data = data[search_start:]
                    if len(data) > 500000:
                        data = b""
            except Exception:
                if self._running:
                    time.sleep(0.5)


class InferenceWorker:		# runs face detection + prediction in a background thread so the display loop never blocks
    def __init__(self, model, idx_to_class, device, face_cascade):
        self.model = model
        self.idx_to_class = idx_to_class
        self.device = device
        self.face_cascade = face_cascade

        self._input_frame = None
        self._result = ("...", 0.0, {}, None)
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def submit(self, frame):		# send a new frame -- overwrites any unprocessed one
        with self._lock:
            self._input_frame = frame
        self._new_frame.set()

    def get_result(self):
        with self._lock:
            return self._result

    def _loop(self):
        while self._running:
            self._new_frame.wait(timeout=1.0)
            self._new_frame.clear()

            with self._lock:
                frame = self._input_frame
                self._input_frame = None

            if frame is None:
                continue

            # detect at half resolution for speed, then scale coords back up
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )

            if len(faces) == 0:
                with self._lock:
                    self._result = ("no face", 0.0, {}, None)
                continue

            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            x, y, w, h = x*2, y*2, w*2, h*2
            bbox = (x, y, w, h)

            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            label, conf, probs_map = self._predict(face)

            with self._lock:
                self._result = (label, conf, probs_map, bbox)

    @torch.no_grad()
    def _predict(self, face_bgr):		# crop to tensor, run through model, return label + confidence + full probs
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = INFER_TF(Image.fromarray(face_rgb)).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        idx = probs.argmax().item()
        label = self.idx_to_class[idx]
        conf = probs[idx].item()
        probs_map = {self.idx_to_class[i]: probs[i].item() for i in range(len(probs))}
        return label, conf, probs_map


def load_model(ckpt_path: Path, device: torch.device):		# loads checkpoint and rebuilds model with trained weights
    ckpt = torch.load(str(ckpt_path), map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = ckpt["idx_to_class"]
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
    return model, class_to_idx, idx_to_class


def detect_largest_face(frame_bgr, face_cascade, min_size: int = 60):		# returns bounding box of the biggest face, or None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


@torch.no_grad()
def predict(face_bgr, model, idx_to_class, device):		# single face prediction: bgr crop in, label + confidence + probs out
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFER_TF(Image.fromarray(face_rgb)).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    idx = probs.argmax().item()
    label = idx_to_class[idx]
    conf = probs[idx].item()
    probs_map = {idx_to_class[i]: probs[i].item() for i in range(len(probs))}
    return label, conf, probs_map


def draw_results(frame, bbox, label, conf, probs_map):		# draws bounding box, label, and probability bar chart overlay
    if bbox is not None:
        x, y, w, h = bbox
        color = EMOTION_COLOR.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label.upper()}  {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 6, y), color, -1)
        cv2.putText(frame, text, (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    panel_x, panel_y, bar_max = 10, 20, 160
    sorted_probs = sorted(probs_map.items(), key=lambda kv: -kv[1])
    for i, (em, prob) in enumerate(sorted_probs):
        em_color = EMOTION_COLOR.get(em, (180, 180, 180))
        bar_w = int(prob * bar_max)
        row_y = panel_y + i * 36
        cv2.rectangle(frame, (panel_x, row_y),
                      (panel_x + bar_max, row_y + 24), (40, 40, 40), -1)
        cv2.rectangle(frame, (panel_x, row_y),
                      (panel_x + bar_w, row_y + 24), em_color, -1)
        cv2.putText(frame, f"{em[:3].upper()} {prob * 100:.0f}%",
                    (panel_x + bar_max + 6, row_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return frame


def run_image(image_path: str, model, idx_to_class, device, face_cascade):		# single image mode: detect face, predict, save result
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    bbox = detect_largest_face(img, face_cascade)
    if bbox is None:
        print("[WARN] No face detected -- using full image.")
        face = cv2.resize(img, (224, 224))
    else:
        x, y, w, h = bbox
        face = img[y:y + h, x:x + w]
    label, conf, probs_map = predict(face, model, idx_to_class, device)
    print(f"\n{'=' * 35}")
    print(f"  Prediction : {label.upper()}")
    print(f"  Confidence : {conf * 100:.1f}%")
    print(f"{'=' * 35}")
    for em, prob in sorted(probs_map.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(prob * 28)
        print(f"  {em:8s}  {prob * 100:5.1f}%  {bar}")
    img = draw_results(img, bbox, label, conf, probs_map)
    out_path = Path(image_path).stem + "_result.jpg"
    cv2.imwrite(out_path, img)
    print(f"Result saved: {out_path}")
    cv2.imshow("Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(model, idx_to_class, device, face_cascade, cam_id: int = 0):		# live webcam mode with async inference
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (id={cam_id})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam started. Press 'q' to quit.")

    worker = InferenceWorker(model, idx_to_class, device, face_cascade)
    worker.start()

    label, conf, probs_map, bbox = "...", 0.0, {}, None
    fps_time = time.time()
    fps, frame_idx = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx % 2 == 0:
            worker.submit(frame)

        label, conf, probs_map, bbox = worker.get_result()
        frame = draw_results(frame, bbox, label, conf, probs_map)

        if frame_idx % 15 == 0:
            fps = 15 / (time.time() - fps_time)
            fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1]-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection  [q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()


def run_iphone(base_url: str, model, idx_to_class, device, face_cascade):		# iphone camera stream mode
    reader = IPhoneStreamReader(base_url)
    reader.start()

    print("Waiting for iPhone connection...")
    for _ in range(30):
        if reader.get_frame() is not None:
            break
        time.sleep(0.2)

    if reader.get_frame() is None:
        print("Could not connect to iPhone!")
        return

    worker = InferenceWorker(model, idx_to_class, device, face_cascade)
    worker.start()

    print("Connected. Emotion detection running. Press 'q' to quit.")

    label, conf, probs_map, bbox = "...", 0.0, {}, None
    fps_time = time.time()
    fps, frame_idx = 0, 0

    while True:
        frame = reader.get_frame()
        if frame is None:
            continue

        frame_idx += 1
        worker.submit(frame)

        label, conf, probs_map, bbox = worker.get_result()
        frame = draw_results(frame, bbox, label, conf, probs_map)

        if frame_idx % 15 == 0:
            fps = 15 / (time.time() - fps_time)
            fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1]-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection -- iPhone  [q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    worker.stop()
    reader.stop()
    cv2.destroyAllWindows()
    print("Stream closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/vit_best.pth")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--iphone", type=str, default=None,
                        help="iPhone base URL, e.g. http://10.130.234.49:8080")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str)
    mode.add_argument("--webcam", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading model: {ckpt_path}")
    model, class_to_idx, idx_to_class = load_model(ckpt_path, device)
    print("Classes:", class_to_idx)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if args.image:
        run_image(args.image, model, idx_to_class, device, face_cascade)
    elif args.iphone:
        run_iphone(args.iphone, model, idx_to_class, device, face_cascade)
    else:
        run_webcam(model, idx_to_class, device, face_cascade, cam_id=args.cam_id)


if __name__ == "__main__":
    main()
