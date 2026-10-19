from __future__ import annotations

from pathlib import Path
from typing import Any, List
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

# Resolve model paths relative to this script's location (works on any machine).
_BASE_DIR = Path(__file__).resolve().parent
DETECT_MODEL_PATH = str(_BASE_DIR / "model" / "phone_watch_dataset.pt")
R3D_MODEL_PATH = str(_BASE_DIR / "model" / "r3d_normal_peek_best.pt")

DETECT_CLASSES = {"phone", "watch"}
R3D_CLASSES = ["normal", "peek"]

IMG_SIZE = 224
NUM_FRAMES = 16

YOLO_MIN_CONF = 0.60
YOLO_MIN_HITS = 1

R3D_THRESHOLD = 0.50
R3D_MARGIN = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_detect_model = None
_r3d_model = None


class R3DClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        model = models.video.r3d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_models():
    global _detect_model, _r3d_model

    if _detect_model is None:
        _detect_model = YOLO(DETECT_MODEL_PATH)

    if _r3d_model is None:
        ckpt = torch.load(R3D_MODEL_PATH, map_location=DEVICE)

        model = R3DClassifier(num_classes=len(R3D_CLASSES)).to(DEVICE)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)

        model.eval()
        _r3d_model = model

    return _detect_model, _r3d_model


def sample_indices(total_frames: int, count: int) -> List[int]:
    if total_frames <= 0:
        return [0] * count

    if total_frames < count:
        idxs = list(range(total_frames))
        while len(idxs) < count:
            idxs.append(idxs[-1] if idxs else 0)
        return idxs

    return np.linspace(0, total_frames - 1, count).astype(int).tolist()


def make_top3(prob_map: dict[str, float]):
    items = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    return [{"label": k, "confidence": float(v)} for k, v in items]


def detect_objects_in_video(video_path: str, sample_frames: int = 32):
    detect_model, _ = load_models()

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return None, 0.0

    idxs = sample_indices(total, sample_frames)

    counts = {"phone": 0, "watch": 0}
    confs = {"phone": [], "watch": []}

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        result = detect_model(frame, conf=YOLO_MIN_CONF, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        seen_this_frame = set()

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = str(detect_model.names[cls_id]).lower().strip()

            if label in DETECT_CLASSES and conf >= YOLO_MIN_CONF:
                seen_this_frame.add(label)
                confs[label].append(conf)

        for label in seen_this_frame:
            counts[label] += 1

    cap.release()

    best_label = None
    best_score = 0.0

    for label in ["phone", "watch"]:
        if counts[label] >= YOLO_MIN_HITS and confs[label]:
            avg_conf = sum(confs[label]) / len(confs[label])
            if avg_conf > best_score:
                best_score = avg_conf
                best_label = label

    return best_label, best_score


def load_video_for_r3d(video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return None

    idxs = sample_indices(total, NUM_FRAMES)
    frames = []

    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
        frames.append(frame)

    cap.release()

    arr = np.array(frames, dtype=np.float32)   # T,H,W,C
    arr = np.transpose(arr, (3, 0, 1, 2))      # C,T,H,W
    arr = np.expand_dims(arr, axis=0)          # 1,C,T,H,W

    return torch.tensor(arr, dtype=torch.float32)


def predict_r3d(video_path: str):
    _, r3d_model = load_models()

    x = load_video_for_r3d(video_path)

    if x is None:
        return "normal", 1.0, [1.0, 0.0]

    x = x.to(DEVICE)

    with torch.no_grad():
        logits = r3d_model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    normal_p = float(probs[0])
    peek_p = float(probs[1])

    if peek_p >= R3D_THRESHOLD and (peek_p - normal_p) >= R3D_MARGIN:
        return "peek", peek_p, probs.tolist()

    return "normal", normal_p, probs.tolist()


def predict_cheat_video(video_path: str) -> dict[str, Any]:
    obj_label, obj_conf = detect_objects_in_video(video_path, sample_frames=32)

    print("YOLO DETECT:", obj_label, obj_conf)

    if obj_label in DETECT_CLASSES and obj_conf >= YOLO_MIN_CONF:
        return {
            "result": "cheat",
            "predicted_label": obj_label,
            "predicted_confidence": float(obj_conf),
            "normal_confidence": 0.0,
            "top3": [{"label": obj_label, "confidence": float(obj_conf)}],
            "source": "yolo_detect",
            "reason": "phone_or_watch_detected",
        }

    r3d_label, r3d_conf, r3d_probs = predict_r3d(video_path)

    print("R3D:", r3d_label, r3d_conf, r3d_probs)

    prob_map = {
        "normal": float(r3d_probs[0]),
        "peek": float(r3d_probs[1]),
    }

    if r3d_label == "peek" and r3d_conf >= R3D_THRESHOLD:
        return {
            "result": "cheat",
            "predicted_label": "peek",
            "predicted_confidence": float(r3d_conf),
            "normal_confidence": float(r3d_probs[0]),
            "top3": make_top3(prob_map),
            "source": "r3d",
            "reason": "peek_detected",
        }

    return {
        "result": "normal",
        "predicted_label": "normal",
        "predicted_confidence": float(r3d_probs[0]),
        "normal_confidence": float(r3d_probs[0]),
        "top3": make_top3(prob_map),
        "source": "combined",
        "reason": "no_phone_no_watch_no_peek",
    }