from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recognition import detect_and_encode


# Use project-relative dataset directory (same folder as this file).
DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def capture_samples_via_camera(label: str, num_images: int = 8, camera_index: int = 0) -> None:
    """
    Capture images from a local webcam for manual enrollment (terminal usage).
    Press SPACE to save a frame, ESC to exit early.
    """
    label = (label or "").strip()
    if not label:
        raise ValueError("label cannot be empty")

    person_dir = DATASET_DIR / label
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    count = 0
    try:
        while count < num_images:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            cv2.imshow("Capture (press SPACE to save)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                out_path = person_dir / f"img_{count:02d}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"✅ Saved {out_path}")
                count += 1

            elif key == 27:  # ESC
                break

        cv2.destroyAllWindows()

    finally:
        cap.release()


def _decode_data_url_image(img_data_url: str) -> bytes | None:
    """
    Decode a typical 'data:image/jpeg;base64,...' string safely.
    Returns raw bytes or None if invalid.
    """
    if not img_data_url or "," not in img_data_url:
        return None
    try:
        b64_part = img_data_url.split(",", 1)[1]
        return base64.b64decode(b64_part, validate=True)
    except Exception:
        return None


def save_frame_from_web(label: str, img_b64: str, index: int) -> str:
    """
    Save a frame received from the web (Base64 data URL) into dataset/<label>/img_XX.jpg.
    Returns the saved file path as string.
    """
    label = (label or "").strip()
    if not label:
        raise ValueError("label cannot be empty")

    person_dir = DATASET_DIR / label
    person_dir.mkdir(parents=True, exist_ok=True)

    img_bytes = _decode_data_url_image(img_b64)
    if not img_bytes:
        raise ValueError("Invalid base64 image payload")

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")

    index = int(index)
    index = max(0, min(index, 9999))
    file_path = person_dir / f"img_{index:02d}.jpg"

    cv2.imwrite(str(file_path), frame)
    return str(file_path)


def build_embeddings_for_label(label: str) -> list[np.ndarray]:
    """
    Build embeddings for all *.jpg files in dataset/<label>/.
    - Takes only the first face encoding per image.
    - Skips images with no detected faces.
    """
    label = (label or "").strip()
    if not label:
        raise ValueError("label cannot be empty")

    person_dir = DATASET_DIR / label
    if not person_dir.exists():
        return []

    encs: list[np.ndarray] = []

    for p in sorted(person_dir.glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is None:
            continue

        _, encs_list = detect_and_encode(img)
        if not encs_list:
            continue

        # Only take the first detected face encoding.
        encs.append(encs_list[0])

    return encs


def serialize_embeddings(encs: list[np.ndarray]) -> str:
    """Serialize embeddings to JSON for SQLite storage."""
    return json.dumps([e.tolist() for e in encs])


def deserialize_embeddings(payload: str) -> list[np.ndarray]:
    """Deserialize embeddings JSON back into numpy arrays."""
    data = json.loads(payload or "[]")
    return [np.asarray(x, dtype=float) for x in data]