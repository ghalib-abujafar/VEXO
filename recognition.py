from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from insightface.app import FaceAnalysis

# NOTE:
# Creating FaceAnalysis on import is fine for a single-process server.
# In Flask debug mode, the reloader may import twice. This is still OK,
# but you can disable the reloader in production or keep debug=False.
_face_app: FaceAnalysis | None = None


def _get_face_app() -> FaceAnalysis:
    """Lazy-load the face model to avoid heavy init at import time."""
    global _face_app
    if _face_app is None:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0)
        _face_app = app
    return _face_app


def detect_and_encode(image_bgr: np.ndarray) -> tuple[list[list[int]], list[np.ndarray]]:
    """
    Detect faces in a BGR image and return:
    - boxes: list of [x1, y1, x2, y2] as ints (JSON-friendly)
    - encodings: list of embeddings (typically 512 floats, L2-normalized)
    """
    app = _get_face_app()
    faces = app.get(image_bgr)

    boxes: list[list[int]] = []
    encodings: list[np.ndarray] = []

    for f in faces:
        # bbox is [x1, y1, x2, y2]
        bbox = f.bbox.astype(int).tolist()
        emb = np.asarray(f.normed_embedding, dtype=np.float32)

        boxes.append(bbox)
        encodings.append(emb)

    return boxes, encodings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity.
    For normalized vectors, this is simply dot(a, b).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    # Safe normalization in case vectors are not normalized.
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    a = a / a_norm
    b = b / b_norm
    return float(np.dot(a, b))


def best_match(
    known_encs: Sequence[np.ndarray],
    labels: Sequence[str],
    query_enc: np.ndarray,
    min_similarity: float = 0.60,
) -> tuple[str, float]:
    """
    Find the closest face embedding among known embeddings.

    Returns:
      (label, score)

    - min_similarity is the minimum cosine similarity to accept a match.
    - If no match passes min_similarity, returns ("Unknown", best_score).
    """
    if not known_encs:
        return "NoData", 0.0

    # Convert known encodings to a matrix for vectorized similarity.
    # Shapes:
    #   known_matrix: (N, D)
    #   query: (D,)
    known_matrix = np.asarray(known_encs, dtype=np.float32)
    query = np.asarray(query_enc, dtype=np.float32)

    # Normalize (safe) to make dot product == cosine similarity.
    km_norms = np.linalg.norm(known_matrix, axis=1, keepdims=True)
    q_norm = np.linalg.norm(query)

    # Avoid division by zero.
    km_norms[km_norms == 0.0] = 1.0
    if q_norm == 0.0:
        return "Unknown", 0.0

    known_matrix = known_matrix / km_norms
    query = query / q_norm

    sims = known_matrix @ query  # (N,)
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])

    if best_sim < float(min_similarity):
        return "Unknown", best_sim

    # Defensive: labels and embeddings should have same length.
    if idx >= len(labels):
        return "Unknown", best_sim

    return str(labels[idx]), best_sim