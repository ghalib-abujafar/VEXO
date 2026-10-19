from __future__ import annotations

import base64
import os


import sqlite3
import threading
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from shutil import rmtree
from typing import Any

import cv2
import numpy as np
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from database import (
    # core
    get_conn,
    init_db,
    # users/face
    add_log,
    add_user,
    delete_user,
    get_user_by_name,
    get_users,
    # logs/attendance
    get_attendance_for_date,
    get_last_action_for_user,
    get_logs,
    ping_attendance,
    # accounts
    create_account,
    get_account_by_username,
    # courses + requests + course_students
    accept_course_request,
    add_student_to_course,
    create_course,
    create_course_request,
    delete_course,
    get_course_students,
    get_student_courses,
    get_students_not_in_course,
    get_teacher_course_requests,
    get_teacher_courses,
    link_account_student_id,
    reject_course_request,
    remove_student_from_course,
)

from enrollment import build_embeddings_for_label, deserialize_embeddings, serialize_embeddings
from recognition import best_match, detect_and_encode
#from video_cheat_model import predict_cheat_video, is_cheating_label
from infer_cheat_video_hybrid import predict_cheat_video

# -----------------------------------------------------------------------------
# App & Security Settings
# -----------------------------------------------------------------------------

app = Flask(__name__)

# Use an environment variable in production. Fallback is for local dev only.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")

# Basic cookie hardening (works without breaking local dev).
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.getenv("SESSION_COOKIE_SECURE", "")),  # set to 1 behind HTTPS
)


# -----------------------------------------------------------------------------
# Access control (Roles & Allowed Endpoints)
# -----------------------------------------------------------------------------

PUBLIC_ENDPOINTS = {"index", "login", "register", "static"}

STUDENT_ALLOWED = {
    "index",
    "logout",
    "my_attendance_history",
    "student_courses",
    "request_course",
    "leave_course",
    "enroll_me",
    "upload_frame",
    "finalize_me",
    "verify_enrollment",
    "cancel_enrollment",
}

TEACHER_ALLOWED = {
    "index",
    "logout",
    # courses page
    "teacher_courses",
    "teacher_accept_request",
    "teacher_reject_request",
    "delete_course_route",
    "my_courses_json",
    # course students management
    "course_students_page",
    "api_course_students",
    "api_course_available_students",
    "api_course_add_student",
    "api_course_remove_student",
    # recognition
    "recognize_live",
    "recognize_frame",
    # attendance/logs
    "attendance",
    "logs",
    "clear_logs_route",
    "log_action",
    # users (face dataset management)
    "users_add",
    "finalize_user",
    "check_student_id",
    "delete_user_route",
    "upload_frame",
    # cheat review
    "cheat_review",
    "upload_cheat_video",
    "serve_cheat_video",
    "approve_cheat_clip",
    "reject_cheat_clip",
    "delete_cheat_clip",
}

ADMIN_ALLOWED = {
    "index",
    "logout",
    "admin_dashboard",
    "admin_create_teacher",
    "admin_delete_account",
    "admin_reset_password",
    "admin_reset_face",
    "admin_list_accounts",
}

# When a student is logged in but has NOT enrolled their face yet,
# only allow these endpoints (plus public endpoints).
ENROLL_ALLOWED_ENDPOINTS = {
    "index",
    "login",
    "register",
    "logout",
    "enroll_me",
    "upload_frame",
    "finalize_me",
    "verify_enrollment",
    "cancel_enrollment",
    "static",
}


def current_user() -> dict[str, Any]:
    """Return a safe dict for the current session user."""
    u = session.get("user")
    return u if isinstance(u, dict) else {}

def set_user_session_fields(student_id: str, name: str) -> None:
    u = session.get("user") or {}

    # نعيد بناء الدكت حتى Flask يعرف إنه تغيّر
    new_user = {
        **u,
        "student_id": student_id,
        "display_name": name,
    }

    session["user"] = new_user
    session.modified = True

def teacher_only() -> bool:
    """Return True if current user is a teacher."""
    return current_user().get("role") == "teacher"


def admin_only() -> bool:
    """Return True if current user is an admin."""
    return current_user().get("role") == "admin"


def student_face_enrolled(student_id: str | None) -> bool:
    """Check if a student_id already has a face profile stored."""
    if not student_id:
        return False
    with get_conn() as conn:
        cur = conn.execute("SELECT 1 FROM users WHERE student_id=? LIMIT 1", (student_id,))
        return cur.fetchone() is not None


@app.before_request
def require_login_and_role():
    endpoint = request.endpoint or ""

    # Public pages or static files
    if endpoint in PUBLIC_ENDPOINTS or endpoint.startswith("static"):
        return None

    # Must be logged in
    if not session.get("user"):
        return redirect(url_for("login"))

    user = current_user()
    role = user.get("role")

    if role == "student":
        sid = user.get("student_id")

        # If face not enrolled, restrict to enrollment flow
        if not sid or not student_face_enrolled(sid):
            if endpoint not in ENROLL_ALLOWED_ENDPOINTS:
                flash("flash.must_enroll_first", "warning")
                return redirect(url_for("enroll_me"))

        # After enrollment, restrict to student pages only
        if endpoint not in STUDENT_ALLOWED:
            flash("flash.not_allowed_student", "danger")
            return redirect(url_for("index"))

    elif role == "teacher":
        if endpoint not in TEACHER_ALLOWED:
            flash("flash.not_allowed_teacher", "danger")
            return redirect(url_for("index"))

    elif role == "admin":
        if endpoint not in ADMIN_ALLOWED:
            flash("flash.not_allowed_teacher", "danger")
            return redirect(url_for("admin_dashboard"))

    else:
        session.pop("user", None)
        return redirect(url_for("login"))

    return None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def decode_data_url_image(img_data_url: str) -> bytes | None:
    """
    Decode a typical 'data:image/jpeg;base64,...' string safely.
    Returns raw image bytes or None if invalid.
    """
    if not img_data_url or "," not in img_data_url:
        return None
    try:
        b64_part = img_data_url.split(",", 1)[1]
        return base64.b64decode(b64_part, validate=True)
    except Exception:
        return None


def safe_int(value: Any, default: int = 0) -> int:
    """Parse an int safely."""
    try:
        return int(value)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Realtime recognition worker (process latest frame only)
# -----------------------------------------------------------------------------
 
_frame_lock = threading.Lock()
_latest_payload = {"img_b64": None, "mode": "attendance", "course_id": 0, "teacher_id": 0}
_latest_result: dict[str, Any] = {"status": "idle"}
_worker_ready = False

def load_course_embeddings(course_id: int, teacher_id: int):
    """
    Returns: (known_encs, labels, err)
    """
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row

        own = conn.execute(
            "SELECT 1 FROM courses WHERE id=? AND teacher_id=?",
            (course_id, teacher_id),
        ).fetchone()

        if not own:
            return [], [], "مادة غير صالحة (ليست من موادك)."

        rows = conn.execute(
            """
            SELECT u.name, u.embeddings
            FROM course_students cs
            JOIN accounts a ON a.id = cs.student_id
            JOIN users u ON u.student_id = a.student_id
            WHERE cs.course_id = ?
            """,
            (course_id,),
        ).fetchall()

    known_encs: list = []
    labels: list[str] = []
    for r in rows:
        encs = deserialize_embeddings(r["embeddings"])
        known_encs.extend(encs)
        labels.extend([r["name"]] * len(encs))

    if not known_encs:
        return [], [], "لا يوجد طلاب مسجلين في هذه المادة."

    return known_encs, labels, None


def _recognition_worker():
    global _latest_result, _worker_ready

    last_reload_ts = 0.0
    cached_known_encs: list = []
    cached_labels: list[str] = []
    cached_course_id = 0
    cached_teacher_id = 0

    # Keep last attendance ping per name to avoid spamming DB writes.
    last_ping: dict[str, datetime] = {}

    while True:
        with _frame_lock:
            img_b64 = _latest_payload.get("img_b64")
            mode = _latest_payload.get("mode", "attendance")
            course_id = safe_int(_latest_payload.get("course_id"), 0)
            teacher_id = safe_int(_latest_payload.get("teacher_id"), 0)

            # Clear the frame after taking a local copy so we don't re-process it.
            _latest_payload["img_b64"] = None

        if not img_b64:
            _worker_ready = True
            time.sleep(0.05)
            continue

        if course_id <= 0:
            _latest_result = {"status": "error", "message": "اختر مادة أولاً."}
            _worker_ready = True
            time.sleep(0.05)
            continue

        try:
            now_ts = time.time()
            need_reload = (
                (course_id != cached_course_id)
                or (teacher_id != cached_teacher_id)
                or ((now_ts - last_reload_ts) > 5)
            )

            if need_reload:
                known_encs, labels, err = load_course_embeddings(course_id, teacher_id)
                if err:
                    _latest_result = {
                        "status": "no_data" if "طلاب" in err else "error",
                        "message": err,
                    }
                    _worker_ready = True
                    time.sleep(0.05)
                    continue

                cached_known_encs, cached_labels = known_encs, labels
                cached_course_id = course_id
                cached_teacher_id = teacher_id
                last_reload_ts = now_ts

            if not cached_known_encs:
                _latest_result = {"status": "no_data"}
                _worker_ready = True
                time.sleep(0.05)
                continue

            img_bytes = decode_data_url_image(img_b64)
            if not img_bytes:
                _latest_result = {"status": "error", "message": "Invalid image payload."}
                _worker_ready = True
                time.sleep(0.05)
                continue

            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            boxes, encs = detect_and_encode(frame)

            # Final cheating decision is made only by the video model.
            if mode == "cheat":
                _latest_result = {"status": "ok"}
                _worker_ready = True
                time.sleep(0.05)
                continue

            if not encs:
                _latest_result = {"status": "no_face"}
                _worker_ready = True
                time.sleep(0.05)
                continue

            recognized = []
            now = datetime.now()
            today = now.date().isoformat()
            now_iso = now.isoformat()

            for enc in encs:
                label_name, score = best_match(cached_known_encs, cached_labels, enc)

                # Reject unknown or weak scores.
                if str(label_name).lower() == "unknown" or float(score) < 0.60:
                    continue

                name = str(label_name)
                user_row = get_user_by_name(name)
                if not user_row:
                    continue

                user_id = user_row["id"]
                student_id_text = user_row["student_id"]

                updated = False
                if mode == "attendance":
                    last = last_ping.get(name)
                    if (last is None) or ((now - last).total_seconds() >= 60):
                        ping_attendance(
                            user_id=user_id,
                            name=name,
                            student_id=student_id_text,
                            date_str=today,
                            now_iso=now_iso,
                            subject_id=course_id,
                        )
                        last_ping[name] = now
                        updated = True

                recognized.append({"name": name, "score": float(score), "updated": updated})

            _latest_result = {"status": "success", "recognized": recognized}
            _worker_ready = True

        except Exception:
            # Avoid leaking internal details in production; keep generic message.
            _latest_result = {"status": "error", "message": "Internal recognition error."}
            _worker_ready = True

        time.sleep(0.05)


threading.Thread(target=_recognition_worker, daemon=True).start()


# -----------------------------------------------------------------------------
# App initialization (DB + folders)
# -----------------------------------------------------------------------------

with app.app_context():
    init_db()
    Path("logs").mkdir(exist_ok=True)
    Path("dataset").mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Routes: Home / Auth
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    acc = get_account_by_username(username)
    if not acc or not check_password_hash(acc["password_hash"], password):
        flash("flash.invalid_login", "danger")
        return redirect(url_for("login"))

    # Resolve a friendly display name for students.
    display_name = acc.get("display_name") or acc["username"]
    if acc["role"] == "student":
        sid_lookup = acc.get("student_id") or acc.get("username")
        if sid_lookup:
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT name FROM users WHERE student_id=? LIMIT 1",
                    (sid_lookup,),
                ).fetchone()
            if row and row[0]:
                display_name = row[0]

    session["user"] = {
        "id": acc["id"],
        "username": acc["username"],
        "display_name": display_name,
        "role": acc["role"],
        "student_id": acc.get("student_id"),
    }

    flash("flash.login_success", "success")

    # Redirect admin to home page (they can access Admin Panel from there).
    if acc["role"] == "admin":
        return redirect(url_for("index"))

    # Force enrollment flow if student has a linked student_id but no face enrolled.
    if acc["role"] == "student":
        sid = acc.get("student_id")
        if sid and not student_face_enrolled(sid):
            return redirect(url_for("enroll_me"))

    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    name = request.form.get("name", "").strip()
    student_id = request.form.get("student_id", "").strip()
    username = (request.form.get("username", "").strip() or student_id).strip()
    password = request.form.get("password", "").strip()
    confirm = request.form.get("confirm", "").strip()

    if not name or not student_id or not password:
        flash("flash.fill_required_fields", "warning")
        return redirect(url_for("register"))

    import re
    if re.search(r"\d", name):
        flash("flash.name_no_numbers", "danger")
        return redirect(url_for("register"))

    if password != confirm:
        flash("flash.password_mismatch", "danger")
        return redirect(url_for("register"))

    if get_account_by_username(username):
        flash("flash.username_exists", "danger")
        return redirect(url_for("register"))

    with get_conn() as conn:
        cur = conn.execute("SELECT 1 FROM accounts WHERE student_id=? LIMIT 1", (student_id,))
        if cur.fetchone():
            flash("flash.student_id_has_account", "danger")
            return redirect(url_for("register"))

    password_hash = generate_password_hash(password)
    create_account(username=username, password_hash=password_hash, role="student", student_id=None, display_name=name)

    # Store pending enrollment data for the face enrollment flow.
    session["pending_name"] = name
    session["pending_student_id"] = student_id

    # تسجيل دخول تلقائي
    acc = get_account_by_username(username)

    session["user"] = {
        "id": acc["id"],
        "username": acc["username"],
        "display_name": name,
        "role": "student",
        "student_id": None
    }

    flash("flash.register_face", "info")

    # تحويل مباشر لتسجيل الوجه
    return redirect(url_for("enroll_me"))


@app.route("/logout")
def logout():
    user = current_user()

    # Clean up pending enrollment images if the student never completed verification.
    if user.get("role") == "student":
        pending_sid = (
            (session.get("pending_student_id") or "").strip()
            or (user.get("student_id") or "").strip()
            or (user.get("username") or "").strip()
        )
        if pending_sid and not student_face_enrolled(pending_sid):
            dataset_path = Path("dataset") / pending_sid
            if dataset_path.exists():
                # Delete pending enrollment file
                for f in ("pending_enrollment.json", "pending_embeddings.json"):
                    (dataset_path / f).unlink(missing_ok=True)
                # Delete all captured images
                for img in dataset_path.glob("img_*.jpg"):
                    img.unlink(missing_ok=True)
                # Remove folder if empty
                try:
                    dataset_path.rmdir()
                except OSError:
                    pass

    session.pop("user", None)
    session.pop("pending_name", None)
    session.pop("pending_student_id", None)
    flash("flash.logged_out", "info")
    return redirect(url_for("login"))


# -----------------------------------------------------------------------------
# Student: Face Enrollment (self-enroll)
# -----------------------------------------------------------------------------

@app.route("/enroll_me")
def enroll_me():
    u = current_user()
    if not u or u.get("role") != "student":
        flash("flash.students_only", "danger")
        return redirect(url_for("index"))

    sid = u.get("student_id")

    # If already linked and enrolled, no need to enroll again.
    if sid and student_face_enrolled(sid):
        flash("flash.already_enrolled", "success")
        return redirect(url_for("index"))

    # Use pending session data first; then account data; finally fallback to username.
    pending_sid = (
        (session.get("pending_student_id") or "").strip()
        or (u.get("student_id") or "").strip()
        or (u.get("username") or "").strip()
    )
    if not pending_sid:
        flash("flash.missing_student_id_enroll", "warning")
        return redirect(url_for("logout"))

    pending_name = (
        (session.get("pending_name") or "").strip()
        or (u.get("display_name") or "").strip()
        or (u.get("username") or "").strip()
    )

    return render_template("enroll_me.html", student_id=pending_sid, name=pending_name)


@app.route("/finalize_me", methods=["POST"])
def finalize_me():
    u = current_user()
    if not u or u.get("role") != "student":
        return {"status": "error", "message": "Unauthorized"}, 401

    student_id = (
        (session.get("pending_student_id") or "").strip()
        or (u.get("student_id") or "").strip()
        or (u.get("username") or "").strip()
    )
    name = (
        (session.get("pending_name") or "").strip()
        or (u.get("display_name") or "").strip()
        or (u.get("username") or "").strip()
    )

    if not student_id or student_id.lower() == "none":
        return {"status": "error", "message": "Student ID is missing for enrollment."}, 400

    # If already enrolled, just link the account and finish.
    if student_face_enrolled(student_id):
        try:
            link_account_student_id(u["username"], student_id)
            session["user"]["student_id"] = student_id
            session["user"]["display_name"] = name
        except Exception:
            pass
        return {"status": "success", "message": "You are already enrolled."}, 200

    dataset_path = Path("dataset") / student_id
    if not dataset_path.exists() or not any(dataset_path.glob("img_*.jpg")):
        return {"status": "error", "message": f"No images found for folder: {student_id}."}, 400

    try:
        encs = build_embeddings_for_label(student_id)
        if not encs:
            return {"status": "error", "message": "No faces were detected in the captured images."}, 400

        import json as _json

        # Store everything in a single temp file on disk.
        # Session cookies are ~4KB max, embeddings are ~25KB — must use disk.
        pending_data = {
            "name": name,
            "student_id": student_id,
            "embeddings": serialize_embeddings(encs),
        }
        pending_file = dataset_path / "pending_enrollment.json"
        pending_file.write_text(_json.dumps(pending_data), encoding="utf-8")

        return {
            "status": "verify",
            "message": f"Embeddings built ({len(encs)} images). Verification needed.",
            "student_id": student_id,
        }, 200

    except Exception:
        return {"status": "error", "message": "Enrollment failed due to an internal error."}, 500


@app.route("/verify_enrollment", methods=["POST"])
def verify_enrollment():
    """
    Verification step: the client sends a single photo + student_id.
    We detect the face, compare against the pending embeddings stored on disk.
    If match → save to DB. If not → return fail.
    """
    import json as _json

    u = current_user()
    if not u or u.get("role") != "student":
        return {"status": "error", "message": "Unauthorized"}, 401

    data = request.get_json(silent=True) or {}
    img_data = data.get("img")
    sid = (data.get("student_id") or "").strip()

    if not img_data:
        return {"status": "error", "message": "No image provided."}, 400

    # Resolve student_id from request or session.
    if not sid:
        sid = (
            (session.get("pending_student_id") or "").strip()
            or (u.get("student_id") or "").strip()
            or (u.get("username") or "").strip()
        )

    if not sid:
        return {"status": "error", "message": "Student ID missing."}, 400

    # Load pending enrollment data from disk.
    pending_file = Path("dataset") / sid / "pending_enrollment.json"
    if not pending_file.exists():
        return {"status": "error", "message": "No pending enrollment data. Please re-enroll."}, 400

    try:
        pending_data = _json.loads(pending_file.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "error", "message": "Corrupted enrollment data. Please re-enroll."}, 400

    pending_name = pending_data.get("name", sid)
    pending_emb_json = pending_data.get("embeddings", "[]")

    img_bytes = decode_data_url_image(img_data)
    if not img_bytes:
        return {"status": "error", "message": "Invalid image data."}, 400

    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "error", "message": "Failed to decode image."}, 400

        _, encs = detect_and_encode(frame)

        if not encs:
            return {"status": "fail", "message": "No face detected in verification photo. Try again."}, 200

        # Compare the verification face against the pending embeddings.
        known_encs = deserialize_embeddings(pending_emb_json)
        labels = [pending_name] * len(known_encs)

        label, score = best_match(known_encs, labels, encs[0], min_similarity=0.45)

        if str(label).lower() == "unknown" or float(score) < 0.45:
            return {
                "status": "fail",
                "message": f"Face not recognized (score {score:.2f}). Try again or re-enroll with clearer images.",
            }, 200

        # Verification passed — save to DB now.
        add_user(pending_name, sid, sid, pending_emb_json)
        link_account_student_id(u["username"], sid)
        set_user_session_fields(sid, pending_name)

        # Clean up temp file and session data.
        pending_file.unlink(missing_ok=True)
        session.pop("pending_name", None)
        session.pop("pending_student_id", None)
        session.pop("user", None)

        return {
            "status": "success",
            "message": f"Identity verified (score {score:.2f}). Enrollment complete.",
        }, 200

    except Exception:
        return {"status": "error", "message": "Verification failed due to an internal error."}, 500


# -----------------------------------------------------------------------------
# Student: Cancel / cleanup pending enrollment
# -----------------------------------------------------------------------------

@app.route("/cancel_enrollment", methods=["POST"])
def cancel_enrollment():
    """
    Clean up all pending enrollment data:
    - Delete pending_enrollment.json
    - Delete captured images (img_*.jpg)
    - Remove the dataset folder if empty
    - Remove from DB if somehow saved
    """
    u = current_user()
    if not u or u.get("role") != "student":
        return {"status": "error", "message": "Unauthorized"}, 401

    data = request.get_json(silent=True) or {}
    sid = (data.get("student_id") or "").strip()

    if not sid:
        sid = (
            (session.get("pending_student_id") or "").strip()
            or (u.get("student_id") or "").strip()
            or (u.get("username") or "").strip()
        )

    if not sid:
        return {"status": "ok"}, 200

    dataset_path = Path("dataset") / sid

    if dataset_path.exists():
        # Delete pending enrollment file
        for f in ("pending_enrollment.json", "pending_embeddings.json"):
            (dataset_path / f).unlink(missing_ok=True)

        # Delete all captured images
        for img in dataset_path.glob("img_*.jpg"):
            img.unlink(missing_ok=True)

        # Remove folder if empty
        try:
            dataset_path.rmdir()
        except OSError:
            pass

    # Remove from DB if somehow saved (edge case)
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE student_id=?", (sid,))

    return {"status": "ok"}, 200


# -----------------------------------------------------------------------------
# Teacher: Users management + camera enrollment for students
# -----------------------------------------------------------------------------


@app.route("/check_student_id", methods=["POST"])
def check_student_id():
    data = request.get_json(silent=True) or {}
    student_id = (data.get("student_id") or "").strip()

    if not student_id:
        return {"exists": False}, 200

    with get_conn() as conn:
        cur = conn.execute("SELECT 1 FROM users WHERE student_id=? LIMIT 1", (student_id,))
        row = cur.fetchone()

    return {"exists": row is not None}


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    data = request.get_json(silent=True) or {}
    student_id = (data.get("student_id") or "").strip()
    img_data = data.get("img")
    index = safe_int(data.get("index"), 0)

    if not student_id or not img_data:
        return {"status": "error", "message": "Invalid payload."}, 400

    user = current_user()
    role = user.get("role")

    # Students can only upload frames for their own pending student_id.
    if role == "student":
        allowed_sid = (session.get("pending_student_id") or "").strip() or (user.get("username") or "").strip()
        if not allowed_sid or student_id != allowed_sid:
            return {"status": "error", "message": "Unauthorized upload target."}, 403

    img_bytes = decode_data_url_image(img_data)
    if not img_bytes:
        return {"status": "error", "message": "Invalid image data."}, 400

    save_path = Path("dataset") / student_id
    save_path.mkdir(parents=True, exist_ok=True)

    # Avoid negative / weird indices.
    index = max(0, min(index, 9999))
    filename = save_path / f"img_{index:02d}.jpg"
    filename.write_bytes(img_bytes)

    return {"status": "success", "message": f"saved {filename.name}"}, 200


@app.route("/finalize_user", methods=["POST"])
def finalize_user():
    if not teacher_only():
        abort(403)

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    student_id = (data.get("student_id") or "").strip()

    if not name or not student_id:
        return {"status": "error", "message": "Missing required fields."}, 400

    with get_conn() as conn:
        cur = conn.execute("SELECT 1 FROM users WHERE student_id=? LIMIT 1", (student_id,))
        if cur.fetchone():
            return {"status": "error", "message": "This Student ID is already enrolled."}, 400

    dataset_path = Path("dataset") / student_id
    if not dataset_path.exists() or not any(dataset_path.glob("*.jpg")):
        return {"status": "error", "message": f"No images found for folder: {student_id}."}, 400

    try:
        encs = build_embeddings_for_label(student_id)
        if not encs:
            return {"status": "error", "message": "No faces were detected in the captured images."}, 400

        add_user(name, student_id, student_id, serialize_embeddings(encs))
        return {"status": "success", "message": f"Student enrolled: {name} ({len(encs)} images)."}, 200

    except Exception:
        return {"status": "error", "message": "Enrollment failed due to an internal error."}, 500


@app.route("/delete_user/<int:user_id>", methods=["POST"])
def delete_user_route(user_id: int):
    if not teacher_only():
        abort(403)

    try:
        with get_conn() as conn:
            row = conn.execute("SELECT name, label FROM users WHERE id=?", (user_id,)).fetchone()
            if not row:
                flash("flash.student_not_found", "warning")
                return redirect(url_for("users_add"))

            user_name, user_label = row
            conn.execute("DELETE FROM logs WHERE user_id=?", (user_id,))

        delete_user(user_id)

        user_dir = Path("dataset") / user_label
        if user_dir.exists():
            rmtree(user_dir)

        flash("flash.student_deleted", "success")

    except Exception:
        flash("flash.delete_student_error", "danger")

    return redirect(url_for("users_add"))


# -----------------------------------------------------------------------------
# Teacher: Courses page + Join Requests
# -----------------------------------------------------------------------------

@app.route("/teacher/courses", methods=["GET", "POST"])
def teacher_courses():
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        section = request.form.get("section", "").strip()

        if not name or not section:
            flash("flash.teacher_courses_missing_fields", "warning")
            return redirect(url_for("teacher_courses"))

        create_course(name, section, teacher_id)
        flash("flash.teacher_course_created", "success")
        return redirect(url_for("teacher_courses"))

    courses = get_teacher_courses(teacher_id)
    requests_rows = get_teacher_course_requests(teacher_id, status="pending")

    return render_template("teacher_courses.html", courses=courses, requests_rows=requests_rows)


@app.route("/teacher/request/<int:request_id>/accept", methods=["POST"])
def teacher_accept_request(request_id: int):
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    ok = accept_course_request(request_id, teacher_id)
    flash(
        "flash.request_accepted" if ok else "flash.request_not_found",
        "success" if ok else "warning",
    )
    return redirect(url_for("teacher_courses"))


@app.route("/teacher/request/<int:request_id>/reject", methods=["POST"])
def teacher_reject_request(request_id: int):
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    ok = reject_course_request(request_id, teacher_id)
    flash("flash.request_deleted" if ok else "flash.request_not_found", "success" if ok else "warning")
    return redirect(url_for("teacher_courses"))


@app.route("/teacher/course/delete/<int:course_id>", methods=["POST"])
def delete_course_route(course_id: int):
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    delete_course(course_id, teacher_id)
    flash("flash.course_deleted", "success")
    return redirect(url_for("teacher_courses"))


# -----------------------------------------------------------------------------
# Teacher: Course Students Management Page + APIs
# -----------------------------------------------------------------------------

@app.route("/teacher/course_students")
def course_students_page():
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    courses = get_teacher_courses(teacher_id)
    return render_template("course_students.html", courses=courses)


@app.route("/teacher/my_courses_json")
def my_courses_json():
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, name, section
            FROM courses
            WHERE teacher_id=?
            ORDER BY id DESC
            """,
            (teacher_id,),
        ).fetchall()

    return jsonify([dict(r) for r in rows])


@app.route("/api/course/<int:course_id>/students")
def api_course_students(course_id: int):
    if not teacher_only():
        abort(403)
    teacher_id = current_user()["id"]
    return jsonify(get_course_students(course_id, teacher_id))


@app.route("/api/course/<int:course_id>/available_students")
def api_course_available_students(course_id: int):
    if not teacher_only():
        abort(403)
    teacher_id = current_user()["id"]
    return jsonify(get_students_not_in_course(course_id, teacher_id))


@app.route("/api/course/<int:course_id>/add_student", methods=["POST"])
def api_course_add_student(course_id: int):
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    data = request.get_json(silent=True) or {}
    student_account_id = safe_int(data.get("student_account_id"), 0)

    ok = add_student_to_course(course_id, student_account_id, teacher_id)
    return jsonify({"status": "ok" if ok else "forbidden"}), (200 if ok else 403)


@app.route("/api/course/<int:course_id>/remove_student", methods=["POST"])
def api_course_remove_student(course_id: int):
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    data = request.get_json(silent=True) or {}
    student_account_id = safe_int(data.get("student_account_id"), 0)

    ok = remove_student_from_course(course_id, student_account_id, teacher_id)
    return jsonify({"status": "ok" if ok else "forbidden"}), (200 if ok else 403)


# -----------------------------------------------------------------------------
# Student: List Courses + request/join state
# -----------------------------------------------------------------------------

@app.route("/student/courses")
def student_courses():
    student_account_id = current_user()["id"]

    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        courses = conn.execute(
            """
            SELECT
              c.id,
              c.name AS course_name,
              c.section,
              t.username AS teacher_name,

              CASE WHEN cs.student_id IS NOT NULL THEN 1 ELSE 0 END AS is_joined,
              CASE WHEN cr.status = 'pending' THEN 'pending' ELSE NULL END AS request_status

            FROM courses c
            JOIN accounts t ON t.id = c.teacher_id

            LEFT JOIN course_students cs
              ON cs.course_id = c.id AND cs.student_id = ?

            LEFT JOIN course_requests cr
              ON cr.course_id = c.id AND cr.student_id = ? AND cr.status='pending'

            ORDER BY c.id DESC
            """,
            (student_account_id, student_account_id),
        ).fetchall()

    return render_template("student_courses.html", courses=courses)


@app.route("/student/request/<int:course_id>")
def request_course(course_id: int):
    student_account_id = current_user()["id"]

    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM course_students WHERE course_id=? AND student_id=?",
            (course_id, student_account_id),
        ).fetchone()
        if row:
            flash("flash.already_joined_course", "success")
            return redirect(url_for("student_courses"))

        row2 = conn.execute(
            "SELECT 1 FROM course_requests WHERE course_id=? AND student_id=? AND status='pending'",
            (course_id, student_account_id),
        ).fetchone()
        if row2:
            flash("flash.pending_request_exists", "warning")
            return redirect(url_for("student_courses"))

    try:
        create_course_request(course_id, student_account_id)
        flash("flash.join_request_sent", "info")
    except Exception:
        flash("flash.join_request_error", "danger")

    return redirect(url_for("student_courses"))


@app.route("/student/leave/<int:course_id>", methods=["POST"])
def leave_course(course_id: int):
    student_id = current_user()["id"]

    with get_conn() as conn:
        conn.execute("DELETE FROM course_students WHERE course_id=? AND student_id=?", (course_id, student_id))
        conn.execute("DELETE FROM course_requests WHERE course_id=? AND student_id=?", (course_id, student_id))

    flash("flash.left_course", "info")
    return redirect(url_for("student_courses"))


# -----------------------------------------------------------------------------
# Live recognition
# -----------------------------------------------------------------------------

@app.route("/recognize_live")
def recognize_live():
    return render_template("recognize_live.html")


@app.route("/recognize_frame", methods=["POST"])
def recognize_frame():
    if not teacher_only():
        return jsonify({"status": "forbidden"}), 403

    data = request.get_json(silent=True) or {}
    img_b64 = data.get("img")
    mode = data.get("mode", "attendance")
    course_id = safe_int(data.get("course_id"), 0)

    if not img_b64:
        return jsonify({"status": "error", "message": "No image provided."}), 400

    if course_id <= 0:
        return jsonify({"status": "error", "message": "اختر مادة أولاً."}), 400

    teacher_id = current_user()["id"]

    with _frame_lock:
        _latest_payload["img_b64"] = img_b64
        _latest_payload["mode"] = mode
        _latest_payload["course_id"] = course_id
        _latest_payload["teacher_id"] = teacher_id

    # Return the last known result immediately; client can poll.
    return jsonify(_latest_result)


# -----------------------------------------------------------------------------
# Attendance + Logs
# -----------------------------------------------------------------------------

@app.route("/attendance")
def attendance():
    teacher_id = current_user()["id"]
    selected_date = request.args.get("date") or date.today().isoformat()

    subject_id = request.args.get("subject_id")
    subject_id_int = int(subject_id) if subject_id and subject_id.isdigit() else None

    subjects = get_teacher_courses(teacher_id)

    rows_raw = []
    if subject_id_int is not None:
        rows_raw = get_attendance_for_date(selected_date, subject_id_int)

    rows = []
    for name, first_seen, last_seen in rows_raw:
        try:
            dt1 = datetime.fromisoformat(first_seen)
            dt2 = datetime.fromisoformat(last_seen)
            first_time = dt1.strftime("%H:%M:%S")
            last_time = dt2.strftime("%H:%M:%S")
            duration_minutes = int((dt2 - dt1).total_seconds() // 60)
        except Exception:
            first_time = first_seen
            last_time = last_seen
            duration_minutes = 0

        user = get_user_by_name(name)
        student_id_text = user["student_id"] if user else "-"

        rows.append(
            {
                "name": name,
                "student_id": student_id_text,
                "first_seen": first_time,
                "last_seen": last_time,
                "duration_minutes": duration_minutes,
            }
        )

    return render_template(
        "attendance.html",
        rows=rows,
        subjects=subjects,
        subject_id=subject_id_int,
        date=selected_date,
        selected_date=selected_date,
    )


@app.route("/logs")
def logs():
    logs_data = get_logs(limit=1000)
    return render_template("logs.html", logs=logs_data)


@app.route("/clear_logs", methods=["POST"])
def clear_logs_route():
    if not teacher_only():
        abort(403)

    from database import clear_logs

    try:
        clear_logs()
        flash("flash.logs_cleared", "success")
    except Exception:
        flash("flash.logs_clear_failed", "danger")

    return redirect(url_for("logs"))


@app.route("/log_action", methods=["POST"])
def log_action():
    if not teacher_only():
        abort(403)

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    action = (data.get("action") or "").strip()

    if not name or action not in {"login", "logout"}:
        return jsonify({"status": "error", "message": "Invalid payload."}), 400

    last_action = get_last_action_for_user(name)
    if last_action == "login" and action == "login":
        return jsonify({"status": "ignored", "message": "User is already logged in."})
    if last_action == "logout" and action == "logout":
        return jsonify({"status": "ignored", "message": "User is already logged out."})

    user_row = get_user_by_name(name)
    user_id = user_row["id"] if user_row else None
    student_id_text = user_row["student_id"] if user_row else None

    add_log(user_id, name, action, 0.0, "", student_id=student_id_text)

    now_str = datetime.now().strftime("%d/%m/%Y %I:%M:%S %p")
    return jsonify({"status": "success", "action": action, "name": name, "time": now_str})


# -----------------------------------------------------------------------------
# Student: My attendance history
# -----------------------------------------------------------------------------

@app.route("/my_attendance_history")
def my_attendance_history():
    u = current_user()
    if not u or u.get("role") != "student":
        flash("flash.students_only", "danger")
        return redirect(url_for("index"))

    student_id_text = u.get("student_id")
    if not student_id_text:
        flash("flash.no_student_id_linked", "warning")
        return redirect(url_for("index"))

    account_id = u.get("id")
    course_id_int = safe_int(request.args.get("course_id"), 0) or None

    courses = get_student_courses(account_id)
    rows = []

    if course_id_int is not None:
        with get_conn() as conn:
            cur = conn.execute(
                """
                SELECT date, first_seen, last_seen
                FROM attendance
                WHERE student_id = ? AND subject_id = ?
                ORDER BY date DESC, first_seen DESC
                """,
                (student_id_text, course_id_int),
            )
            rows_raw = cur.fetchall()

        for (date_str, first_seen, last_seen) in rows_raw:
            try:
                dt1 = datetime.fromisoformat(first_seen)
                dt2 = datetime.fromisoformat(last_seen)
                duration_minutes = int((dt2 - dt1).total_seconds() // 60)
                first_seen_fmt = dt1.strftime("%H:%M:%S")
                last_seen_fmt = dt2.strftime("%H:%M:%S")
            except Exception:
                duration_minutes = 0
                first_seen_fmt = first_seen
                last_seen_fmt = last_seen

            rows.append(
                {
                    "date": date_str,
                    "first_seen": first_seen_fmt,
                    "last_seen": last_seen_fmt,
                    "duration_minutes": duration_minutes,
                }
            )

    return render_template(
        "my_attendance_history.html",
        rows=rows,
        courses=courses,
        course_id=course_id_int,
    )


# -----------------------------------------------------------------------------
# Cheat images (one image per row)
# -----------------------------------------------------------------------------

CHEAT_ROOT = Path(app.root_path) / "cheat_videos"
CHEAT_PENDING_DIR = CHEAT_ROOT / "pending"
CHEAT_POSITIVE_DIR = CHEAT_ROOT / "positive"
CHEAT_NORMAL_DIR = CHEAT_ROOT / "normal"

CHEAT_ROOT.mkdir(exist_ok=True)
CHEAT_PENDING_DIR.mkdir(exist_ok=True)
CHEAT_POSITIVE_DIR.mkdir(exist_ok=True)
CHEAT_NORMAL_DIR.mkdir(exist_ok=True)


def is_ajax_request() -> bool:
    """Detect if request expects JSON."""
    return (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.accept_mimetypes.best == "application/json"
        or "application/json" in (request.headers.get("Accept", "") or "")
    )


def allowed_video_extension(filename: str) -> bool:
    ext = Path(filename or "").suffix.lower()
    return ext in {".webm", ".mp4", ".mov", ".mkv"}


def build_cheat_filename(course_id: int, teacher_id: int, suffix: str = ".webm") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"teacher_{teacher_id}_course_{course_id}_{ts}_{uuid.uuid4().hex[:8]}{suffix}"


@app.route("/upload_cheat_video", methods=["POST"])
def upload_cheat_video():
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]

    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No video uploaded."}), 400

    if not allowed_video_extension(file.filename):
        return jsonify({"status": "error", "message": "Unsupported video format."}), 400

    course_id = safe_int(request.form.get("course_id"), 0)
    camera_id = (request.form.get("camera_id") or "").strip()
    message = (request.form.get("message") or "20-second cheat clip").strip()

    if course_id <= 0:
        return jsonify({"status": "error", "message": "Invalid course_id."}), 400

    with get_conn() as conn:
        own = conn.execute(
            "SELECT 1 FROM courses WHERE id=? AND teacher_id=?",
            (course_id, teacher_id),
        ).fetchone()

    if not own:
        return jsonify({"status": "error", "message": "Invalid course for this teacher."}), 403

    ext = Path(file.filename).suffix.lower() or ".webm"
    filename = build_cheat_filename(course_id, teacher_id, suffix=ext)

    pending_path = CHEAT_PENDING_DIR / filename
    file.save(str(pending_path))

    try:
        pred = predict_cheat_video(str(pending_path))
    except Exception as e:
        print("Hybrid inference error:", repr(e))
        if pending_path.exists():
            pending_path.unlink(missing_ok=True)
        return jsonify({
            "status": "error",
            "message": "Video model inference failed.",
        }), 500

    result = str(pred.get("result") or "normal").strip().lower()
    predicted_label = str(pred.get("predicted_label") or "normal").strip().lower()
    confidence = float(pred.get("predicted_confidence") or 0.0)
    normal_confidence = float(pred.get("normal_confidence") or 0.0)
    top3 = pred.get("top3") or []
    source = pred.get("source") or ""
    reason = pred.get("reason") or ""

    if result != "cheat":
        if pending_path.exists():
            pending_path.unlink(missing_ok=True)

        return jsonify({
            "status": "ok",
            "result": "normal",
            "predicted_label": predicted_label,
            "confidence": confidence,
            "normal_confidence": normal_confidence,
            "top3": top3,
            "source": source,
            "reason": reason,
        }), 200

    positive_path = CHEAT_POSITIVE_DIR / filename
    pending_path.replace(positive_path)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO cheat_clips (
                camera_id, student_id, name, message,
                video_path, predicted_label, confidence, course_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera_id,
                "",
                "",
                message,
                filename,
                predicted_label,
                confidence,
                course_id,
            ),
        )

    return jsonify({
        "status": "ok",
        "result": "cheat",
        "predicted_label": predicted_label,
        "confidence": confidence,
        "normal_confidence": normal_confidence,
        "top3": top3,
        "source": source,
        "reason": reason,
        "file": filename,
    }), 200


@app.route("/cheat_video/<path:filename>")
def serve_cheat_video(filename: str):
    if not teacher_only():
        abort(403)
    return send_from_directory(str(CHEAT_POSITIVE_DIR), filename)


@app.route("/cheat_review")
def cheat_review():
    if not teacher_only():
        flash("flash.not_allowed_teacher", "danger")
        return redirect(url_for("index"))

    teacher_id = current_user()["id"]
    course_id_filter = safe_int(request.args.get("course_id"), 0) or None

    # Get teacher's courses for the dropdown.
    courses = get_teacher_courses(teacher_id)

    rows = []
    if course_id_filter:
        with get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                cc.id,
                cc.camera_id,
                cc.student_id,
                cc.name,
                cc.message,
                cc.video_path,
                cc.predicted_label,
                cc.confidence,
                cc.course_id,
                cc.created_at,
                cc.status,
                cc.reviewed_at,
                c.name AS course_name,
                c.section AS course_section
                FROM cheat_clips cc
                JOIN courses c ON c.id = cc.course_id
                WHERE c.teacher_id = ? AND cc.course_id = ?
                ORDER BY cc.id DESC
                """,
                (teacher_id, course_id_filter),
            ).fetchall()

    return render_template(
        "cheat_review.html",
        rows=rows,
        courses=courses,
        selected_course_id=course_id_filter,
    )

@app.route("/cheat_clip/<int:clip_id>/approve", methods=["POST"])
def approve_cheat_clip(clip_id: int):
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE cheat_clips
            SET status='approved', reviewed_by=?, reviewed_at=datetime('now')
            WHERE id=?
            """,
            (teacher_id, clip_id),
        )

    if is_ajax_request():
        return jsonify(
            status="ok",
            new_status="approved",
            delete_url=url_for("delete_cheat_clip", clip_id=clip_id),
        )

    flash("flash.case_approved", "success")
    return redirect(url_for("cheat_review"))


@app.route("/cheat_clip/<int:clip_id>/reject", methods=["POST"])
def reject_cheat_clip(clip_id: int):
    if not teacher_only():
        abort(403)

    teacher_id = current_user()["id"]
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE cheat_clips
            SET status='rejected', reviewed_by=?, reviewed_at=datetime('now')
            WHERE id=?
            """,
            (teacher_id, clip_id),
        )

    if is_ajax_request():
        return jsonify(
            status="ok",
            new_status="rejected",
            delete_url=url_for("delete_cheat_clip", clip_id=clip_id),
        )

    flash("flash.case_rejected", "warning")
    return redirect(url_for("cheat_review"))


@app.route("/cheat_clip/<int:clip_id>/delete", methods=["POST"])
def delete_cheat_clip(clip_id: int):
    if not teacher_only():
        abort(403)

    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT video_path FROM cheat_clips WHERE id=?", (clip_id,)).fetchone()
        if not row:
            if is_ajax_request():
                return jsonify(status="error", message="Case not found"), 404
            flash("flash.case_not_found", "warning")
            return redirect(url_for("cheat_review"))

        filename = row["video_path"]
        conn.execute("DELETE FROM cheat_clips WHERE id=?", (clip_id,))

    p = CHEAT_POSITIVE_DIR / filename
    if p.exists():
        p.unlink()

    if is_ajax_request():
        return jsonify(status="ok")

    flash("flash.case_deleted", "success")
    return redirect(url_for("cheat_review"))


# -----------------------------------------------------------------------------
# Admin: Dashboard + Management APIs
# -----------------------------------------------------------------------------

@app.route("/dashboard")
def admin_dashboard():
    if not admin_only():
        abort(403)
    return render_template("admin_dashboard.html")


@app.route("/dashboard/list_accounts")
def admin_list_accounts():
    """Return all accounts as JSON for the admin panel."""
    if not admin_only():
        abort(403)

    with get_conn(row_factory=True) as conn:
        rows = conn.execute("""
            SELECT a.username, COALESCE(u.name, a.display_name) AS student_name, a.role
            FROM accounts a
            LEFT JOIN users u ON a.student_id = u.student_id
            ORDER BY
              CASE a.role
                WHEN 'admin'   THEN 1
                WHEN 'teacher' THEN 2
                WHEN 'student' THEN 3
              END,
              a.username
        """).fetchall()

    return jsonify([dict(r) for r in rows])


@app.route("/dashboard/create_teacher", methods=["POST"])
def admin_create_teacher():
    """Create a new teacher account."""
    if not admin_only():
        abort(403)

    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password are required."}), 400

    if get_account_by_username(username):
        return jsonify({"status": "error", "message": f"Username already exists: {username}"}), 400

    password_hash = generate_password_hash(password)
    create_account(username=username, password_hash=password_hash, role="teacher", student_id=None)

    return jsonify({"status": "success", "message": f"Teacher '{username}' created successfully."}), 200


@app.route("/dashboard/delete_account", methods=["POST"])
def admin_delete_account():
    """Delete an account and all related data."""
    if not admin_only():
        abort(403)

    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()

    if not username:
        return jsonify({"status": "error", "message": "Username is required."}), 400

    acc = get_account_by_username(username)
    if not acc:
        return jsonify({"status": "error", "message": f"Account not found: {username}"}), 404

    role = (acc.get("role") or "").lower()
    account_id = int(acc.get("id") or 0)
    sid_text = (acc.get("student_id") or "").strip() or (acc.get("username") or "").strip()

    deleted_user_row = False
    deleted_dataset = False
    label = sid_text

    with get_conn() as conn:
        if role == "student":
            row = conn.execute(
                "SELECT id, label FROM users WHERE student_id=? LIMIT 1",
                (sid_text,),
            ).fetchone()

            user_id = int(row[0]) if row else 0
            label = (row[1] if row and row[1] else sid_text).strip()

            if user_id:
                conn.execute("DELETE FROM logs WHERE user_id=?", (user_id,))

            conn.execute("DELETE FROM attendance WHERE student_id=?", (sid_text,))

            if account_id:
                conn.execute("DELETE FROM course_students WHERE student_id=?", (account_id,))
                conn.execute("DELETE FROM course_requests WHERE student_id=?", (account_id,))

            if user_id:
                conn.execute("DELETE FROM users WHERE id=?", (user_id,))
                deleted_user_row = True

        conn.execute("DELETE FROM accounts WHERE username=?", (username,))

    # Delete dataset folder for students.
    if role == "student" and label:
        dataset_path = Path("dataset") / label
        if dataset_path.exists() and dataset_path.is_dir():
            rmtree(dataset_path)
            deleted_dataset = True

    msg = f"Account '{username}' deleted successfully."
    if role == "student":
        msg += f" User row: {'yes' if deleted_user_row else 'no'}. Dataset: {'yes' if deleted_dataset else 'no'}."

    return jsonify({"status": "success", "message": msg}), 200


@app.route("/dashboard/reset_password", methods=["POST"])
def admin_reset_password():
    """Reset a user's password."""
    if not admin_only():
        abort(403)

    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    new_password = (data.get("new_password") or "").strip()

    if not username or not new_password:
        return jsonify({"status": "error", "message": "Username and new password are required."}), 400

    acc = get_account_by_username(username)
    if not acc:
        return jsonify({"status": "error", "message": f"Account not found: {username}"}), 404

    password_hash = generate_password_hash(new_password)
    with get_conn() as conn:
        conn.execute(
            "UPDATE accounts SET password_hash=? WHERE username=?",
            (password_hash, username),
        )

    return jsonify({"status": "success", "message": f"Password for '{username}' has been reset."}), 200


@app.route("/dashboard/reset_face", methods=["POST"])
def admin_reset_face():
    """Reset a student's face data so they must re-enroll."""
    if not admin_only():
        abort(403)

    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()

    if not username:
        return jsonify({"status": "error", "message": "Username is required."}), 400

    acc = get_account_by_username(username)
    if not acc:
        return jsonify({"status": "error", "message": f"Account not found: {username}"}), 404

    role = (acc.get("role") or "").lower()
    if role != "student":
        return jsonify({"status": "error", "message": f"'{username}' is not a student account."}), 400

    sid_text = (acc.get("student_id") or "").strip()

    deleted_user_row = False
    deleted_dataset = False
    label = sid_text

    with get_conn() as conn:
        if sid_text:
            # Temporarily disable FK so ON DELETE SET NULL doesn't clear
            # accounts.student_id when we remove the users row.
            conn.execute("PRAGMA foreign_keys = OFF")

            row = conn.execute(
                "SELECT id, label FROM users WHERE student_id=? LIMIT 1",
                (sid_text,),
            ).fetchone()

            if row:
                user_id = int(row[0])
                label = (row[1] if row[1] else sid_text).strip()
                conn.execute("DELETE FROM users WHERE id=?", (user_id,))
                deleted_user_row = True

            conn.execute("PRAGMA foreign_keys = ON")

        # student_id stays on the account — enrollment check uses the users table,
        # so the student will still be forced to re-enroll.

    # Delete dataset folder.
    if label:
        dataset_path = Path("dataset") / label
        if dataset_path.exists() and dataset_path.is_dir():
            rmtree(dataset_path)
            deleted_dataset = True

    msg = f"Face data for '{username}' has been reset."
    msg += f" User row deleted: {'yes' if deleted_user_row else 'no'}."
    msg += f" Dataset deleted: {'yes' if deleted_dataset else 'no'}."

    return jsonify({"status": "success", "message": msg}), 200



if __name__ == "__main__":
    # Debug should be disabled in production.
    app.run(debug=True)
