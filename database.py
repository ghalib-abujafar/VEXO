from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

DB_PATH = Path(__file__).with_name("app.db")


@contextmanager
def get_conn(row_factory: bool = False) -> Iterable[sqlite3.Connection]:
    """
    SQLite connection helper.

    - Enables foreign key constraints on every connection.
    - Uses WAL mode for better concurrency (especially with threads).
    - Commits only if no exception happened.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    if row_factory:
        conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """
    Create tables from schema.sql, then apply small migrations for older DBs.
    """
    schema_path = Path(__file__).with_name("schema.sql")
    with get_conn() as conn, open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    # Migration: ensure attendance.subject_id exists (legacy DBs).
    try:
        with get_conn() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(attendance)").fetchall()]
            if "subject_id" not in cols:
                conn.execute("ALTER TABLE attendance ADD COLUMN subject_id INTEGER")
    except sqlite3.OperationalError:
        # If attendance table doesn't exist yet (very early init), ignore.
        pass

    # Migration: ensure accounts.role CHECK constraint includes 'admin'.
    # SQLite cannot ALTER CHECK constraints, so we rebuild the table if needed.
    try:
        with get_conn() as conn:
            # Check if the current schema already allows 'admin' role.
            schema_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='accounts'"
            ).fetchone()
            if schema_row and schema_row[0] and "'admin'" not in schema_row[0]:
                conn.executescript("""
                    PRAGMA foreign_keys = OFF;

                    CREATE TABLE IF NOT EXISTS accounts_new (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password_hash TEXT NOT NULL,
                      role TEXT NOT NULL CHECK(role IN ('teacher','student','admin')),
                      student_id TEXT UNIQUE,
                      FOREIGN KEY(student_id) REFERENCES users(student_id) ON DELETE SET NULL
                    );

                    INSERT INTO accounts_new (id, username, password_hash, role, student_id)
                    SELECT id, username, password_hash, role, student_id FROM accounts;

                    DROP TABLE accounts;

                    ALTER TABLE accounts_new RENAME TO accounts;

                    PRAGMA foreign_keys = ON;
                """)
    except sqlite3.OperationalError:
        pass

    # Migration: ensure accounts.display_name exists.
    try:
        with get_conn() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(accounts)").fetchall()]
            if "display_name" not in cols:
                conn.execute("ALTER TABLE accounts ADD COLUMN display_name TEXT")
                # Backfill from users.name for existing students.
                conn.execute("""
                    UPDATE accounts
                    SET display_name = (
                        SELECT u.name FROM users u WHERE u.student_id = accounts.student_id
                    )
                    WHERE role = 'student'
                      AND student_id IS NOT NULL
                      AND display_name IS NULL
                """)
    except sqlite3.OperationalError:
        pass


# =============================================================================
# Users
# =============================================================================

def add_user(name: str, student_id: str, label: str, embeddings_json: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (name, student_id, label, embeddings) VALUES (?, ?, ?, ?)",
            (name, student_id, label, embeddings_json),
        )


def get_users() -> list[dict[str, Any]]:
    with get_conn(row_factory=True) as conn:
        rows = conn.execute(
            "SELECT id, name, student_id, label, embeddings FROM users ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_user(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))


def get_user_by_name(name: str) -> dict[str, Any] | None:
    with get_conn(row_factory=True) as conn:
        row = conn.execute(
            "SELECT id, name, student_id, label FROM users WHERE name=?",
            (name,),
        ).fetchone()
        return dict(row) if row else None


# =============================================================================
# Logs
# =============================================================================

def add_log(
    user_id: int | None,
    name: str,
    result: str,
    score: float,
    image_path: str,
    student_id: str | None = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO logs (user_id, name, result, score, image_path, student_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, name, result, float(score), image_path, student_id),
        )


def get_logs(limit: int = 200):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT id, timestamp, name, result, score, image_path
            FROM logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()


def clear_logs() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM logs")


def get_last_action_for_user(name: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT result FROM logs WHERE name=? ORDER BY id DESC LIMIT 1",
            (name,),
        ).fetchone()
        return row[0] if row else None


# =============================================================================
# Attendance (supports subject_id = courses.id)
# =============================================================================

def ping_attendance(
    user_id: int,
    name: str,
    student_id: str,
    date_str: str,
    now_iso: str,
    subject_id: int | None = None,
) -> None:
    """
    Create or update attendance record for:
      (user_id + date + subject_id)

    If subject_id is None, we store NULL in subject_id and also match on NULL.
    """
    with get_conn() as conn:
        if subject_id is None:
            row = conn.execute(
                """
                SELECT id FROM attendance
                WHERE user_id=? AND date=? AND subject_id IS NULL
                """,
                (user_id, date_str),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id FROM attendance
                WHERE user_id=? AND date=? AND subject_id=?
                """,
                (user_id, date_str, subject_id),
            ).fetchone()

        if row:
            conn.execute(
                "UPDATE attendance SET last_seen=? WHERE id=?",
                (now_iso, row[0]),
            )
        else:
            conn.execute(
                """
                INSERT INTO attendance (user_id, name, student_id, date, first_seen, last_seen, subject_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, name, student_id, date_str, now_iso, now_iso, subject_id),
            )


def get_attendance_for_date(date_str: str, subject_id: int | None = None):
    """
    Returns rows: (name, first_seen, last_seen)
    Filtered by subject_id if provided.
    """
    with get_conn() as conn:
        if subject_id is None:
            return conn.execute(
                """
                SELECT name, first_seen, last_seen
                FROM attendance
                WHERE date=?
                ORDER BY name
                """,
                (date_str,),
            ).fetchall()

        return conn.execute(
            """
            SELECT name, first_seen, last_seen
            FROM attendance
            WHERE date=? AND subject_id=?
            ORDER BY name
            """,
            (date_str, subject_id),
        ).fetchall()


# =============================================================================
# Accounts
# =============================================================================

def create_account(
    username: str,
    password_hash: str,
    role: str,
    student_id: str | None = None,
    display_name: str | None = None,
) -> None:
    """
    Create an account. student_id may be NULL (especially before face enrollment).
    display_name stores the student's real name independent of face enrollment.
    """
    username = username.strip()
    role = role.strip()

    sid = (student_id or "").strip() or None
    dname = (display_name or "").strip() or None
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO accounts (username, password_hash, role, student_id, display_name) VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, role, sid, dname),
        )


def get_account_by_username(username: str) -> dict[str, Any] | None:
    with get_conn(row_factory=True) as conn:
        row = conn.execute(
            """
            SELECT id, username, password_hash, role, student_id, display_name
            FROM accounts
            WHERE username=?
            """,
            (username,),
        ).fetchone()
        return dict(row) if row else None


def get_account_by_id(account_id: int) -> dict[str, Any] | None:
    with get_conn(row_factory=True) as conn:
        row = conn.execute(
            "SELECT id, username, role, student_id, display_name FROM accounts WHERE id=?",
            (account_id,),
        ).fetchone()
        return dict(row) if row else None


def link_account_student_id(username: str, student_id: str) -> None:
    """
    Link an existing account to a student_id (after successful face enrollment).
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE accounts SET student_id=? WHERE username=?",
            (student_id.strip(), username.strip()),
        )


# =============================================================================
# Courses (Subjects)
# =============================================================================

def create_course(name: str, section: str, teacher_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO courses (name, section, teacher_id) VALUES (?, ?, ?)",
            (name.strip(), section.strip(), teacher_id),
        )


def delete_course(course_id: int, teacher_id: int) -> None:
    """Delete a course owned by this teacher only."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM courses WHERE id=? AND teacher_id=?",
            (course_id, teacher_id),
        )


def get_course(course_id: int) -> dict[str, Any] | None:
    with get_conn(row_factory=True) as conn:
        row = conn.execute(
            "SELECT id, name, section, teacher_id, created_at FROM courses WHERE id=?",
            (course_id,),
        ).fetchone()
        return dict(row) if row else None


def get_teacher_courses(teacher_id: int) -> list[dict[str, Any]]:
    with get_conn(row_factory=True) as conn:
        rows = conn.execute(
            """
            SELECT id, name, section
            FROM courses
            WHERE teacher_id=?
            ORDER BY id DESC
            """,
            (teacher_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_student_courses(account_id: int) -> list[dict[str, Any]]:
    with get_conn(row_factory=True) as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.name, c.section
            FROM course_students cs
            JOIN courses c ON c.id = cs.course_id
            WHERE cs.student_id = ?
            ORDER BY c.created_at DESC
            """,
            (account_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# =============================================================================
# Course Requests (Join Requests)
# =============================================================================

def create_course_request(course_id: int, student_account_id: int) -> None:
    """
    Student requests to join a course.
    Stored as pending by default (schema).
    """
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO course_requests (course_id, student_id, status)
            VALUES (?, ?, 'pending')
            """,
            (course_id, student_account_id),
        )


def get_teacher_course_requests(teacher_id: int, status: str = "pending") -> list[dict[str, Any]]:
    """
    Returns join requests for teacher's courses only.
    Includes student info from accounts + users by matching accounts.student_id to users.student_id.
    """
    with get_conn(row_factory=True) as conn:
        rows = conn.execute(
            """
            SELECT
              cr.id AS request_id,
              cr.course_id,
              c.name AS course_name,
              c.section,
              cr.status,
              cr.requested_at,
              a.id AS student_account_id,
              a.username AS student_username,
              a.student_id AS student_id_text,
              u.name AS student_name,
              u.label AS student_label
            FROM course_requests cr
            JOIN courses c ON c.id = cr.course_id
            JOIN accounts a ON a.id = cr.student_id
            LEFT JOIN users u ON u.student_id = a.student_id
            WHERE c.teacher_id = ? AND cr.status = ?
            ORDER BY cr.id DESC
            """,
            (teacher_id, status),
        ).fetchall()
        return [dict(r) for r in rows]


def accept_course_request(request_id: int, teacher_id: int) -> bool:
    """
    Accept:
    - ensure request belongs to this teacher
    - add to course_students
    - delete request so it disappears immediately
    """
    with get_conn(row_factory=True) as conn:
        row = conn.execute(
            """
            SELECT cr.id, cr.course_id, cr.student_id
            FROM course_requests cr
            JOIN courses c ON c.id = cr.course_id
            WHERE cr.id=? AND c.teacher_id=? AND cr.status='pending'
            """,
            (request_id, teacher_id),
        ).fetchone()

        if not row:
            return False

        conn.execute(
            "INSERT OR IGNORE INTO course_students (course_id, student_id) VALUES (?, ?)",
            (row["course_id"], row["student_id"]),
        )
        conn.execute("DELETE FROM course_requests WHERE id=?", (request_id,))
        return True


def reject_course_request(request_id: int, teacher_id: int) -> bool:
    """
    Reject:
    - ensure request belongs to this teacher
    - delete request (you can switch to status='rejected' if you want archiving)
    """
    with get_conn(row_factory=True) as conn:
        row = conn.execute(
            """
            SELECT cr.id
            FROM course_requests cr
            JOIN courses c ON c.id = cr.course_id
            WHERE cr.id=? AND c.teacher_id=? AND cr.status='pending'
            """,
            (request_id, teacher_id),
        ).fetchone()

        if not row:
            return False

        conn.execute("DELETE FROM course_requests WHERE id=?", (request_id,))
        return True


# =============================================================================
# Course Students (Enrollment)
# =============================================================================

def get_course_students(course_id: int, teacher_id: int | None = None) -> list[dict[str, Any]]:
    """
    Returns students in a course.
    If teacher_id provided, ensures teacher owns the course.
    """
    with get_conn(row_factory=True) as conn:
        if teacher_id is None:
            rows = conn.execute(
                """
                SELECT
                  a.id AS student_account_id,
                  a.username AS student_username,
                  a.student_id AS student_id_text,
                  COALESCE(u.name, a.display_name) AS name,
                  u.label AS label,
                  a.display_name AS display_name
                FROM course_students cs
                JOIN accounts a ON a.id = cs.student_id
                LEFT JOIN users u ON u.student_id = a.student_id
                WHERE cs.course_id = ?
                ORDER BY COALESCE(u.name, a.display_name, a.username)
                """,
                (course_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                  a.id AS student_account_id,
                  a.username AS student_username,
                  a.student_id AS student_id_text,
                  COALESCE(u.name, a.display_name) AS name,
                  u.label AS label,
                  a.display_name AS display_name
                FROM course_students cs
                JOIN courses c ON c.id = cs.course_id
                JOIN accounts a ON a.id = cs.student_id
                LEFT JOIN users u ON u.student_id = a.student_id
                WHERE cs.course_id = ? AND c.teacher_id = ?
                ORDER BY COALESCE(u.name, a.display_name, a.username)
                """,
                (course_id, teacher_id),
            ).fetchall()

        return [dict(r) for r in rows]


def get_students_not_in_course(course_id: int, teacher_id: int) -> list[dict[str, Any]]:
    """
    For "Add Student" button: return student accounts not in this course.
    Restricted to teacher-owned course.
    """
    with get_conn(row_factory=True) as conn:
        rows = conn.execute(
            """
            SELECT
              a.id AS student_account_id,
              a.username AS student_username,
              a.student_id AS student_id_text,
              COALESCE(u.name, a.display_name) AS name,
              u.label AS label,
              a.display_name AS display_name
            FROM accounts a
            LEFT JOIN users u ON u.student_id = a.student_id
            JOIN courses c ON c.id = ?
            WHERE
              c.teacher_id = ?
              AND a.role = 'student'
              AND a.id NOT IN (
                SELECT student_id FROM course_students WHERE course_id = ?
              )
            ORDER BY COALESCE(u.name, a.display_name, a.username)
            """,
            (course_id, teacher_id, course_id),
        ).fetchall()

        return [dict(r) for r in rows]


def add_student_to_course(course_id: int, student_account_id: int, teacher_id: int) -> bool:
    """Teacher manually adds a student to a course (only if owns course)."""
    with get_conn() as conn:
        own = conn.execute(
            "SELECT 1 FROM courses WHERE id=? AND teacher_id=?",
            (course_id, teacher_id),
        ).fetchone()
        if not own:
            return False

        conn.execute(
            "INSERT OR IGNORE INTO course_students (course_id, student_id) VALUES (?, ?)",
            (course_id, student_account_id),
        )
        return True


def remove_student_from_course(course_id: int, student_account_id: int, teacher_id: int) -> bool:
    """Remove student from course (teacher must own course)."""
    with get_conn() as conn:
        own = conn.execute(
            "SELECT 1 FROM courses WHERE id=? AND teacher_id=?",
            (course_id, teacher_id),
        ).fetchone()
        if not own:
            return False

        conn.execute(
            "DELETE FROM course_students WHERE course_id=? AND student_id=?",
            (course_id, student_account_id),
        )
        conn.execute(
            "DELETE FROM course_requests WHERE course_id=? AND student_id=?",
            (course_id, student_account_id),
        )
        return True


# =============================================================================
# Student attendance helper
# =============================================================================

def get_student_attendance_for_date(student_id_text: str, date_str: str, course_id: int):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT first_seen, last_seen
            FROM attendance
            WHERE student_id=? AND date=? AND subject_id=?
            ORDER BY first_seen ASC
            """,
            (student_id_text, date_str, course_id),
        ).fetchall()