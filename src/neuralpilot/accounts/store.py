"""Users, login sessions and run ownership (SQLite, separate from experiments).

Ownership lives here rather than on the experiment `runs` table because a run
is owned from the moment it is created, while its experiment row is only
written when it finishes. Guidance never reads this file.

Each call opens its own connection, so the store is safe to share between
FastAPI's worker threads and the background run threads.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, NamedTuple

from .. import config
from .passwords import hash_password, verify_password

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 256  # bounds the scrypt work a single request can cause
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_owners (
    run_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    dataset_name TEXT NOT NULL,
    goal TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dataset_path TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS run_owners_by_user ON run_owners(user_id, created_at);
"""


# Columns added after the first release. `CREATE TABLE IF NOT EXISTS` leaves an
# existing table alone, so they are added here instead.
_ADDED_COLUMNS = (
    # Where the upload was saved, so a failed run can be retried from the same file.
    ("run_owners", "dataset_path", "TEXT NOT NULL DEFAULT ''"),
)


class AccountError(ValueError):
    """A sign-up that cannot be accepted. The message is safe to show the user."""


class User(NamedTuple):
    id: str
    email: str


class OwnedRun(NamedTuple):
    run_id: str
    dataset_name: str
    goal: str
    created_at: str
    dataset_path: str = ""  # empty for runs recorded before the column existed


def _owned(row: sqlite3.Row) -> OwnedRun:
    return OwnedRun(row["run_id"], row["dataset_name"], row["goal"], row["created_at"], row["dataset_path"])


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, column, declaration in _ADDED_COLUMNS:
        present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in present:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _hash_token(token: str) -> str:
    # Sessions are stored hashed, so a leaked database cannot be used to log in.
    return hashlib.sha256(token.encode()).hexdigest()


class AccountStore:
    def __init__(self, db_path: Path | str = config.ACCOUNTS_DB_PATH):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            _add_missing_columns(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- users ---------------------------------------------------------------

    def create_user(self, email: str, password: str) -> User:
        email = normalize_email(email)
        if not _EMAIL_RE.match(email):
            raise AccountError("Enter a valid email address.")
        if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
            raise AccountError(f"Password must be between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters.")

        user = User(id=secrets.token_hex(16), email=email)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (user.id, user.email, hash_password(password), _now().isoformat()),
                )
        except sqlite3.IntegrityError as exc:
            raise AccountError("An account with this email already exists.") from exc
        return user

    def authenticate(self, email: str, password: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ?", (normalize_email(email),)
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return User(id=row["id"], email=row["email"])

    # --- sessions ------------------------------------------------------------

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = _now() + timedelta(seconds=config.SESSION_TTL_S)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (_hash_token(token), user_id, expires_at.isoformat()),
            )
        return token

    def user_for_session(self, token: str | None) -> User | None:
        if not token:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT users.id, users.email, sessions.expires_at
                FROM sessions JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ?
                """,
                (_hash_token(token),),
            ).fetchone()
        if row is None or datetime.fromisoformat(row["expires_at"]) <= _now():
            return None
        return User(id=row["id"], email=row["email"])

    def end_session(self, token: str | None) -> None:
        if not token:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))

    # --- run ownership -------------------------------------------------------

    def record_run(
        self, run_id: str, user_id: str, dataset_name: str, goal: str, dataset_path: str = ""
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO run_owners (run_id, user_id, dataset_name, goal, created_at, dataset_path)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, user_id, dataset_name, goal, _now().isoformat(), dataset_path),
            )

    def owned_run(self, run_id: str, user_id: str) -> OwnedRun | None:
        """The run, only if `user_id` owns it. Anyone else gets None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, dataset_name, goal, created_at, dataset_path"
                " FROM run_owners WHERE run_id = ? AND user_id = ?",
                (run_id, user_id),
            ).fetchone()
        return _owned(row) if row else None

    def runs_for_user(self, user_id: str) -> list[OwnedRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, dataset_name, goal, created_at, dataset_path"
                " FROM run_owners WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [_owned(r) for r in rows]
