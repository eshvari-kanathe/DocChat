"""
database.py — Persistent storage for DocChat.

Tables:
  users         — authentication (username, password_hash, salt)
  chats         — one row per conversation session (chat_id, username, title, timestamps)
  messages      — one row per message in a chat (message_id, chat_id, role, content, timestamp)

The old flat `chat_messages` table is still created for backward-compatibility
migration, but is no longer written to by the application.
"""
import sqlite3
import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.resolve() / "users.db"


# ─── Schema Init ──────────────────────────────────────────────────────────────

def init_db() -> None:
    """Initialize all database tables."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL
        )
    """)

    # Chats (one per conversation session)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id    TEXT PRIMARY KEY,
            username   TEXT NOT NULL,
            title      TEXT NOT NULL DEFAULT 'New Chat',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Messages (per chat)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id    TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
        )
    """)

    # Legacy table kept for backward-compat (not written to anymore)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ─── Password Helpers ─────────────────────────────────────────────────────────

def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


# ─── User Auth ────────────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    if not username or not password:
        return False, "Username and password cannot be empty."
    username = username.strip()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            return False, "Username already exists."
        salt = os.urandom(16)
        pw_hash = _hash_password(password, salt)
        cur.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, pw_hash, salt.hex()),
        )
        conn.commit()
        return True, "User registered successfully."
    except Exception as e:
        return False, f"Database error: {e}"
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> bool:
    """Return True if credentials are valid."""
    if not username or not password:
        return False
    username = username.strip()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return False
        stored_hash, salt_hex = row
        return _hash_password(password, bytes.fromhex(salt_hex)) == stored_hash
    except Exception:
        return False
    finally:
        conn.close()


# ─── Chat Session CRUD ────────────────────────────────────────────────────────

def create_chat(username: str, title: str = "New Chat") -> str:
    """Create a new chat session. Returns the new chat_id (UUID)."""
    chat_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO chats (chat_id, username, title) VALUES (?, ?, ?)",
            (chat_id, username.strip(), title),
        )
        conn.commit()
    finally:
        conn.close()
    return chat_id


def get_user_chats(username: str) -> list[dict]:
    """Return all chats for a user, newest first.
    Each dict: {chat_id, title, updated_at}
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = []
    try:
        cur.execute(
            "SELECT chat_id, title, updated_at FROM chats WHERE username = ? ORDER BY updated_at DESC",
            (username.strip(),),
        )
        rows = [{"chat_id": r[0], "title": r[1], "updated_at": r[2]} for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


def get_chat_messages(chat_id: str) -> list[dict]:
    """Return all messages for a chat in chronological order.
    Each dict: {role, content}
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    msgs = []
    try:
        cur.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY message_id ASC",
            (chat_id,),
        )
        msgs = [{"role": r[0], "content": r[1]} for r in cur.fetchall()]
    finally:
        conn.close()
    return msgs


def save_message(chat_id: str, role: str, content: str) -> None:
    """Save a message and bump the chat's updated_at timestamp."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content),
        )
        cur.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?",
            (chat_id,),
        )
        conn.commit()
    except Exception as e:
        print(f"Error saving message: {e}")
    finally:
        conn.close()


def update_chat_title(chat_id: str, title: str) -> None:
    """Set a chat's display title (used after first user message)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title, chat_id))
        conn.commit()
    except Exception as e:
        print(f"Error updating title: {e}")
    finally:
        conn.close()


def rename_chat(chat_id: str, new_title: str) -> None:
    """Rename a chat session."""
    update_chat_title(chat_id, new_title.strip() or "New Chat")


def delete_chat(chat_id: str) -> None:
    """Delete a chat and all its messages (CASCADE)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cur.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
        conn.commit()
    except Exception as e:
        print(f"Error deleting chat: {e}")
    finally:
        conn.close()


def clear_chat_messages(chat_id: str) -> None:
    """Delete all messages inside a chat but keep the chat row."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cur.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP, title = 'New Chat' WHERE chat_id = ?",
            (chat_id,),
        )
        conn.commit()
    except Exception as e:
        print(f"Error clearing messages: {e}")
    finally:
        conn.close()


# ─── Legacy Shims (kept so old call-sites don't crash) ────────────────────────

def save_chat_message(username: str, role: str, content: str) -> None:
    """Deprecated shim — use save_message(chat_id, ...) instead."""
    pass


def get_chat_history(username: str) -> list[dict]:
    """Deprecated shim — returns empty list."""
    return []


def clear_chat_history(username: str) -> None:
    """Deprecated shim — no-op."""
    pass


# ─── Bootstrap ────────────────────────────────────────────────────────────────
init_db()
