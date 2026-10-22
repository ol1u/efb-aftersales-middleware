# coding=utf-8
"""Archive storage for the after-sales formatting pipeline.

Every group message becomes a record: the raw text, the AI verdict
(problem / reply / none), the formatted rendering, and — for replies —
which problem record it answers. These records are the training material
this middleware exists to collect. Formatted texts the middleware posted
itself are stored as kind ``posted`` so their relayed copies are never
reprocessed.
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    sender_id TEXT,
    sender_name TEXT,
    kind TEXT NOT NULL,
    raw_text TEXT,
    formatted_text TEXT,
    target_id INTEGER,
    efb_msg_id TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_group ON records (group_id, created_at);
CREATE INDEX IF NOT EXISTS idx_records_kind ON records (group_id, kind, created_at);

CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    opener_id TEXT,
    opener_name TEXT,
    summary TEXT,
    plan TEXT,
    version INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_group ON threads (group_id, status, updated_at);

CREATE TABLE IF NOT EXISTS kb_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_key TEXT,
    sender_name TEXT,
    sent_at INTEGER,
    text TEXT NOT NULL,
    batch TEXT,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

KINDS = ("message", "posted", "problem", "reply", "none")


def _normalize_text(text: Optional[str]) -> str:
    return " ".join((text or "").split())


class Database:
    def __init__(self, path: Union[str, Path], clock: Callable[[], float] = time.time):
        self.clock = clock
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            self._migrate()
            self.conn.commit()

    def _migrate(self) -> None:
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(records)")}
        if "thread_id" not in columns:
            self.conn.execute("ALTER TABLE records ADD COLUMN thread_id INTEGER")

    def now(self) -> int:
        return int(self.clock())

    def close(self):
        with self.lock:
            self.conn.close()

    # -- records ---------------------------------------------------------

    def add_record(self, *, group_id: str, sender_id: Optional[str], sender_name: Optional[str],
                   kind: str, raw_text: str, formatted_text: Optional[str] = None,
                   target_id: Optional[int] = None, efb_msg_id: Optional[str] = None,
                   thread_id: Optional[int] = None) -> sqlite3.Row:
        if kind not in KINDS:
            raise ValueError("unknown record kind: {}".format(kind))
        with self.lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO records (group_id, sender_id, sender_name, kind, raw_text, formatted_text,"
                " target_id, efb_msg_id, thread_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (group_id, sender_id, sender_name, kind, raw_text, formatted_text, target_id,
                 efb_msg_id, thread_id, self.now()))
            record_id = cur.lastrowid
        return self.get_record(record_id)

    def set_record_thread(self, record_id: int, thread_id: int) -> None:
        with self.lock, self.conn:
            self.conn.execute("UPDATE records SET thread_id = ? WHERE id = ?", (thread_id, record_id))

    def get_record(self, record_id: int) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()

    def list_records(self, group_id: Optional[str] = None, kind: Optional[str] = None,
                     limit: int = 100) -> List[sqlite3.Row]:
        sql = "SELECT * FROM records"
        params: List[Any] = []
        where = []
        if group_id is not None:
            where.append("group_id = ?")
            params.append(group_id)
        if kind is not None:
            where.append("kind = ?")
            params.append(kind)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    # -- threads -----------------------------------------------------------

    def create_thread(self, *, group_id: str, opener_id: Optional[str], opener_name: Optional[str],
                      summary: Optional[str], plan: Optional[str]) -> sqlite3.Row:
        now = self.now()
        with self.lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO threads (group_id, opener_id, opener_name, summary, plan, version, status,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, 'open', ?, ?)",
                (group_id, opener_id, opener_name, summary, plan, now, now))
            thread_id = cur.lastrowid
        return self.get_thread(thread_id)

    def get_thread(self, thread_id: int) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()

    def update_thread(self, thread_id: int, summary: Optional[str], plan: Optional[str]) -> sqlite3.Row:
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE threads SET summary = ?, plan = ?, version = version + 1, updated_at = ?"
                " WHERE id = ?", (summary, plan, self.now(), thread_id))
        return self.get_thread(thread_id)

    def touch_thread(self, thread_id: int) -> None:
        with self.lock, self.conn:
            self.conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?",
                              (self.now(), thread_id))

    def recent_threads(self, group_id: str, within_seconds: int, limit: int = 8) -> List[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(
                "SELECT * FROM threads WHERE group_id = ? AND status = 'open' AND updated_at >= ?"
                " ORDER BY updated_at DESC LIMIT ?",
                (group_id, self.now() - within_seconds, limit)).fetchall()

    def thread_opening(self, thread_id: int) -> Optional[sqlite3.Row]:
        """The first archived group message of a thread (what gets quoted)."""
        with self.lock:
            return self.conn.execute(
                "SELECT * FROM records WHERE thread_id = ? AND kind = 'message'"
                " ORDER BY id ASC LIMIT 1", (thread_id,)).fetchone()

    def thread_messages(self, thread_id: int, limit: int = 6) -> List[sqlite3.Row]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM records WHERE thread_id = ? AND kind = 'message'"
                " ORDER BY id DESC LIMIT ?", (thread_id, limit)).fetchall()
        return list(reversed(rows))

    # -- knowledge base ------------------------------------------------------

    def kb_add_messages(self, batch: str, messages: List[Dict[str, Any]]) -> int:
        now = self.now()
        with self.lock, self.conn:
            self.conn.executemany(
                "INSERT INTO kb_messages (chat_key, sender_name, sent_at, text, batch, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(m.get("chat_key"), m.get("sender_name"), m.get("sent_at"), m["text"], batch, now)
                 for m in messages])
        return len(messages)

    def kb_stats(self) -> sqlite3.Row:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) AS n, MIN(sent_at) AS lo, MAX(sent_at) AS hi"
                                     " FROM kb_messages").fetchone()

    def kb_candidates(self, terms: List[str], per_term_limit: int = 150) -> List[sqlite3.Row]:
        seen: Dict[int, sqlite3.Row] = {}
        with self.lock:
            for term in terms:
                if len(term) < 2:
                    continue
                rows = self.conn.execute(
                    "SELECT * FROM kb_messages WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
                    ("%{}%".format(term), per_term_limit)).fetchall()
                for row in rows:
                    seen[row["id"]] = row
            if not seen:
                rows = self.conn.execute(
                    "SELECT * FROM kb_messages ORDER BY id DESC LIMIT 300").fetchall()
                for row in rows:
                    seen[row["id"]] = row
        return list(seen.values())

    def kb_window(self, chat_key: Optional[str], center_id: int, before: int = 3,
                  after: int = 3) -> List[sqlite3.Row]:
        with self.lock:
            prev_rows = self.conn.execute(
                "SELECT * FROM kb_messages WHERE chat_key IS ? AND id < ? ORDER BY id DESC LIMIT ?",
                (chat_key, center_id, before)).fetchall()
            next_rows = self.conn.execute(
                "SELECT * FROM kb_messages WHERE chat_key IS ? AND id >= ? ORDER BY id ASC LIMIT ?",
                (chat_key, center_id, after + 1)).fetchall()
        return list(reversed(prev_rows)) + list(next_rows)

    # -- settings ------------------------------------------------------------

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.lock:
            row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else default

    def set_setting(self, key: str, value: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)"
                              " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    # -- tracing helpers ---------------------------------------------------

    def find_record_by_text(self, group_id: str, quoted_text: str) -> Optional[sqlite3.Row]:
        """Find the archived record a native quote points at.

        Compares whitespace-normalised text over recent group records,
        newest first; tolerates quotes that carry the full original or a
        truncated form of it.
        """
        target = _normalize_text(quoted_text)
        for suffix in ("…", "..."):
            if target.endswith(suffix):
                target = target[: -len(suffix)].rstrip()
                break
        if len(target) < 4:
            return None
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM records WHERE group_id = ? AND raw_text IS NOT NULL"
                " ORDER BY id DESC LIMIT 500", (group_id,)).fetchall()
        for row in rows:
            stored = _normalize_text(row["raw_text"])
            if not stored:
                continue
            if (stored == target or stored.startswith(target) or target.startswith(stored)
                    or (len(target) >= 10 and target in stored)):
                return row
        return None

    def find_copy_source(self, group_id: str, text: str) -> Optional[Tuple[sqlite3.Row, str]]:
        """Detect the copy-and-append pattern.

        If ``text`` starts with a verbatim copy of an earlier record of
        this group, return (that record, the appended delta). The longest
        matching source wins.
        """
        incoming = _normalize_text(text)
        if len(incoming) < 12:
            return None
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM records WHERE group_id = ? AND raw_text IS NOT NULL"
                " ORDER BY id DESC LIMIT 300", (group_id,)).fetchall()
        best_row, best_len = None, 0
        for row in rows:
            stored = _normalize_text(row["raw_text"])
            if len(stored) < 10 or len(stored) <= best_len:
                continue
            if incoming.startswith(stored) and len(incoming) - len(stored) >= 2:
                best_row, best_len = row, len(stored)
        if best_row is None:
            return None
        delta = incoming[best_len:].lstrip(" \t:：,，;；")
        return (best_row, delta) if delta else None

    def is_posted(self, group_id: str, text: str) -> bool:
        """True when ``text`` is one of our own formatted posts coming back."""
        target = _normalize_text(text)
        if not target:
            return False
        with self.lock:
            rows = self.conn.execute(
                "SELECT raw_text FROM records WHERE group_id = ? AND kind = 'posted'"
                " ORDER BY id DESC LIMIT 300", (group_id,)).fetchall()
        return any(_normalize_text(row["raw_text"]) == target for row in rows)
