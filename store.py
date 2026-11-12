"""HCE persistence — SQLite store; all cross-pass state lives here (invariant 3).

One database file per Hermes home (``$HERMES_HOME/context-engine/context.db``), shared
by all sessions. Every pass runs inside a single ``BEGIN IMMEDIATE`` transaction so
byte changes and state writes commit atomically (design 5.2). Concurrent writers
(front pass vs hygiene compress) serialize on SQLite's write lock + busy_timeout.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tags (
  session_id  TEXT NOT NULL,
  tag_number  INTEGER NOT NULL,
  kind        TEXT NOT NULL,              -- 'tool' | 'file'
  carrier_id  TEXT NOT NULL,              -- identity of the message carrying the unit
  uid         TEXT,                       -- message_uid when available at tag time
  content_hash TEXT NOT NULL,             -- sha1 of canonical carrier content
  owner_id    TEXT,                       -- tool: assistant carrier identity
  call_id     TEXT,                       -- tool: provider tool-call id
  call_name   TEXT,                       -- tool: function name
  args_bytes  INTEGER,                    -- tool: utf8 bytes of string values in args
  part_index  INTEGER,                    -- file: index of the part within carrier content
  seq         INTEGER NOT NULL,           -- monotonic arrival order (per session)
  status      TEXT NOT NULL DEFAULT 'active',  -- active | dropped | compacted
  drop_form   TEXT,                       -- 'skeleton' | 'remove'
  token_count INTEGER NOT NULL DEFAULT 0,
  created_at  REAL NOT NULL,
  landed_at   REAL,
  PRIMARY KEY (session_id, tag_number)
);
CREATE INDEX IF NOT EXISTS idx_tags_session_seq ON tags (session_id, seq);
CREATE INDEX IF NOT EXISTS idx_tags_carrier ON tags (session_id, carrier_id);
CREATE TABLE IF NOT EXISTS pending_ops (
  session_id TEXT NOT NULL,
  tag_number INTEGER NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (session_id, tag_number)
);
CREATE TABLE IF NOT EXISTS source_contents (
  session_id TEXT NOT NULL,
  tag_number INTEGER NOT NULL,
  payload    TEXT NOT NULL,               -- JSON: original content snapshot
  stored_at  REAL NOT NULL,
  PRIMARY KEY (session_id, tag_number)
);
CREATE TABLE IF NOT EXISTS frozen_markers (
  session_id  TEXT NOT NULL,
  message_id  TEXT NOT NULL,              -- carrier identity of the user message
  marker_text TEXT NOT NULL,
  created_at  REAL NOT NULL,
  PRIMARY KEY (session_id, message_id)
);
CREATE TABLE IF NOT EXISTS frozen_nudges (
  session_id TEXT NOT NULL,
  message_id TEXT NOT NULL,               -- carrier identity that carries the reminder
  text       TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (session_id, message_id)
);
CREATE TABLE IF NOT EXISTS session_meta (
  session_id TEXT NOT NULL,
  key        TEXT NOT NULL,
  value      TEXT NOT NULL,               -- JSON-encoded
  PRIMARY KEY (session_id, key)
);
CREATE TABLE IF NOT EXISTS lkg (
  session_id TEXT PRIMARY KEY,
  payload    TEXT NOT NULL,               -- JSON: rendered history messages (head excluded)
  out_hash   TEXT NOT NULL,
  head_len   INTEGER NOT NULL,            -- leading non-history messages in the request
  n_history  INTEGER NOT NULL,            -- history messages covered by payload
  head_fps   TEXT NOT NULL,               -- JSON: fingerprints of head (request) messages
  base_fps   TEXT NOT NULL,               -- JSON: fingerprints of the unrendered history base
  model      TEXT,
  updated_at REAL NOT NULL
);
"""


class StoreError(Exception):
    """Base class for store failures."""


class SchemaFenceError(StoreError):
    """Database schema version incompatible with this code (design 9.17).

    Fail-frozen: the engine must refuse to manage and never silently degrade.
    """


class Store:
    """Thin sqlite wrapper. One connection per instance; lazily opened in the
    thread that first uses it (compress runs in a host worker thread)."""

    def __init__(self, path: str):
        self._path = path
        self._conn: Optional[sqlite3.Connection] = None
        self._open_lock = threading.Lock()
        # Eager connect: the schema fence fires at construction, not mid-pass —
        # callers can trust that a returned Store is usable.
        self._connect()

    # -- connection ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        with self._open_lock:
            if self._conn is not None:
                return self._conn
            conn = sqlite3.connect(self._path, timeout=5.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema(conn)
            self._conn = conn
            return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Fence on an incompatible database, then make sure every table exists.

        The DDL is idempotent (``IF NOT EXISTS``) and runs on EVERY open, so adding a
        table in a later release is a no-op upgrade for existing databases instead of a
        silent ``no such table`` at query time. Anything that changes an existing
        column's meaning must bump ``SCHEMA_VERSION`` instead — the fence refuses rather
        than migrate.
        """
        with self._transaction(conn):
            meta_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
            ).fetchone() is not None
            if meta_exists:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()
                if row is None:
                    raise SchemaFenceError("meta table exists but has no schema_version row")
                try:
                    db_version = int(row["value"])
                except (TypeError, ValueError):
                    raise SchemaFenceError(f"corrupt schema_version row: {row['value']!r}")
                if db_version != SCHEMA_VERSION:
                    raise SchemaFenceError(
                        f"context.db schema version {db_version} != supported {SCHEMA_VERSION}; "
                        "refusing to open (rollback = restore from backup, not downgrade)"
                    )
            for statement in _SCHEMA.strip().split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(statement)
            if not meta_exists:
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )

    @contextmanager
    def _transaction(self, conn: sqlite3.Connection):
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    @contextmanager
    def transaction(self):
        """Single write transaction spanning all lane decisions + render (design 5.2)."""
        conn = self._connect()
        with self._transaction(conn):
            yield conn

    def close(self) -> None:
        with self._open_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # -- tags ---------------------------------------------------------------

    def next_tag_number(self, conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(tag_number), 0) AS m FROM tags WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return int(row["m"]) + 1

    def insert_tag(self, conn: sqlite3.Connection, tag: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO tags (session_id, tag_number, kind, carrier_id, uid, content_hash,
                                 owner_id, call_id, call_name, args_bytes, part_index,
                                 seq, status, drop_form, token_count, created_at)
               VALUES (:session_id, :tag_number, :kind, :carrier_id, :uid, :content_hash,
                       :owner_id, :call_id, :call_name, :args_bytes, :part_index,
                       :seq, :status, :drop_form, :token_count, :created_at)""",
            tag,
        )

    def load_tags(self, conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM tags WHERE session_id=? AND status IN ('active','dropped') ORDER BY seq",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_tag_status(
        self, conn: sqlite3.Connection, session_id: str, tag_number: int,
        status: str, drop_form: Optional[str] = None,
    ) -> None:
        import time as _time

        conn.execute(
            "UPDATE tags SET status=?, drop_form=COALESCE(?, drop_form), landed_at=? "
            "WHERE session_id=? AND tag_number=?",
            (status, drop_form, _time.time() if status == "dropped" else None,
             session_id, tag_number),
        )

    def load_tags_all(self, conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
        """Every tag regardless of status (tool handlers need compacted/refused states)."""
        rows = conn.execute(
            "SELECT * FROM tags WHERE session_id=? ORDER BY seq", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def tag_status(self, session_id: str, tag_number: int) -> Optional[str]:
        conn = self._connect()
        row = conn.execute(
            "SELECT status FROM tags WHERE session_id=? AND tag_number=?",
            (session_id, tag_number),
        ).fetchone()
        return str(row["status"]) if row is not None else None

    def max_tag_number(self, session_id: str) -> int:
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(MAX(tag_number), 0) AS m FROM tags WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return int(row["m"])

    def rebind_tag_carrier(
        self, conn: sqlite3.Connection, session_id: str, tag_number: int,
        carrier_id: str, uid: Optional[str], content_hash: str,
    ) -> None:
        conn.execute(
            "UPDATE tags SET carrier_id=?, uid=?, content_hash=? "
            "WHERE session_id=? AND tag_number=?",
            (carrier_id, uid, content_hash, session_id, tag_number),
        )

    def all_tag_numbers(self, conn: sqlite3.Connection, session_id: str) -> list[int]:
        rows = conn.execute(
            "SELECT tag_number FROM tags WHERE session_id=?", (session_id,)
        ).fetchall()
        return [int(r["tag_number"]) for r in rows]

    # -- pending ops ----------------------------------------------------------

    def queue_drop(self, conn: sqlite3.Connection, session_id: str, tag_number: int,
                   drop_form: str, created_at: float) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO pending_ops (session_id, tag_number, created_at) VALUES (?,?,?)",
            (session_id, tag_number, created_at),
        )
        # Form is decided at queue time and never recomputed (design 9.1).
        conn.execute(
            "UPDATE tags SET drop_form=? WHERE session_id=? AND tag_number=?",
            (drop_form, session_id, tag_number),
        )

    def load_pending(self, conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            """SELECT p.tag_number, p.created_at, t.status, t.drop_form, t.seq
               FROM pending_ops p JOIN tags t
                 ON t.session_id = p.session_id AND t.tag_number = p.tag_number
               WHERE p.session_id=? ORDER BY t.seq""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def dequeue(self, conn: sqlite3.Connection, session_id: str, tag_number: int) -> None:
        conn.execute(
            "DELETE FROM pending_ops WHERE session_id=? AND tag_number=?",
            (session_id, tag_number),
        )

    def pending_token_sum(self, conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute(
            """SELECT COALESCE(SUM(t.token_count), 0) AS s FROM pending_ops p
               JOIN tags t ON t.session_id = p.session_id AND t.tag_number = p.tag_number
               WHERE p.session_id=?""",
            (session_id,),
        ).fetchone()
        return int(row["s"])

    def pending_exists(self, conn: sqlite3.Connection, session_id: str, tag_number: int) -> bool:
        row = conn.execute(
            "SELECT 1 FROM pending_ops WHERE session_id=? AND tag_number=?",
            (session_id, tag_number),
        ).fetchone()
        return row is not None

    def pending_token_sum_quick(self, session_id: str) -> int:
        conn = self._connect()
        return self.pending_token_sum(conn, session_id)

    # -- source contents ------------------------------------------------------

    def store_source(self, conn: sqlite3.Connection, session_id: str,
                     tag_number: int, payload: dict[str, Any]) -> None:
        import time as _time

        conn.execute(
            "INSERT OR REPLACE INTO source_contents (session_id, tag_number, payload, stored_at) "
            "VALUES (?,?,?,?)",
            (session_id, tag_number, json.dumps(payload, ensure_ascii=False), _time.time()),
        )

    def load_source(self, session_id: str, tag_number: int) -> Optional[dict[str, Any]]:
        conn = self._connect()
        row = conn.execute(
            "SELECT payload FROM source_contents WHERE session_id=? AND tag_number=?",
            (session_id, tag_number),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except (TypeError, ValueError):
            return None

    # -- frozen markers ---------------------------------------------------------

    def freeze_marker(self, conn: sqlite3.Connection, session_id: str,
                      message_id: str, marker_text: str) -> None:
        import time as _time

        conn.execute(
            "INSERT OR REPLACE INTO frozen_markers (session_id, message_id, marker_text, created_at) "
            "VALUES (?,?,?,?)",
            (session_id, message_id, marker_text, _time.time()),
        )

    def load_markers(self, conn: sqlite3.Connection, session_id: str) -> dict[str, str]:
        rows = conn.execute(
            "SELECT message_id, marker_text FROM frozen_markers WHERE session_id=?",
            (session_id,),
        ).fetchall()
        return {r["message_id"]: r["marker_text"] for r in rows}

    def drop_markers(self, conn: sqlite3.Connection, session_id: str,
                     message_ids: Iterable[str]) -> None:
        ids = list(message_ids)
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        conn.execute(
            f"DELETE FROM frozen_markers WHERE session_id=? AND message_id IN ({marks})",
            (session_id, *ids),
        )

    # -- frozen nudges ---------------------------------------------------------------

    def freeze_nudge(self, conn: sqlite3.Connection, session_id: str,
                     message_id: str, text: str) -> None:
        import time as _time

        conn.execute(
            "INSERT OR REPLACE INTO frozen_nudges (session_id, message_id, text, created_at) "
            "VALUES (?,?,?,?)",
            (session_id, message_id, text, _time.time()),
        )

    def load_nudges(self, conn: sqlite3.Connection, session_id: str) -> dict[str, str]:
        rows = conn.execute(
            "SELECT message_id, text FROM frozen_nudges WHERE session_id=?", (session_id,)
        ).fetchall()
        return {r["message_id"]: r["text"] for r in rows}

    def clear_nudges(self, conn: sqlite3.Connection, session_id: str) -> None:
        conn.execute("DELETE FROM frozen_nudges WHERE session_id=?", (session_id,))

    def drop_nudges(self, conn: sqlite3.Connection, session_id: str,
                    message_ids: Iterable[str]) -> None:
        ids = list(message_ids)
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        conn.execute(
            f"DELETE FROM frozen_nudges WHERE session_id=? AND message_id IN ({marks})",
            (session_id, *ids),
        )

    # -- session meta ------------------------------------------------------------

    def get_meta(self, session_id: str, key: str, default: Any = None) -> Any:
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM session_meta WHERE session_id=? AND key=?", (session_id, key)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError):
            return default

    def set_meta(self, conn: sqlite3.Connection, session_id: str, key: str, value: Any) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO session_meta (session_id, key, value) VALUES (?,?,?)",
            (session_id, key, json.dumps(value)),
        )

    def get_meta_in_tx(self, conn: sqlite3.Connection, session_id: str, key: str,
                       default: Any = None) -> Any:
        row = conn.execute(
            "SELECT value FROM session_meta WHERE session_id=? AND key=?", (session_id, key)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError):
            return default

    def write_meta(self, session_id: str, key: str, value: Any) -> None:
        """Convenience single-write meta update outside a pass transaction."""
        with self.transaction() as conn:
            self.set_meta(conn, session_id, key, value)

    # -- LKG ----------------------------------------------------------------------

    def load_lkg(self, session_id: str) -> Optional[dict[str, Any]]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM lkg WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row is not None else None

    def write_lkg(self, conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO lkg
                 (session_id, payload, out_hash, head_len, n_history, head_fps, base_fps,
                  model, updated_at)
               VALUES (:session_id, :payload, :out_hash, :head_len, :n_history, :head_fps,
                       :base_fps, :model, :updated_at)""",
            record,
        )

    # -- session lifecycle ----------------------------------------------------------

    def migrate_session(self, conn: sqlite3.Connection, old_sid: str, new_sid: str) -> None:
        """Rotation: carry every per-session row to the new session id (compress boundary)."""
        for table in ("tags", "pending_ops", "source_contents", "frozen_markers",
                      "session_meta", "lkg"):
            conn.execute(
                f"UPDATE OR REPLACE {table} SET session_id=? WHERE session_id=?", (new_sid, old_sid)
            )
        # UPDATE OR REPLACE can silently drop rows colliding with existing new_sid rows.
        # Sessions never legitimately share ids, so a collision means corruption; delete
        # stragglers loudly instead of failing the boundary notification.
        for table in ("tags", "pending_ops", "source_contents", "frozen_markers",
                      "session_meta", "lkg"):
            conn.execute(f"DELETE FROM {table} WHERE session_id=?", (old_sid,))
