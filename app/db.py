"""
SQLite storage. Deliberately plain sqlite3 (no ORM, no separate DB server) —
this is a single-user tool meant to run on the cheapest possible box, and a
DB server is one more process than it needs.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',   -- draft -> queued -> processing -> done -> error
    error TEXT,
    glossary_json TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    order_index INTEGER NOT NULL,
    filename TEXT NOT NULL,
    orig_path TEXT NOT NULL,
    rendered_path TEXT,
    width INTEGER,
    height INTEGER
);

-- One row per text block a real trained detector (see detection.py) found
-- on a page, after merging nearby raw detections (columns/lines of one
-- bubble) into one box — rendering fills that box with a flat color and
-- draws the translation into it (render.py); no bubble-shape
-- classification, no inpainting: see detection.py's module docstring for
-- why. x/y/w/h is the detector's own box; the manual_* columns are a
-- user's optional override of the block actually drawn (see
-- app/main.py's edit_line) — null means "use the automatic layout".
CREATE TABLE IF NOT EXISTS lines (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    order_index INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    w INTEGER NOT NULL,
    h INTEGER NOT NULL,
    fill_color TEXT NOT NULL,   -- json [r,g,b] -- the flat block drawn over this line
    text_color TEXT NOT NULL,   -- json [r,g,b]
    source_text TEXT,
    translated_text TEXT,
    low_confidence INTEGER NOT NULL DEFAULT 0,
    manual_x INTEGER,           -- override for the rendered block's position/size —
    manual_y INTEGER,           -- null in each means "use the automatic layout"
    manual_w INTEGER,
    manual_h INTEGER,
    manual_font_size INTEGER    -- override for the rendered font size — null means "auto-fit"
);

CREATE INDEX IF NOT EXISTS idx_pages_batch ON pages(batch_id, order_index);
CREATE INDEX IF NOT EXISTS idx_lines_page ON lines(page_id, order_index);
"""

# Columns added after the `lines` table's first release — CREATE TABLE IF
# NOT EXISTS above only applies to a brand new database, so an existing
# database file (from before one of these was added) needs each missing
# column added by hand. SQLite has no "ADD COLUMN IF NOT EXISTS", hence
# checking PRAGMA table_info first.
_LINES_MIGRATIONS = [
    ("manual_x", "INTEGER"),
    ("manual_y", "INTEGER"),
    ("manual_w", "INTEGER"),
    ("manual_h", "INTEGER"),
    ("manual_font_size", "INTEGER"),
]


def _migrate_lines_table(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(lines)")}
    for name, col_type in _LINES_MIGRATIONS:
        if name not in existing:
            conn.execute(f"ALTER TABLE lines ADD COLUMN {name} {col_type}")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    settings.ensure_dirs()
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate_lines_table(conn)


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def create_batch(target_lang: str) -> str:
    batch_id = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO batches (id, created_at, target_lang, status) VALUES (?, ?, ?, 'draft')",
            (batch_id, now_iso(), target_lang),
        )
    return batch_id


def get_batch(batch_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()


def list_batches() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()


def set_batch_status(batch_id: str, status: str, error: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE batches SET status = ?, error = ? WHERE id = ?",
            (status, error, batch_id),
        )


def try_start_batch(batch_id: str) -> bool:
    """Atomically flip a batch to 'queued', but only from 'draft' or
    'error' — and report whether it actually did.

    This has to be one statement, not "read the status, then decide, then
    write it" (which is what the /translate route used to do): two
    overlapping requests for the same batch id (a double-click, a retried
    request) can both read 'draft' before either has written anything,
    both pass the check, and both enqueue a real processing run. The
    single background worker then runs the same batch through detection
    and translation twice, back to back, and since line rows are inserted
    fresh each run rather than replacing what's there (see worker.py), the
    second run's translations don't overwrite the first's — they render on
    top of them. Doing the check and the write as one UPDATE closes that
    window: only whichever request's UPDATE actually changes a row gets to
    proceed.
    """
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE batches SET status = 'queued' WHERE id = ? AND status IN ('draft', 'error')",
            (batch_id,),
        )
        return cur.rowcount > 0


def set_batch_glossary(batch_id: str, glossary: list[dict[str, Any]]) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE batches SET glossary_json = ? WHERE id = ?",
            (json.dumps(glossary), batch_id),
        )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def add_page(batch_id: str, order_index: int, filename: str, orig_path: str,
             width: int, height: int) -> str:
    page_id = new_id()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO pages (id, batch_id, order_index, filename, orig_path, width, height)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (page_id, batch_id, order_index, filename, orig_path, width, height),
        )
    return page_id


def get_page(page_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()


def list_pages(batch_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM pages WHERE batch_id = ? ORDER BY order_index", (batch_id,)
        ).fetchall()


def reorder_pages(batch_id: str, page_ids_in_order: list[str]) -> None:
    with get_conn() as conn:
        for idx, page_id in enumerate(page_ids_in_order):
            conn.execute(
                "UPDATE pages SET order_index = ? WHERE id = ? AND batch_id = ?",
                (idx, page_id, batch_id),
            )


def set_page_rendered(page_id: str, rendered_path: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE pages SET rendered_path = ? WHERE id = ?", (rendered_path, page_id))


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

def clear_lines(page_id: str) -> None:
    """Delete any lines already stored for a page before detecting fresh
    ones — a (re)processing pass is a redo, not an append. Belt-and-
    suspenders alongside `try_start_batch`'s concurrency guard: that closes
    off a batch being processed twice at once, this makes reprocessing a
    single page idempotent regardless of how it got triggered again."""
    with get_conn() as conn:
        conn.execute("DELETE FROM lines WHERE page_id = ?", (page_id,))


def add_line(page_id: str, order_index: int, bbox: tuple[int, int, int, int],
             fill_color: tuple[int, int, int], text_color: tuple[int, int, int],
             source_text: str) -> str:
    line_id = new_id()
    x, y, w, h = bbox
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO lines (id, page_id, order_index, x, y, w, h, fill_color,
                                   text_color, source_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (line_id, page_id, order_index, x, y, w, h,
             json.dumps(fill_color), json.dumps(text_color), source_text),
        )
    return line_id


def get_line(line_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM lines WHERE id = ?", (line_id,)).fetchone()


def list_lines(page_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM lines WHERE page_id = ? ORDER BY order_index", (page_id,)
        ).fetchall()


def set_line_translation(line_id: str, translated_text: str, low_confidence: bool = False) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE lines SET translated_text = ?, low_confidence = ? WHERE id = ?",
            (translated_text, 1 if low_confidence else 0, line_id),
        )


def set_line_style(
    line_id: str,
    fill_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] | None = None,
    manual_box: tuple[int, int, int, int] | None = None,
    manual_font_size: int | None = None,
) -> None:
    """Overwrite a block's rendering style. `fill_color`/`text_color`
    replace the auto-sampled ones outright (there's no "auto" to fall back
    to for these — they're always some concrete color). `manual_box`
    (x, y, w, h) and `manual_font_size` replace the automatic layout when
    given, and — this is the whole point of them being nullable columns —
    passing None for either explicitly resets that one back to "use the
    automatic layout" rather than leaving a stale override in place.
    """
    with get_conn() as conn:
        current = conn.execute(
            "SELECT fill_color, text_color FROM lines WHERE id = ?", (line_id,)
        ).fetchone()
        fill_json = json.dumps(fill_color) if fill_color is not None else current["fill_color"]
        text_json = json.dumps(text_color) if text_color is not None else current["text_color"]
        mx, my, mw, mh = manual_box if manual_box is not None else (None, None, None, None)
        conn.execute(
            """UPDATE lines SET fill_color = ?, text_color = ?,
                                 manual_x = ?, manual_y = ?, manual_w = ?, manual_h = ?,
                                 manual_font_size = ?
               WHERE id = ?""",
            (fill_json, text_json, mx, my, mw, mh, manual_font_size, line_id),
        )
