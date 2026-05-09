"""SQLite schema + helpers for transcripts, speakers, and summaries."""

from __future__ import annotations

import datetime as dt
import pathlib
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterable, Iterator

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    camera TEXT NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    duration_s REAL NOT NULL,
    text TEXT NOT NULL,
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    speaker_locked INTEGER NOT NULL DEFAULT 0,
    chunk_file TEXT,
    chunk_offset_s REAL,
    embedding BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_segments_start ON segments(start_ts);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(speaker_id);
CREATE INDEX IF NOT EXISTS idx_segments_camera_date ON segments(camera, substr(start_ts, 1, 10));

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    camera TEXT NOT NULL,
    summary_date TEXT NOT NULL,
    segment_count INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(camera, summary_date)
);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    speaker_name UNINDEXED,
    camera UNINDEXED,
    start_ts UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS segments_fts_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text, speaker_name, camera, start_ts)
    VALUES (new.id, new.text,
            (SELECT name FROM speakers WHERE id = new.speaker_id),
            new.camera, new.start_ts);
END;

CREATE TRIGGER IF NOT EXISTS segments_fts_ad AFTER DELETE ON segments BEGIN
    DELETE FROM segments_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS segments_fts_au AFTER UPDATE ON segments BEGIN
    DELETE FROM segments_fts WHERE rowid = old.id;
    INSERT INTO segments_fts(rowid, text, speaker_name, camera, start_ts)
    VALUES (new.id, new.text,
            (SELECT name FROM speakers WHERE id = new.speaker_id),
            new.camera, new.start_ts);
END;
"""


class Database:
    """Thin wrapper around a shared sqlite3 connection.

    SQLite in WAL mode supports many readers + one writer concurrently; the
    writer lock here serialises our own writes across the transcriber worker,
    the summariser, and the FastAPI app which all share one Database instance.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=30.0,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(SCHEMA)
        self._backfill_fts()

    def _backfill_fts(self) -> None:
        """Populate segments_fts from existing segments if the FTS index is empty."""
        try:
            fts_count = self.conn.execute("SELECT COUNT(*) FROM segments_fts").fetchone()[0]
            seg_count = self.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        except sqlite3.DatabaseError:
            return
        if seg_count == 0 or fts_count >= seg_count:
            return
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE;")
            try:
                self.conn.execute("DELETE FROM segments_fts")
                self.conn.execute(
                    """
                    INSERT INTO segments_fts(rowid, text, speaker_name, camera, start_ts)
                    SELECT s.id, s.text, sp.name, s.camera, s.start_ts
                      FROM segments s LEFT JOIN speakers sp ON sp.id = s.speaker_id
                    """
                )
                self.conn.execute("COMMIT;")
            except Exception:
                self.conn.execute("ROLLBACK;")
                raise

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE;")
            try:
                yield self.conn
                self.conn.execute("COMMIT;")
            except Exception:
                self.conn.execute("ROLLBACK;")
                raise

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---- speakers -----------------------------------------------------------

    def ensure_speaker(self, name: str, notes: str | None = None) -> int:
        name = name.strip()
        if not name:
            raise ValueError("speaker name cannot be empty")
        with self.tx() as c:
            row = c.execute("SELECT id FROM speakers WHERE name=?", (name,)).fetchone()
            if row:
                return int(row["id"])
            cur = c.execute(
                "INSERT INTO speakers(name, notes) VALUES(?, ?)", (name, notes),
            )
            return int(cur.lastrowid)

    def rename_speaker(self, speaker_id: int, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("name cannot be empty")
        with self.tx() as c:
            c.execute("UPDATE speakers SET name=? WHERE id=?", (new_name, speaker_id))

    def delete_speaker(self, speaker_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE segments SET speaker_id=NULL, speaker_locked=0 WHERE speaker_id=?",
                      (speaker_id,))
            c.execute("DELETE FROM speakers WHERE id=?", (speaker_id,))

    def list_speakers(self) -> list[sqlite3.Row]:
        return self.conn.execute("""
            SELECT s.id, s.name, s.notes, s.created_at,
                   COALESCE(COUNT(seg.id), 0) AS segment_count
              FROM speakers s
         LEFT JOIN segments seg ON seg.speaker_id = s.id
          GROUP BY s.id
          ORDER BY s.name COLLATE NOCASE
        """).fetchall()

    # ---- segments -----------------------------------------------------------

    def insert_segment(self, *, camera: str, start_ts: dt.datetime, end_ts: dt.datetime,
                       text: str, speaker_id: int | None, chunk_file: str | None,
                       chunk_offset_s: float | None, embedding: np.ndarray | None) -> int:
        duration = max((end_ts - start_ts).total_seconds(), 0.0)
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        with self.tx() as c:
            cur = c.execute("""
                INSERT INTO segments(camera, start_ts, end_ts, duration_s, text,
                                     speaker_id, chunk_file, chunk_offset_s, embedding)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (camera, start_ts.isoformat(timespec="seconds"),
                  end_ts.isoformat(timespec="seconds"), duration, text,
                  speaker_id, chunk_file, chunk_offset_s, emb_blob))
            return int(cur.lastrowid)

    def set_segment_speaker(self, segment_id: int, speaker_id: int | None, *,
                            locked: bool = True) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE segments SET speaker_id=?, speaker_locked=? WHERE id=?",
                (speaker_id, 1 if locked else 0, segment_id),
            )

    def propagate_speaker(self, speaker_id: int, similar_ids: Iterable[int]) -> int:
        ids = list(similar_ids)
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        with self.tx() as c:
            cur = c.execute(
                f"UPDATE segments SET speaker_id=? "
                f" WHERE id IN ({placeholders}) AND speaker_locked=0",
                (speaker_id, *ids),
            )
            return cur.rowcount

    def segments_for_date(self, date: str, camera: str | None = None) -> list[sqlite3.Row]:
        base = """
            SELECT seg.*, sp.name AS speaker_name
              FROM segments seg
         LEFT JOIN speakers sp ON sp.id = seg.speaker_id
             WHERE substr(seg.start_ts, 1, 10) = ?
        """
        args: list = [date]
        if camera:
            base += " AND seg.camera = ?"
            args.append(camera)
        base += " ORDER BY seg.start_ts ASC"
        return self.conn.execute(base, args).fetchall()

    def segments_for_speaker(self, speaker_id: int, limit: int = 500,
                             offset: int = 0) -> list[sqlite3.Row]:
        return self.conn.execute("""
            SELECT seg.*, sp.name AS speaker_name
              FROM segments seg
         LEFT JOIN speakers sp ON sp.id = seg.speaker_id
             WHERE seg.speaker_id = ?
          ORDER BY seg.start_ts DESC
             LIMIT ? OFFSET ?
        """, (speaker_id, limit, offset)).fetchall()

    def search(self, query: str, limit: int = 200,
               date_from: str | None = None, date_to: str | None = None,
               speaker_id: int | None = None) -> list[sqlite3.Row]:
        """FTS5 ranked search with <mark>-highlighted snippet. Falls back to LIKE
        on pathological queries (FTS5 syntax errors from punctuation etc)."""
        q = query.strip()
        if not q:
            return []
        fts_query = self._to_fts_query(q)
        where = ["segments_fts.segments_fts MATCH ?"]
        args: list = [fts_query]
        if date_from:
            where.append("substr(seg.start_ts, 1, 10) >= ?")
            args.append(date_from)
        if date_to:
            where.append("substr(seg.start_ts, 1, 10) <= ?")
            args.append(date_to)
        if speaker_id is not None:
            where.append("seg.speaker_id = ?")
            args.append(speaker_id)
        sql = f"""
            SELECT seg.*, sp.name AS speaker_name,
                   snippet(segments_fts, 0, '<mark>', '</mark>', '…', 18) AS snippet,
                   bm25(segments_fts) AS rank
              FROM segments_fts
              JOIN segments seg ON seg.id = segments_fts.rowid
         LEFT JOIN speakers sp ON sp.id = seg.speaker_id
             WHERE {' AND '.join(where)}
          ORDER BY rank
             LIMIT ?
        """
        args.append(limit)
        try:
            return self.conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return self._search_like(q, limit, date_from, date_to, speaker_id)

    @staticmethod
    def _to_fts_query(q: str) -> str:
        """Produce a safe FTS5 query that defaults to AND-of-terms prefix search."""
        import re
        if '"' in q:
            return q
        tokens = [t for t in re.findall(r"[\w']+", q) if t]
        if not tokens:
            return f'"{q}"'
        return " AND ".join(f"{t}*" for t in tokens)

    def _search_like(self, query: str, limit: int,
                     date_from: str | None, date_to: str | None,
                     speaker_id: int | None) -> list[sqlite3.Row]:
        where = ["seg.text LIKE ?"]
        args: list = [f"%{query}%"]
        if date_from:
            where.append("substr(seg.start_ts, 1, 10) >= ?")
            args.append(date_from)
        if date_to:
            where.append("substr(seg.start_ts, 1, 10) <= ?")
            args.append(date_to)
        if speaker_id is not None:
            where.append("seg.speaker_id = ?")
            args.append(speaker_id)
        args.append(limit)
        return self.conn.execute(
            f"""
            SELECT seg.*, sp.name AS speaker_name,
                   REPLACE(seg.text, '', '') AS snippet, 0.0 AS rank
              FROM segments seg
         LEFT JOIN speakers sp ON sp.id = seg.speaker_id
             WHERE {' AND '.join(where)}
          ORDER BY seg.start_ts DESC
             LIMIT ?
            """, args,
        ).fetchall()

    def update_segment_text(self, segment_id: int, new_text: str) -> None:
        new_text = new_text.strip()
        if not new_text:
            raise ValueError("text cannot be empty")
        with self.tx() as c:
            c.execute("UPDATE segments SET text=? WHERE id=?", (new_text, segment_id))

    def delete_segment(self, segment_id: int) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM segments WHERE id=?", (segment_id,))

    def merge_speakers(self, source_id: int, into_id: int) -> int:
        if source_id == into_id:
            return 0
        with self.tx() as c:
            cur = c.execute(
                "UPDATE segments SET speaker_id=? WHERE speaker_id=?",
                (into_id, source_id),
            )
            moved = cur.rowcount
            c.execute("DELETE FROM speakers WHERE id=?", (source_id,))
            return int(moved)

    def segment(self, segment_id: int) -> sqlite3.Row | None:
        return self.conn.execute("""
            SELECT seg.*, sp.name AS speaker_name
              FROM segments seg
         LEFT JOIN speakers sp ON sp.id = seg.speaker_id
             WHERE seg.id = ?
        """, (segment_id,)).fetchone()

    def dates_with_segments(self, limit: int = 60) -> list[sqlite3.Row]:
        return self.conn.execute("""
            SELECT substr(start_ts, 1, 10) AS day,
                   COUNT(*) AS n,
                   MIN(start_ts) AS first_ts,
                   MAX(start_ts) AS last_ts
              FROM segments
          GROUP BY day
          ORDER BY day DESC
             LIMIT ?
        """, (limit,)).fetchall()

    # ---- speaker centroids for auto-assignment -----------------------------

    def speaker_centroids(self) -> list[tuple[int, str, np.ndarray]]:
        rows = self.conn.execute("""
            SELECT sp.id AS speaker_id, sp.name AS name, seg.embedding AS emb
              FROM segments seg
              JOIN speakers sp ON sp.id = seg.speaker_id
             WHERE seg.embedding IS NOT NULL AND seg.speaker_locked = 1
        """).fetchall()
        grouped: dict[tuple[int, str], list[np.ndarray]] = {}
        for r in rows:
            arr = np.frombuffer(r["emb"], dtype=np.float32)
            grouped.setdefault((int(r["speaker_id"]), r["name"]), []).append(arr)
        result: list[tuple[int, str, np.ndarray]] = []
        for (sid, name), embs in grouped.items():
            mean = np.mean(np.stack(embs), axis=0)
            norm = np.linalg.norm(mean) or 1.0
            result.append((sid, name, (mean / norm).astype(np.float32)))
        return result

    def segments_with_embeddings(self, speaker_id: int | None = None,
                                 limit: int = 20000) -> list[sqlite3.Row]:
        where = "seg.embedding IS NOT NULL"
        args: list = []
        if speaker_id is None:
            where += " AND seg.speaker_id IS NULL"
        else:
            where += " AND seg.speaker_id = ?"
            args.append(speaker_id)
        args.append(limit)
        return self.conn.execute(
            f"SELECT seg.id, seg.embedding, seg.speaker_id FROM segments seg "
            f"WHERE {where} ORDER BY seg.start_ts DESC LIMIT ?",
            args,
        ).fetchall()

    # ---- summaries ----------------------------------------------------------

    def save_summary(self, *, camera: str, summary_date: str,
                     segment_count: int, summary: str) -> None:
        with self.tx() as c:
            c.execute("""
                INSERT INTO summaries(camera, summary_date, segment_count, summary)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(camera, summary_date) DO UPDATE SET
                    segment_count=excluded.segment_count,
                    summary=excluded.summary,
                    created_at=datetime('now')
            """, (camera, summary_date, segment_count, summary))

    def summary(self, camera: str, summary_date: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM summaries WHERE camera=? AND summary_date=?",
            (camera, summary_date),
        ).fetchone()

    def recent_summaries(self, limit: int = 60) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM summaries ORDER BY summary_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
