"""SQLite-backed cache for Wikipedia chunks and embeddings."""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from rag.chunker import WikiChunk

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "wiki_cache.db"
_EVICTION_DAYS = 7


class WikiRAGCache:
    """Persistent cache for WikiChunks and their embeddings.

    Table schema:
        wiki_pages(title TEXT PK, chunks_json TEXT, embeddings BLOB, fetched_at REAL)

    Auto-evicts entries older than 7 days on each access.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DEFAULT_DB)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._init_table()
        self._evict_stale()

    def _init_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_pages (
                title TEXT PRIMARY KEY,
                chunks_json TEXT,
                embeddings BLOB,
                fetched_at REAL
            )
        """)
        self._conn.commit()

    def _evict_stale(self) -> None:
        cutoff = time.time() - _EVICTION_DAYS * 86400
        self._conn.execute("DELETE FROM wiki_pages WHERE fetched_at < ?", (cutoff,))
        self._conn.commit()

    # -- Chunks --

    def get_chunks(self, title: str) -> list[WikiChunk] | None:
        row = self._conn.execute(
            "SELECT chunks_json FROM wiki_pages WHERE title = ?", (title,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return [WikiChunk(**d) for d in json.loads(row[0])]

    def store_chunks(self, title: str, chunks: list[WikiChunk]) -> None:
        chunks_json = json.dumps([
            {
                "text": c.text, "page_title": c.page_title,
                "section_header": c.section_header,
                "chunk_index": c.chunk_index, "char_offset": c.char_offset,
            }
            for c in chunks
        ])
        self._conn.execute("""
            INSERT INTO wiki_pages (title, chunks_json, fetched_at)
            VALUES (?, ?, ?)
            ON CONFLICT(title) DO UPDATE SET chunks_json = excluded.chunks_json,
                                             fetched_at = excluded.fetched_at
        """, (title, chunks_json, time.time()))
        self._conn.commit()

    # -- Embeddings --

    def get_embeddings(self, title: str) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT embeddings FROM wiki_pages WHERE title = ?", (title,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).reshape(-1, 384)

    def store_embeddings(self, title: str, embeddings: np.ndarray) -> None:
        blob = embeddings.astype(np.float32).tobytes()
        self._conn.execute("""
            UPDATE wiki_pages SET embeddings = ? WHERE title = ?
        """, (blob, title))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
