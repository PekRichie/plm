"""PLM v0.4 — Könnyűsúlyú, telefonbarát SQLite FTS5 + BM25 helyi memória (On-Device RAG)."""
import re
import sqlite3
from dataclasses import dataclass


@dataclass
class MemoryHit:
    note_id: str
    title: str
    body: str
    score: float


class RagLiteMemory:
    """Offline, nulla-függőségű SQLite FTS5 memória kis helyi LLM-ekhez."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5(note_id UNINDEXED, title, body)"
        )
        self.conn.commit()

    def add_note(self, note_id: str, title: str, body: str) -> None:
        self.conn.execute("INSERT INTO notes (note_id, title, body) VALUES (?, ?, ?)", (note_id, title, body))
        self.conn.commit()

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ0-9]+", query)
        return " OR ".join(f'"{t}"' for t in terms)

    def search(self, query: str, top_k: int = 3) -> list[MemoryHit]:
        fts_q = self._fts_query(query)
        if not fts_q:
            return []
        rows = self.conn.execute(
            "SELECT note_id, title, body, bm25(notes) AS score FROM notes WHERE notes MATCH ? ORDER BY score LIMIT ?",
            (fts_q, top_k),
        ).fetchall()
        return [MemoryHit(note_id=r[0], title=r[1], body=r[2], score=round(float(r[3]), 4)) for r in rows]

    def build_context(self, query: str, top_k: int = 2) -> str:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "Nincs releváns helyi jegyzet."
        return "\n".join(f"[{h.note_id}] {h.title}: {h.body}" for h in hits)
