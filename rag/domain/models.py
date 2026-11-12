# rag/domain/models.py
from dataclasses import dataclass


@dataclass
class Document:
    url: str
    page_title: str
    text: str
    source_id: str | None = None
    revision_id: str | None = None
    page_rank: int | None = None


@dataclass
class Chunk:
    text: str
    page_title: str
    section_header: str
    chunk_index: int
    char_offset: int

    url: str = ""
    chunk_id: str = ""
    source_id: str | None = None
    revision_id: str | None = None
    page_rank: int | None = None