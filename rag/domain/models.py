# rag/domain/models.py
from dataclasses import dataclass


@dataclass
class Document:
    url: str
    page_title: str
    text: str
    page_id: int | None = None
    revision_id: int | None = None
    page_rank: int | None = None


@dataclass
class Chunk:
    text: str
    page_title: str
    section_header: str
    chunk_index: int
    char_offset: int

    pageid: int | None = None
    revid: int | None = None
    url: str = ""
    chunk_id: str = ""
    page_rank: int | None = None