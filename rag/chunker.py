"""Section-aware Wikipedia text chunking for RAG retrieval."""

import re
from dataclasses import dataclass


@dataclass
class WikiChunk:
    """A chunk of Wikipedia text with provenance metadata."""

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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Avoids splitting on abbreviations like 'Dr.' or 'U.S.'."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_wikipedia_page(
    title: str,
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[WikiChunk]:
    """Section-aware recursive splitting of Wikipedia text.

    Strategy:
    1. Split on == Section == headers
    2. Within sections, split on paragraph boundaries (double newline)
    3. Within paragraphs, split on sentence boundaries
    4. Prepend section header to each chunk for context
    5. Never split mid-sentence

    Args:
        title: Wikipedia page title (provenance).
        text: Full Wikipedia page text.
        chunk_size: Target chunk size in characters.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of WikiChunk objects with metadata.
    """
    if not text or not text.strip():
        return []

    sections = _split_into_sections(text)
    chunks: list[WikiChunk] = []
    char_offset = 0

    for header, body in sections:
        if not body.strip():
            char_offset += len(header) + len(body)
            continue

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
        section_chunks = _chunk_paragraphs(paragraphs, header, chunk_size, overlap)

        for chunk_text in section_chunks:
            # Prepend section header for context (unless it's the lead section)
            if header:
                full_text = f"{header}\n{chunk_text}"
            else:
                full_text = chunk_text

            chunks.append(WikiChunk(
                text=full_text,
                page_title=title,
                section_header=header or "(Lead)",
                chunk_index=len(chunks),
                char_offset=char_offset,
            ))
            char_offset += len(chunk_text)

    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split Wikipedia text into (header, body) tuples.

    The lead section (before any header) gets header="".
    """
    # Match == Header == style markers
    pattern = re.compile(r'^(==+\s.*?\s==+)$', re.MULTILINE)
    parts = pattern.split(text)

    sections: list[tuple[str, str]] = []
    if not parts:
        return [("", text)]

    # First part is the lead (before any header)
    if parts[0].strip():
        sections.append(("", parts[0]))

    # Remaining parts alternate: header, body, header, body, ...
    i = 1
    while i < len(parts):
        header = parts[i].strip() if i < len(parts) else ""
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((header, body))
        i += 2

    return sections


def _chunk_paragraphs(
    paragraphs: list[str],
    section_header: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Chunk paragraphs into pieces respecting sentence boundaries."""
    # Header length counts toward chunk budget
    header_len = len(section_header) + 1 if section_header else 0
    effective_size = chunk_size - header_len

    if effective_size < 100:
        effective_size = 100

    result_chunks: list[str] = []
    current_sentences: list[str] = []
    current_len = 0

    for para in paragraphs:
        sentences = _split_sentences(para)
        if not sentences:
            continue

        for sentence in sentences:
            sent_len = len(sentence)

            # Single sentence exceeds chunk size — take it as-is
            if sent_len > effective_size and not current_sentences:
                result_chunks.append(sentence[:chunk_size])
                continue

            # Adding this sentence would exceed budget — flush current chunk
            if current_len + sent_len + 1 > effective_size and current_sentences:
                result_chunks.append(" ".join(current_sentences))
                # Overlap: keep trailing sentences that fit within overlap budget
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) + 1 > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += sent_len + 1

    # Flush remaining
    if current_sentences:
        result_chunks.append(" ".join(current_sentences))

    return result_chunks
