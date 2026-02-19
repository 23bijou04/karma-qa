"""SPARQL escaping utilities to prevent injection."""

import re

# Words too common to be useful in bif:contains full-text search
BIF_STOPWORDS = frozenset({
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to",
    "for", "is", "was", "by", "it", "its", "be", "are", "were",
})


def escape_bif_term(term: str) -> str:
    """Escape a term for use inside Virtuoso bif:contains expressions.

    bif:contains uses single-quoted terms inside a double-quoted SPARQL string,
    so escaping rules differ from escape_sparql_string():
      - Double single-quotes: O'Brien -> O''Brien
      - Strip characters that break bif:contains syntax: \\ " ( )
    """
    term = term.replace("'", "''")
    term = re.sub(r'[\\\"()\[\]{}<>;]', '', term)
    return term.strip()


def build_bif_expression(name: str) -> str | None:
    """Build a bif:contains boolean expression from a name string.

    Splits on whitespace, filters stopwords and short words,
    then joins with AND. Returns None if no usable terms remain.

    Examples:
        "Monster a Go-Go" -> "'Monster' AND 'Go-Go'"
        "The Gamecock"    -> "'Gamecock'"
        "a the"           -> None
    """
    words = name.split()
    terms = []
    for w in words:
        if w.lower() in BIF_STOPWORDS:
            continue
        if len(w) <= 1:
            continue
        escaped = escape_bif_term(w)
        if escaped:
            terms.append(f"'{escaped}'")
    return " AND ".join(terms) if terms else None


def escape_sparql_string(value: str) -> str:
    """Escape for safe embedding in SPARQL string literal."""
    return (value
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t"))


def escape_sparql_uri(uri: str) -> str:
    """Guard URI for <uri> syntax.

    Escapes characters illegal in SPARQL IRI references (<...>).
    Does NOT percent-encode other characters because DBpedia URIs
    already contain valid percent-encoded chars like %27.
    """
    return (uri
        .replace(">", "%3E")
        .replace(" ", "%20")
        .replace('"', "%22")
        .replace("{", "%7B")
        .replace("}", "%7D"))
