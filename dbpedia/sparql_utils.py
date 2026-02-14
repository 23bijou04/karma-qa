"""SPARQL escaping utilities to prevent injection."""


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
