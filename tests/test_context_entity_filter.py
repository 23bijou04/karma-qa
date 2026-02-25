"""Tests that _find_context_entities keeps KG-discovered entities.

Regression test for the bug where a word-overlap relevance filter silently
dropped entities like "Christopher Nolan" because the name doesn't appear
in the question "Where was the director of Inception born?".
"""

from unittest.mock import MagicMock, patch

from dbpedia_qa.models import Entity


# ── helpers ──────────────────────────────────────────────────────────────
def _make_sparql_row(uri: str, label: str) -> dict:
    """Simulate a row returned by execute_sparql()."""
    return {"entity": uri, "label": label}


# ── tests ────────────────────────────────────────────────────────────────
class TestContextEntityRetention:
    """Entities discovered via LLM-generated SPARQL must NOT be dropped."""

    @patch("hybrid_qa.execute_sparql")
    @patch("hybrid_qa.call_llm")
    def test_kg_discovered_entity_not_dropped(self, mock_llm, mock_sparql):
        """'Christopher Nolan' found via dbo:director must survive."""
        from hybrid_qa import _find_context_entities

        question = "Where was the director of Inception born?"
        seed = Entity(
            uri="http://dbpedia.org/resource/Inception",
            label="Inception",
            source="spotlight",
        )

        # LLM returns a SPARQL query (content doesn't matter, we mock execute)
        mock_llm.return_value = (
            "SELECT ?entity ?label WHERE { "
            "<http://dbpedia.org/resource/Inception> dbo:director ?entity . "
            "?entity rdfs:label ?label . FILTER(LANG(?label)='en') } LIMIT 10"
        )
        # execute_sparql returns seed + the director
        mock_sparql.return_value = (
            [
                _make_sparql_row(
                    "http://dbpedia.org/resource/Inception", "Inception"
                ),
                _make_sparql_row(
                    "http://dbpedia.org/resource/Christopher_Nolan",
                    "Christopher Nolan",
                ),
            ],
            None,  # no error
        )

        result = _find_context_entities(
            question=question,
            entities=[seed],
            schema_ctx="Entity: Inception\n  dbo:director -> Person",
            llm_client=MagicMock(),
            model="test-model",
            kg_client=MagicMock(),
        )

        labels = {e.label for e in result}
        assert "Christopher Nolan" in labels, (
            f"KG-discovered 'Christopher Nolan' was dropped. Got: {labels}"
        )
        assert "Inception" in labels  # seed must also survive

    @patch("hybrid_qa.execute_sparql")
    @patch("hybrid_qa.call_llm")
    def test_multiple_hop_entities_retained(self, mock_llm, mock_sparql):
        """All entities from a multi-hop SPARQL result are kept."""
        from hybrid_qa import _find_context_entities

        question = "Where was the director of Inception born?"
        seed = Entity(
            uri="http://dbpedia.org/resource/Inception",
            label="Inception",
            source="spotlight",
        )

        mock_llm.return_value = "SELECT ?entity ?label WHERE { ... } LIMIT 10"
        mock_sparql.return_value = (
            [
                _make_sparql_row(
                    "http://dbpedia.org/resource/Inception", "Inception"
                ),
                _make_sparql_row(
                    "http://dbpedia.org/resource/Christopher_Nolan",
                    "Christopher Nolan",
                ),
                _make_sparql_row(
                    "http://dbpedia.org/resource/London", "London"
                ),
            ],
            None,
        )

        result = _find_context_entities(
            question=question,
            entities=[seed],
            schema_ctx="Entity: Inception\n  dbo:director -> Person: birthPlace",
            llm_client=MagicMock(),
            model="test-model",
            kg_client=MagicMock(),
        )

        labels = {e.label for e in result}
        assert labels == {"Inception", "Christopher Nolan", "London"}, (
            f"Expected all 3 entities, got: {labels}"
        )

    @patch("hybrid_qa.execute_sparql")
    @patch("hybrid_qa.call_llm")
    def test_tier2_fallback_when_sparql_finds_nothing(
        self, mock_llm, mock_sparql
    ):
        """When Tier 1 SPARQL finds no new entities, falls back to Tier 2."""
        from hybrid_qa import _find_context_entities

        question = "Where was the director of Inception born?"
        seed = Entity(
            uri="http://dbpedia.org/resource/Inception",
            label="Inception",
            source="spotlight",
        )

        mock_llm.return_value = "SELECT ?entity ?label WHERE { ... } LIMIT 10"
        # SPARQL returns only the seed entity — no new discoveries
        mock_sparql.return_value = (
            [
                _make_sparql_row(
                    "http://dbpedia.org/resource/Inception", "Inception"
                ),
            ],
            None,
        )

        kg_client = MagicMock()
        # Tier 2 neighbor query returns Christopher Nolan
        kg_client.query.return_value = {
            "results": {
                "bindings": [
                    {
                        "p": {"value": "http://dbpedia.org/ontology/director"},
                        "neighbor": {
                            "value": "http://dbpedia.org/resource/Christopher_Nolan"
                        },
                        "label": {"value": "Christopher Nolan"},
                    }
                ]
            }
        }

        result = _find_context_entities(
            question=question,
            entities=[seed],
            schema_ctx="Entity: Inception\n  dbo:director -> Person",
            llm_client=MagicMock(),
            model="test-model",
            kg_client=kg_client,
        )

        labels = {e.label for e in result}
        assert "Christopher Nolan" in labels, (
            f"Tier 2 fallback should find 'Christopher Nolan'. Got: {labels}"
        )
