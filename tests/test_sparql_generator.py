"""Tests for SPARQL query generation with hybrid schema discovery."""

import pytest

from dbpedia_qa.sparql_generator import SPARQLGenerator


class TestSPARQLGenerator:
    """Test SPARQL generation from questions."""

    def setup_method(self):
        self.generator = SPARQLGenerator()

    def test_grandfather_query(self):
        """Test grandfather (2-hop father chain) query."""
        sparql = self.generator.generate(
            "Who is the paternal grandfather of Eadred?"
        )

        assert "SELECT" in sparql
        assert "dbp:father" in sparql  # Uses dbp: namespace for actual data
        assert "?father" in sparql
        assert "grandfather" in sparql.lower()

    def test_comparison_born_first_query(self):
        """Test comparison query for birth dates."""
        sparql = self.generator.generate(
            "Who was born first, Albert Einstein or Isaac Newton?"
        )

        assert "SELECT" in sparql
        assert "birthDate" in sparql
        assert "Albert_Einstein" in sparql
        assert "Isaac_Newton" in sparql
        assert "ORDER BY" in sparql

    def test_director_birthplace_query(self):
        """Test compositional director->birthplace query."""
        sparql = self.generator.generate(
            "What is the birthplace of the director of film La La Land?"
        )

        assert "SELECT" in sparql
        assert "director" in sparql.lower()
        assert "birthPlace" in sparql or "birthplace" in sparql.lower()

    def test_nationality_comparison_query(self):
        """Test nationality comparison query."""
        sparql = self.generator.generate(
            "Were Scott Derrickson and Ed Wood of the same nationality?"
        )

        assert "SELECT" in sparql
        assert "nationality" in sparql.lower()
        assert "Scott_Derrickson" in sparql
        assert "Ed_Wood" in sparql

    def test_entity_extraction_or_pattern(self):
        """Test entity extraction with 'or' pattern."""
        entities = self.generator._extract_entities(
            "Who was born first, Einstein or Newton?"
        )

        assert "Einstein" in entities
        assert "Newton" in entities

    def test_entity_extraction_and_pattern(self):
        """Test entity extraction with 'and' pattern."""
        entities = self.generator._extract_entities(
            "Were Scott Derrickson and Ed Wood famous?"
        )

        assert "Scott Derrickson" in entities
        assert "Ed Wood" in entities

    def test_entity_extraction_of_pattern(self):
        """Test entity extraction with 'of X' pattern."""
        entities = self.generator._extract_entities(
            "Who is the father of Prince William?"
        )

        assert any("Prince William" in e or "William" in e for e in entities)

    def test_uri_conversion(self):
        """Test entity name to URI conversion."""
        uri = self.generator._to_dbpedia_uri("Albert Einstein")
        assert uri == "dbr:Albert_Einstein"

        uri = self.generator._to_dbpedia_uri("La La Land")
        assert uri == "dbr:La_La_Land"

    def test_sparql_has_prefixes(self):
        """Test that generated SPARQL includes required prefixes."""
        sparql = self.generator.generate("Who is the father of John?")

        # Should have either dbo: or dbp: prefix
        assert "PREFIX dbo:" in sparql or "PREFIX dbp:" in sparql

    def test_sparql_has_labels(self):
        """Test that generated SPARQL requests labels for readability."""
        sparql = self.generator.generate(
            "Who was born first, A or B?"
        )

        assert "Label" in sparql or "rdfs:label" in sparql


class TestSPARQLPatterns:
    """Test specific SPARQL pattern generation."""

    def setup_method(self):
        self.generator = SPARQLGenerator()

    def test_two_hop_chain(self):
        """Test 2-hop query pattern."""
        sparql = self.generator.generate(
            "Who is the paternal grandfather of someone?"
        )

        # Should have two dbp:father patterns chained (uses dbp: for actual data)
        assert sparql.count("dbp:father") >= 2

    def test_values_clause(self):
        """Test VALUES clause for multiple entities."""
        sparql = self.generator.generate(
            "Who was born first, Albert Einstein or Isaac Newton?"
        )

        assert "VALUES" in sparql

    def test_filter_optional(self):
        """Test OPTIONAL and FILTER patterns."""
        sparql = self.generator.generate(
            "Who is the father of someone?"
        )

        # Should have OPTIONAL for labels
        assert "OPTIONAL" in sparql or "label" in sparql.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
