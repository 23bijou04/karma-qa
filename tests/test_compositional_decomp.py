"""Tests for compositional question decomposition (Patterns 1-6)."""

from hybrid_qa import (
    _PROP_DATE_DBO,
    _PROP_TO_DBO,
    _ROLE_TO_DBO,
    _decompose_compositional,
)


class TestDecomposeCompositional:
    def test_where_director_born(self):
        result = _decompose_compositional(
            "Where was the director of Inception born?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "director"
        assert "inception" in entity.lower()
        assert prop.lower() == "born"

    def test_when_author_died(self):
        result = _decompose_compositional(
            "When did the author of Harry Potter die?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "author"
        assert "harry potter" in entity.lower()
        assert prop.lower() in ("die", "died")

    def test_what_birthplace_of_director(self):
        result = _decompose_compositional(
            "What is the birthplace of the director of Inception?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "director"
        assert "inception" in entity.lower()
        assert prop.lower() == "birthplace"

    def test_where_director_of_film_born(self):
        """Handle 'the film X' or 'the movie X' prefix."""
        result = _decompose_compositional(
            "Where was the director of the film Inception born?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "director"
        assert "inception" in entity.lower()

    def test_non_compositional_returns_none(self):
        result = _decompose_compositional(
            "Who directed Inception?", [],
        )
        assert result is None

    def test_comparison_returns_none(self):
        """Comparison questions should NOT decompose here."""
        result = _decompose_compositional(
            "Which film was released earlier, Inception or Interstellar?", [],
        )
        assert result is None

    def test_simple_fact_returns_none(self):
        result = _decompose_compositional(
            "When was Albert Einstein born?", [],
        )
        # This is a direct question, not compositional (no bridge role)
        assert result is None


class TestRoleMappings:
    def test_all_roles_mapped(self):
        """Key roles have DBO mappings."""
        for role in ["director", "producer", "author", "writer", "composer"]:
            assert role in _ROLE_TO_DBO

    def test_all_props_mapped(self):
        """Key property words have DBO mappings."""
        for prop in ["born", "died", "educated"]:
            assert prop in _PROP_TO_DBO

    def test_compound_props_mapped(self):
        """Compound property words from Pattern 2 have DBO mappings."""
        for prop in ["birthplace", "deathplace", "nationality"]:
            assert prop in _PROP_TO_DBO

    def test_compound_date_props_mapped(self):
        """Compound date property words have DBO date mappings."""
        for prop in ["birthdate", "deathdate"]:
            assert prop in _PROP_DATE_DBO

    def test_pattern2_prop_resolves(self):
        """Pattern 2 extraction yields prop words that map to DBO properties."""
        result = _decompose_compositional(
            "What is the birthplace of the director of Inception?", [],
        )
        assert result is not None
        _, _, prop = result
        assert _PROP_TO_DBO.get(prop.lower()) is not None

    def test_family_roles_mapped(self):
        """Family roles have DBO mappings."""
        assert _ROLE_TO_DBO["father"] == "dbo:parent"
        assert _ROLE_TO_DBO["mother"] == "dbo:parent"
        assert _ROLE_TO_DBO["husband"] == "dbo:spouse"
        assert _ROLE_TO_DBO["wife"] == "dbo:spouse"
        assert _ROLE_TO_DBO["spouse"] == "dbo:spouse"

    def test_multiword_props_mapped(self):
        """Multi-word property phrases resolve to DBO properties."""
        assert _PROP_TO_DBO["place of birth"] == "dbo:birthPlace"
        assert _PROP_TO_DBO["place of death"] == "dbo:deathPlace"
        assert _PROP_TO_DBO["country"] == "dbo:nationality"
        assert _PROP_TO_DBO["graduated"] == "dbo:almaMater"

    def test_multiword_date_props_mapped(self):
        """Multi-word date property phrases resolve to DBO date properties."""
        assert _PROP_DATE_DBO["date of birth"] == "dbo:birthDate"
        assert _PROP_DATE_DBO["date of death"] == "dbo:deathDate"


class TestPossessivePatterns:
    """Tests for possessive X's <role> decomposition patterns."""

    # --- Pattern 3: "Where/When was/did X's <role> <verb>" ---

    def test_where_fathers_born(self):
        """Pattern 3: 'Where was X's father born?'"""
        result = _decompose_compositional(
            "Where was James Adam's father born?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "father"
        assert "james adam" in entity.lower()
        assert prop.lower() == "born"

    def test_where_father_graduated(self):
        """Pattern 3: 'Where did X's father graduate from?'"""
        result = _decompose_compositional(
            "Where did Charles Stewart's father graduate from?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "father"
        assert "charles stewart" in entity.lower()

    def test_why_wife_died(self):
        """Pattern 3: 'Why did X's wife die?'"""
        result = _decompose_compositional(
            "Why did John Middleton Murry's wife die?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "wife"
        assert "john middleton murry" in entity.lower()
        assert prop.lower() in ("die", "died")

    # --- Pattern 4: "What is the <prop-phrase> of X's <role>" ---

    def test_place_of_birth_husband(self):
        """Pattern 4: 'What is the place of birth of X's husband?'"""
        result = _decompose_compositional(
            "What is the place of birth of Consort Liang's husband?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "husband"
        assert "consort liang" in entity.lower()
        assert prop.lower() == "place of birth"

    def test_date_of_death_father(self):
        """Pattern 4: 'What is the date of death of X's father?'"""
        result = _decompose_compositional(
            "What is the date of death of Mohammed's father?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "father"
        assert "mohammed" in entity.lower()
        assert prop.lower() == "date of death"

    # --- Pattern 5: "What <noun> is X's <role>" ---

    def test_what_nationality_husband(self):
        """Pattern 5: 'What nationality is X's husband?'"""
        result = _decompose_compositional(
            "What nationality is Beatrice I's husband?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "husband"
        assert "beatrice i" in entity.lower()
        assert prop.lower() == "nationality"

    # --- Pattern 6: "Which country X's <role> is from" ---

    def test_which_country_father_from(self):
        """Pattern 6: 'Which country X's father is from?'"""
        result = _decompose_compositional(
            "Which country Thomas Lyon-Bowes's father is from?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "father"
        assert "thomas lyon-bowes" in entity.lower()
        assert prop.lower() == "country"

    # --- Edge cases ---

    def test_obrien_possessive_correct(self):
        """O'Brien-style names: possessive 's boundary must be after full name."""
        result = _decompose_compositional(
            "Where was O'Brien's father born?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "father"
        assert "o'brien" in entity.lower()

    def test_multiword_title_entity(self):
        """Multi-word titled entities like 'Mary Queen of Scots'."""
        result = _decompose_compositional(
            "Where was Mary Queen of Scots's husband born?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "husband"
        assert "mary queen of scots" in entity.lower()
        assert prop.lower() == "born"

    def test_married_verb(self):
        """Pattern 3 with 'married' verb."""
        result = _decompose_compositional(
            "When did Beatrice I's husband married?", [],
        )
        assert result is not None
        role, entity, prop = result
        assert role.lower() == "husband"
        assert prop.lower() == "married"

    def test_non_family_role_returns_none(self):
        """Possessive with unmapped role: decompose may match but _try_ rejects."""
        result = _decompose_compositional(
            "Where was Einstein's colleague born?", [],
        )
        # _decompose_ may match the pattern (returns tuple with role=colleague)
        # but _try_ would return None because 'colleague' not in _ROLE_TO_DBO
        if result is not None:
            role, _, _ = result
            assert role.lower() == "colleague"
            assert "colleague" not in _ROLE_TO_DBO

    def test_possessive_prop_resolves_in_dict(self):
        """Property words from possessive patterns resolve in _PROP_TO_DBO."""
        result = _decompose_compositional(
            "What is the place of birth of Consort Liang's husband?", [],
        )
        assert result is not None
        _, _, prop = result
        assert _PROP_TO_DBO.get(prop.lower()) is not None
