"""Stress test case data: 106 cases across 10 categories.

Organized by the function under test. Each constant is a list of pytest.param
objects for use with @pytest.mark.parametrize.

Verified against actual code behavior on 2026-03-25.
"""

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Category 1: Mention Extraction — _extract_mentions (cases 1–45)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (question, expected_fragments, expected_missing, desc)
#   expected_fragments: substrings that SHOULD appear in ≥1 mention (case-insensitive)
#   expected_missing:   strings that should NOT appear as exact mentions (case-insensitive)
# For xfail P0/P1: expected values describe CORRECT (desired) behavior.

# --- 1.1 Stopword stripping (cases 1–5) ---
MENTION_STOPWORD_CASES = [
    pytest.param(
        "Which of the in or?", [], [], "all stopwords → empty",
        id="p3_001_all_stopwords", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Who is The Who?", ["The Who"], [], "strip destroys 'The Who'",
        id="p1_002_the_who_strip", marks=pytest.mark.p1,
    ),
    pytest.param(
        "Who starred in It?", ["It"], [], "strip destroys 'It'",
        id="p1_003_it_strip", marks=pytest.mark.p1,
    ),
    pytest.param(
        "When did The Band form?", ["The Band"], [], "strip destroys 'The Band'",
        id="p1_004_the_band_strip", marks=pytest.mark.p1,
    ),
    pytest.param(
        "Who directed A Clockwork Orange?", ["A Clockwork Orange"], [],
        "strip removes article from title",
        id="p1_005_a_clockwork_strip", marks=pytest.mark.p1,
    ),
]

# --- 1.2 Possessive splitting (cases 6–12) ---
MENTION_POSSESSIVE_CASES = [
    pytest.param(
        "What is Shakespeare's birthplace?", ["Shakespeare"], [],
        "standard possessive split",
        id="p3_006_standard_possessive", marks=pytest.mark.p3,
    ),
    pytest.param(
        "What is Shakespeare\u2019s birthplace?", ["Shakespeare"], [],
        "curly apostrophe possessive",
        id="p3_007_curly_possessive", marks=pytest.mark.p3,
    ),
    pytest.param(
        "What is James's wife's name?", ["James"], [],
        "double possessive extracts first possessor",
        id="p2_008_double_possessive", marks=pytest.mark.p2,
    ),
    pytest.param(
        "What is McDonald's revenue?", [], ["McDonald"],
        "possessor 'McDonald' without apostrophe-s is wrong entity",
        id="p1_009_mcdonalds_possessive", marks=[pytest.mark.p1,
            pytest.mark.xfail(reason="P1: possessive split creates wrong mention", strict=False)],
    ),
    pytest.param(
        "Beyonc\u00e9\u2018s album", ["Beyonc\u00e9"], [],
        "left single quote not in possessive regex",
        id="p2_010_left_single_quote", marks=pytest.mark.p2,
    ),
    pytest.param(
        "The Beatles' first album", ["Beatles"], [],
        "plural possessive s' not handled",
        id="p2_011_plural_possessive", marks=pytest.mark.p2,
    ),
    pytest.param(
        "What is I's meaning?", [], [], "single-char possessor filtered",
        id="p3_012_single_char_possessor", marks=pytest.mark.p3,
    ),
]

# --- 1.3 NER entity processing (cases 13–21) ---
MENTION_NER_CASES = [
    pytest.param(
        "The Singing Fool or The Jazz Singer",
        ["Singing Fool", "Jazz Singer"], [],
        "or-splitting produces two entities",
        id="p3_013_or_split_two_entities", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Tell me about Simon or Garfunkel",
        ["Simon", "Garfunkel"], [],
        "or inside entity name causes split",
        id="p2_014_or_inside_entity", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who wrote The Aviator?", ["Aviator"], [],
        "entity ending with 'or' substring not stripped",
        id="p3_015_aviator_not_stripped", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Is Thor or Loki stronger?", ["Thor", "Loki"], [],
        "trailing 'or' stripped from NER span",
        id="p3_016_trailing_or_stripped", marks=pytest.mark.p3,
    ),
    pytest.param(
        "X or or", [], [], "multiple trailing or stripped",
        id="p3_017_multiple_trailing_or", marks=pytest.mark.p3,
    ),
    pytest.param(
        "or", [], [], "degenerate NER span of just 'or'",
        id="p3_018_just_or", marks=pytest.mark.p3,
    ),
    pytest.param(
        "In 1984, who directed the film?", [], [],
        "CARDINAL year filtered by NER label check",
        id="p2_019_cardinal_year_filtered", marks=pytest.mark.p2,
    ),
    pytest.param(
        "The first person to walk on the moon", [], [],
        "ORDINAL filtered out",
        id="p3_020_ordinal_filtered", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Who directed The Departed?", ["Departed"], [],
        "NER may miss entity, but noun chunk path catches partial",
        id="p2_021_departed_invisible_to_ner", marks=pytest.mark.p2,
    ),
]

# --- 1.4 Title-after-keyword extraction (cases 22–32) ---
MENTION_TITLE_KEYWORD_CASES = [
    pytest.param(
        "song Aren'T You Kind Of Glad We Did?",
        ["Aren'T You Kind Of Glad We Did"], [],
        "keyword + full title with apostrophe",
        id="p3_022_keyword_full_title", marks=pytest.mark.p3,
    ),
    pytest.param(
        "film was released first", [], [],
        "keyword followed by lowercase → skip",
        id="p3_023_keyword_lowercase_skip", marks=pytest.mark.p3,
    ),
    pytest.param(
        "What is this film", [], [],
        "keyword at end of question → skip",
        id="p3_024_keyword_at_end", marks=pytest.mark.p3,
    ),
    pytest.param(
        "film Lord of", ["Lord of"], [],
        "connector at end → title truncated to 'Lord'",
        id="p1_025_connector_at_end", marks=[pytest.mark.p1,
            pytest.mark.xfail(reason="P1: connector at end truncates title", strict=False)],
    ),
    pytest.param(
        "book Lord of the Rings", ["Lord of the Rings"], [],
        "connector chain 'of the' breaks lookahead",
        id="p0_026_connector_chain", marks=pytest.mark.p0,
    ),
    pytest.param(
        "film THE MATRIX", ["THE MATRIX"], [],
        "all-uppercase title extracted",
        id="p3_027_all_uppercase_title", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Who directed The Film?", [], [],
        "keyword IS the entity name → missed",
        id="p2_028_keyword_is_entity", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Is the song Yesterday or the film Yesterday?",
        ["Yesterday"], [],
        "two keywords → two title mentions (may dedup)",
        id="p3_029_two_keywords_two_titles", marks=pytest.mark.p3,
    ),
    pytest.param(
        "film \u201CInception\u201D", ["Inception"], [],
        "title starts with curly quote → lstrip handles it",
        id="p3_030_curly_quote_title_start", marks=pytest.mark.p3,
    ),
    pytest.param(
        "film Inception?", ["Inception"], [],
        "single-word title extracted",
        id="p3_031_single_word_title", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Which song, Inception or The Matrix?", ["Inception"], [],
        "keyword with trailing punctuation stripped",
        id="p3_032_keyword_trailing_punct", marks=pytest.mark.p3,
    ),
]

# --- 1.5 Deduplication & filtering (cases 33–39) ---
MENTION_DEDUP_CASES = [
    pytest.param(
        "Where is New York?", ["New York"], [],
        "case-insensitive dedup",
        id="p3_033_case_dedup", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Who is the director?", [], ["director"],
        "blocklisted word filtered",
        id="p3_034_blocklist_director", marks=pytest.mark.p3,
    ),
    pytest.param(
        "When was Prince born?", ["Prince"], [],
        "non-blocklisted entity passes",
        id="p3_035_prince_not_blocklisted", marks=pytest.mark.p3,
    ),
    pytest.param(
        "What is I?", [], [],
        "single-char mention filtered by len>1",
        id="p3_036_single_char_filtered", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Where is New York City?", ["New York"], [],
        "substring 'York' removed when 'New York' exists",
        id="p3_037_substring_removal", marks=pytest.mark.p3,
    ),
    pytest.param(
        "What is Dr. Dre's real name?", ["Dr. Dre"], [],
        "possessor-derived survives substring removal",
        id="p3_038_possessor_survives", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Is Angel or The Badman better?", [], [],
        "mutual non-substrings both kept",
        id="p3_039_mutual_non_substrings", marks=pytest.mark.p3,
    ),
]

# --- 1.6 Quoted entities (cases 40–45) ---
MENTION_QUOTED_CASES = [
    pytest.param(
        'Who wrote "Hamlet"?', ["Hamlet"], [],
        "standard double quotes captured",
        id="p3_040_standard_quotes", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Who wrote 'Hamlet'?", [], [],
        "single quotes NOT captured by regex",
        id="p2_041_single_quotes_miss", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who wrote \u201CHamlet\u201D?", ["Hamlet"], [],
        "curly double quotes not captured",
        id="p1_042_curly_quotes", marks=pytest.mark.p1,
    ),
    pytest.param(
        'He said "don\'t say \\"hello\\""', [], [],
        "nested quotes → partial match",
        id="p3_043_nested_quotes", marks=pytest.mark.p3,
    ),
    pytest.param(
        'Who wrote ""?', [], [],
        "empty quotes → no match",
        id="p3_044_empty_quotes", marks=pytest.mark.p3,
    ),
    pytest.param(
        'Who wrote "Hamlet?', [], [],
        "unbalanced quote → no match",
        id="p2_045_unbalanced_quote", marks=pytest.mark.p2,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Type Hinting — _extract_type_hint (cases 46–51)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (question, expected_type, desc)

TYPE_HINT_CASES = [
    pytest.param(
        "Who is the director of Inception?", "dbo:Film",
        "relation hint priority: director of → Film",
        id="p3_046_relation_hint_priority", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Where was Obama born?", None,
        "no type keyword → None",
        id="p2_047_no_type_keyword", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Which film and book share a name?", "dbo:Film",
        "multiple keywords → first match wins",
        id="p2_048_multiple_keywords", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who directed traffic?", "dbo:Film",
        "false positive: 'directed' triggers Film for non-film Q",
        id="p2_049_false_positive_directed", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who is this person?", "dbo:Person",
        "type hint returned but _TYPE_SUFFIX_MAP lacks it",
        id="p0_050_missing_suffix_map", marks=pytest.mark.p0,
    ),
    pytest.param(
        "She performed surgery", "dbo:MusicalWork",
        "false positive: 'performed' → MusicalWork",
        id="p2_051_false_positive_performed", marks=pytest.mark.p2,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: Type Validation — _validate_entity_types (cases 52–60)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (entities_data, question, expected_type, mock_types_response,
#          expected_replacement, desc)
# entities_data: list of (uri, label, source) tuples
# mock_types_response: return value for get_entity_types
# expected_replacement: True if entity should be replaced, False if kept

TYPE_VALIDATION_CASES = [
    pytest.param(
        [("http://dbpedia.org/resource/Abraham_Lincoln", "Lincoln", "spotlight")],
        "Who directed Lincoln?", "dbo:Film",
        [],  # no Film type
        True,  # should attempt replacement
        "Spotlight links person, expected Film → probe/search",
        id="p3_052_person_to_film", marks=pytest.mark.p3,
    ),
    pytest.param(
        [("http://dbpedia.org/resource/Lincoln", "Lincoln", "spotlight")],
        "Who directed Lincoln?", "dbo:Film",
        [],
        True,
        "URI probe fails → bif:contains search",
        id="p3_053_probe_fails_search", marks=pytest.mark.p3,
    ),
    pytest.param(
        [("http://dbpedia.org/resource/Lincoln", "Lincoln", "spotlight")],
        "Who directed Lincoln?", "dbo:Film",
        [],
        False,  # both fail → keeps original
        "both probe and search fail → keeps wrong entity",
        id="p2_054_both_fail_keep", marks=pytest.mark.p2,
    ),
    pytest.param(
        [("http://dbpedia.org/resource/The_Matrix", "The Matrix", "spotlight")],
        "the director of The Matrix", "dbo:Film",
        ["Film"],
        False,  # correct type → kept
        "obj_phrase extraction: 'X of Y' form",
        id="p3_055_obj_phrase_of_form", marks=pytest.mark.p3,
    ),
    pytest.param(
        [("http://dbpedia.org/resource/The_Matrix", "The Matrix", "spotlight")],
        "directed The Matrix", "dbo:Film",
        ["Film"],
        False,
        "obj_phrase extraction: verb form",
        id="p3_056_obj_phrase_verb_form", marks=pytest.mark.p3,
    ),
    pytest.param(
        [("http://dbpedia.org/resource/Test", "Test", "spotlight")],
        "directed the 2010 documentary Test", "dbo:Film",
        ["Film"],
        False,
        "year stripped by _clean regex",
        id="p3_057_year_stripped", marks=pytest.mark.p3,
    ),
    pytest.param(
        [("http://dbpedia.org/resource/The_Year_2000", "The Year 2000", "spotlight")],
        "wrote The Year 2000", "dbo:Book",
        [],
        True,  # _clean strips "The" and "2000" → searches for "Year" → wrong
        "_clean strips too much from title with year",
        id="p1_058_clean_strips_year_title", marks=[pytest.mark.p1,
            pytest.mark.xfail(reason="P1: _clean strips year from entity title", strict=False)],
    ),
    pytest.param(
        [("http://dbpedia.org/resource/Art", "Art", "spotlight")],
        "directed The Art of War", "dbo:Film",
        [],
        True,  # "art" in "the art of war".lower() → triggers validation
        "false positive substring match triggers validation",
        id="p2_059_false_positive_substring", marks=pytest.mark.p2,
    ),
    pytest.param(
        [("http://dbpedia.org/resource/Inception", "Inception", "spotlight")],
        "When was Inception released?", "dbo:Film",
        ["Film"],
        False,  # no relation word → returns unchanged
        "no relation word → entities unchanged",
        id="p3_060_no_relation_word", marks=pytest.mark.p3,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 4: Unlinked Detection — _find_unlinked (cases 61–67)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (mentions, entities_data, expected_unlinked, desc)
# entities_data: list of (uri, label, source) tuples

UNLINKED_CASES = [
    pytest.param(
        ["A B"], [("x", "A C", "s")], ["A B"],
        "Jaccard 1/3 < 0.5 → unlinked",
        id="p3_061_jaccard_below", marks=pytest.mark.p3,
    ),
    pytest.param(
        ["A B C"], [("x", "A B D", "s")], [],
        "Jaccard 2/4 = 0.5 → matched",
        id="p3_062_jaccard_boundary", marks=pytest.mark.p3,
    ),
    pytest.param(
        ["York"], [("x", "New York", "s")], [],
        "mention substring of surface → matched",
        id="p3_063_mention_in_surface", marks=pytest.mark.p3,
    ),
    pytest.param(
        ["New York City"], [("x", "New York", "s")], ["New York City"],
        "surface substring of mention → unlinked (disambiguation)",
        id="p3_064_surface_in_mention", marks=pytest.mark.p3,
    ),
    pytest.param(
        ["York New"], [("x", "New York", "s")], [],
        "word order differs → Jaccard 1.0 (set-based)",
        id="p3_065_word_order_differs", marks=pytest.mark.p3,
    ),
    pytest.param(
        ["Lincoln", "Inception"], [], ["Lincoln", "Inception"],
        "empty entities → all unlinked",
        id="p3_066_empty_entities", marks=pytest.mark.p3,
    ),
    pytest.param(
        ["Lincoln."], [("x", "Lincoln", "s")], [],
        "trailing period stripped → now matches correctly",
        id="p2_067_trailing_period", marks=pytest.mark.p2,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: bif:contains — build_bif_expression (cases 68–72)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (name, expected_expr, desc)
# expected_expr: expected return value (string or None)

BIF_CONTAINS_CASES = [
    pytest.param(
        "The One", "'One'", "stopword stripped → thin search term",
        id="p2_068_thin_search_term", marks=pytest.mark.p2,
    ),
    pytest.param(
        "A", None, "single-char name → filtered → None",
        id="p1_069_single_char_filtered", marks=pytest.mark.p1,
    ),
    pytest.param(
        "O'Brien", "'O''Brien'", "apostrophe doubled in bif:contains",
        id="p3_070_apostrophe_doubled", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Spider-Man", "'Spider-Man'", "hyphen preserved",
        id="p3_071_hyphen_preserved", marks=pytest.mark.p3,
    ),
    pytest.param(
        "It", None, "'It' filtered by BIF_STOPWORDS",
        id="p1_072_it_stopword_filtered", marks=pytest.mark.p1,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 5b: bif:contains search disambiguation (cases 73–75)
# ═══════════════════════════════════════════════════════════════════════════
# These test search_entity_by_name behavior — need mocked SPARQL client.
# Format: (search_name, entity_type, mock_return, expected_uri, desc)

BIF_SEARCH_DISAMBIGUATION_CASES = [
    pytest.param(
        "Paris", "dbo:City", "http://dbpedia.org/resource/Paris",
        "http://dbpedia.org/resource/Paris",
        "exact label match → returns city (no Film disambiguation)",
        id="p2_073_exact_label_match", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Lincoln", None, "http://dbpedia.org/resource/Lincoln",
        "http://dbpedia.org/resource/Lincoln",
        "shortest URI fallback heuristic",
        id="p2_074_shortest_uri_fallback", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Lincoln", "dbo:Film", None,
        None,
        "type-constrained search fails → returns None",
        id="p2_075_type_search_fails", marks=pytest.mark.p2,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 6: SPARQL Escaping — positive cases (cases 76–79)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (value, expected, desc)

SPARQL_ESCAPE_POSITIVE_CASES = [
    pytest.param(
        'Say "Hello"', 'Say \\"Hello\\"', "double quote escaped",
        id="p3_076_double_quote", marks=pytest.mark.p3,
    ),
    pytest.param(
        "C:\\path", "C:\\\\path", "backslash escaped",
        id="p3_077_backslash", marks=pytest.mark.p3,
    ),
    pytest.param(
        "line1\nline2", "line1\\nline2", "newline escaped",
        id="p3_078_newline", marks=pytest.mark.p3,
    ),
    pytest.param(
        "http://example.org/<test>", None, "angle brackets percent-encoded",
        id="p3_079_angle_brackets", marks=pytest.mark.p3,
    ),
]

# --- URI escaping gaps (cases 80–82) ---
SPARQL_URI_ESCAPE_CASES = [
    pytest.param(
        "http://dbpedia.org/resource/Aren't_You", "%27",
        "apostrophe should be %27 in URIs",
        id="p1_080_uri_apostrophe", marks=pytest.mark.p1,
    ),
    pytest.param(
        "http://dbpedia.org/resource/Did?", "%3F",
        "question mark should be %3F in URIs",
        id="p1_081_uri_question_mark", marks=pytest.mark.p1,
    ),
    pytest.param(
        "http://dbpedia.org/resource/Aren%27t_You", None,
        "already percent-encoded passthrough",
        id="p3_082_percent_encoded_passthrough", marks=pytest.mark.p3,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 7: Comparison Patterns — _has_incomplete_entity_coverage (cases 83–88)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (question, entities_data, expected_incomplete, desc)
# entities_data: list of (uri, label, source) tuples

COMPARISON_COVERAGE_CASES = [
    pytest.param(
        "Which came first, Inception or The Matrix?",
        [("x", "Inception", "s"), ("y", "The Matrix", "s")],
        False,
        "standard 'X or Y?' → both linked → not incomplete",
        id="p3_083_standard_or", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Are both Inception and The Matrix sci-fi?",
        [("x", "Inception", "s"), ("y", "The Matrix", "s")],
        True,  # regex captures ('Inception', 'The') — "The" ≠ linked label
        "'both X and Y' → non-greedy splits at first 'and'",
        id="p3_084_both_and", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Which is older Inception or The Matrix?",
        [("x", "Inception", "s"), ("y", "The Matrix", "s")],
        False,
        "missing comma → regex doesn't match → returns False",
        id="p2_085_missing_comma", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Which came first, X, Y, or Z?",
        [("x", "X", "s"), ("y", "Y", "s"), ("z", "Z", "s")],
        True,  # regex captures ('X, Y,', 'Z') — 'X, Y,' ≠ linked label
        "three-way comparison → regex captures wrong fragments",
        id="p2_086_three_way", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Which came first, Truth or Dare or The Matrix?",
        [("x", "Truth or Dare", "s"), ("y", "The Matrix", "s")],
        True,
        "entity name contains ' or ' → regex splits wrong",
        id="p2_087_entity_contains_or", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Are both Rock and Roll Hall of Fame and Madison Square Garden famous?",
        [("x", "Rock and Roll Hall of Fame", "s"),
         ("y", "Madison Square Garden", "s")],
        True,
        "entity name contains ' and ' → non-greedy splits wrong",
        id="p2_088_entity_contains_and", marks=pytest.mark.p2,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 8: spaCy NER limitations (cases 89–96)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (question, entity_name, expected_found, desc)
# expected_found: True if entity_name should appear in _extract_mentions output

SPACY_NER_CASES = [
    pytest.param(
        "Who directed Lincoln?", "Lincoln", True,
        "film title = person name → found via noun chunk",
        id="p2_089_lincoln_person_name", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who directed Drive?", "Drive", True,
        "film title = common noun → found via noun chunk",
        id="p2_090_drive_common_noun", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who directed Paris, Texas?", "Paris", True,
        "film title = location → spaCy sees GPE",
        id="p2_091_paris_texas_split", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who directed The Departed?", "Departed", True,
        "entity stripped of 'The' but partial survives",
        id="p2_092_departed_partial", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Is Angel And The Badman a good film?", "Angel", True,
        "entity split across NER spans",
        id="p2_093_angel_badman_split", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who directed New York, New York?", "New York", True,
        "GPE absorbs part of film title",
        id="p2_094_new_york_gpe", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Who directed the film Her?", "Her", True,
        "short entity name invisible to pipeline",
        id="p1_095_short_entity_her", marks=[pytest.mark.p1,
            pytest.mark.xfail(reason="P1: short entity name invisible", strict=False)],
    ),
    pytest.param(
        "Who directed Crouching Tiger, Hidden Dragon?",
        "Crouching Tiger", True,
        "non-English title partial recognition",
        id="p2_096_non_english_title", marks=pytest.mark.p2,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 9: _sparql_fallback (cases 97–100)
# ═══════════════════════════════════════════════════════════════════════════
# Format: (text, expected_phrases, desc)
# expected_phrases: capitalized phrases that should be passed to get_entity_by_label

SPARQL_FALLBACK_CASES = [
    pytest.param(
        "X? Who directed Y?", ["Y"],
        "capital after '?' excluded (sentence boundary)",
        id="p3_097_sentence_boundary", marks=pytest.mark.p3,
    ),
    pytest.param(
        "who directed inception?", [],
        "all lowercase → nothing found",
        id="p2_098_all_lowercase", marks=pytest.mark.p2,
    ),
    pytest.param(
        "Where is Z\u00fcrich?", ["Z\u00fcrich"],
        "non-ASCII capital recognized",
        id="p3_099_non_ascii_capital", marks=pytest.mark.p3,
    ),
    pytest.param(
        "Inception", ["Inception"],
        "single word entity found",
        id="p3_100_single_word", marks=pytest.mark.p3,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Category 10: Cache behavior (cases 101–103) — inline tests, no shared data
# ═══════════════════════════════════════════════════════════════════════════
# Cache tests are defined inline in test_fallback_cache_spacy_stress.py


# ═══════════════════════════════════════════════════════════════════════════
# Category 11: Additional cases (104–106)
# ═══════════════════════════════════════════════════════════════════════════

# Case 104 — Unicode normalization (tested in Phase 06)
# Case 105 — Curly quote in _extract_mentions (tested in Phase 02)
# Case 106 — NER/title interaction (tested in Phase 02)

MENTION_ADDITIONAL_CASES = [
    pytest.param(
        "Who directed \u201CInception\u201D?", ["Inception"], [],
        "curly quotes in extraction path (same as #42)",
        id="p1_105_curly_extraction", marks=pytest.mark.p1,
    ),
    pytest.param(
        "song Truth or Dare?", ["Truth or Dare"], [],
        "title-after-keyword vs NER or-splitting conflict",
        id="p2_106_title_vs_ner_or", marks=pytest.mark.p2,
    ),
]
