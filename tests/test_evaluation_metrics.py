"""Comprehensive tests for evaluation/metrics.py.

Every evaluation number in the paper flows through this module.
Tests are grouped by function. Failures indicate potential metric bugs — do NOT
adjust tests to make them pass; report instead.
"""

import unicodedata

from evaluation.metrics import (
    compute_containment,
    compute_em,
    compute_f1,
    normalize_answer,
    normalize_date,
)

# ---------------------------------------------------------------------------
# normalize_date
# ---------------------------------------------------------------------------


class TestNormalizeDate:
    def test_iso_format(self):
        assert normalize_date("1932-08-16") == "1932-08-16"

    def test_iso_format_zero_padded(self):
        assert normalize_date("2024-01-05") == "2024-01-05"

    def test_iso_single_digit_month_and_day(self):
        # Regex allows 1-digit; output must be zero-padded
        assert normalize_date("2024-1-5") == "2024-01-05"

    def test_dmy_format(self):
        assert normalize_date("16 August 1932") == "1932-08-16"

    def test_dmy_single_digit_day(self):
        assert normalize_date("7 September 1868") == "1868-09-07"

    def test_dmy_case_insensitive(self):
        assert normalize_date("16 AUGUST 1932") == "1932-08-16"

    def test_mdy_format(self):
        assert normalize_date("September 7, 1868") == "1868-09-07"

    def test_mdy_with_comma(self):
        assert normalize_date("August 16, 1932") == "1932-08-16"

    def test_mdy_without_comma(self):
        # Regex has `?` on comma — "September 7 1868" should parse too
        assert normalize_date("September 7 1868") == "1868-09-07"

    def test_non_date_string(self):
        assert normalize_date("Paris") is None

    def test_empty_string(self):
        assert normalize_date("") is None

    def test_whitespace_only(self):
        assert normalize_date("   ") is None

    def test_partial_date_no_year(self):
        assert normalize_date("16 August") is None

    def test_invalid_month_name(self):
        assert normalize_date("16 Augustt 1932") is None

    def test_strips_surrounding_whitespace(self):
        # strip() is called inside normalize_date
        assert normalize_date("  1932-08-16  ") == "1932-08-16"

    def test_all_months_present(self):
        months = [
            ("january", "01"), ("february", "02"), ("march", "03"),
            ("april", "04"), ("may", "05"), ("june", "06"),
            ("july", "07"), ("august", "08"), ("september", "09"),
            ("october", "10"), ("november", "11"), ("december", "12"),
        ]
        for name, num in months:
            result = normalize_date(f"1 {name.capitalize()} 2000")
            assert result == f"2000-{num}-01", f"Failed for month {name}"


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Paris") == "paris"

    def test_removes_punctuation(self):
        assert normalize_answer("Hello, World!") == "hello world"

    def test_removes_article_the(self):
        assert normalize_answer("the United States") == "united states"

    def test_removes_article_a(self):
        assert normalize_answer("a dog") == "dog"

    def test_removes_article_an(self):
        assert normalize_answer("an apple") == "apple"

    def test_word_boundary_theater_not_modified(self):
        # "the" in "theater" must NOT be stripped
        result = normalize_answer("theater")
        assert "theater" in result, f"Expected 'theater' intact, got '{result}'"

    def test_word_boundary_another_not_modified(self):
        # "an" in "another" must NOT be stripped
        result = normalize_answer("another one")
        assert "another" in result, f"Expected 'another' intact, got '{result}'"

    def test_normalizes_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_only_punctuation(self):
        assert normalize_answer("...!!!") == ""

    def test_only_articles(self):
        assert normalize_answer("the a an") == ""

    def test_date_normalization_before_punctuation(self):
        # Date hyphens must survive — normalization order is critical
        assert normalize_answer("1932-08-16") == "1932-08-16"

    def test_dmy_date_normalized(self):
        assert normalize_answer("16 August 1932") == "1932-08-16"

    def test_mdy_date_normalized(self):
        assert normalize_answer("September 7, 1868") == "1868-09-07"

    def test_unicode_nfc_normalization(self):
        # é as single char (U+00E9) vs decomposed (e + U+0301)
        composed = "caf\u00e9"
        decomposed = unicodedata.normalize("NFD", composed)
        assert normalize_answer(composed) == normalize_answer(decomposed)

    def test_real_person_name(self):
        assert normalize_answer("John Quincy Adams") == "john quincy adams"

    def test_real_country_with_article(self):
        assert normalize_answer("the United Kingdom") == "united kingdom"


# ---------------------------------------------------------------------------
# compute_em
# ---------------------------------------------------------------------------


class TestComputeEM:
    def test_exact_match(self):
        assert compute_em("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert compute_em("Paris", "paris") == 1.0

    def test_article_insensitive(self):
        assert compute_em("the United States", "United States") == 1.0

    def test_date_format_insensitive(self):
        assert compute_em("16 August 1932", "1932-08-16") == 1.0

    def test_no_match(self):
        assert compute_em("Paris", "London") == 0.0

    def test_empty_vs_empty(self):
        assert compute_em("", "") == 1.0

    def test_empty_vs_non_empty(self):
        assert compute_em("", "Paris") == 0.0

    def test_non_empty_vs_empty(self):
        assert compute_em("Paris", "") == 0.0

    def test_returns_float(self):
        result = compute_em("Paris", "Paris")
        assert isinstance(result, float)

    def test_regression_john_quincy_adams(self):
        assert compute_em("John Quincy Adams", "john quincy adams") == 1.0

    def test_regression_date_cross_format(self):
        assert compute_em("1932-08-16", "16 August 1932") == 1.0


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------


class TestComputeF1:
    def test_exact_match(self):
        assert compute_f1("Paris", "Paris") == 1.0

    def test_partial_overlap(self):
        # pred="new york city" (3 tokens), gold="new york" (2 tokens)
        # common=2, P=2/3, R=2/2=1.0, F1=2*(2/3*1)/(2/3+1)=0.8
        result = compute_f1("New York City", "New York")
        assert abs(result - 0.8) < 1e-9

    def test_no_overlap(self):
        assert compute_f1("Paris", "London") == 0.0

    def test_both_empty(self):
        assert compute_f1("", "") == 1.0

    def test_prediction_empty_gold_non_empty(self):
        assert compute_f1("", "Paris") == 0.0

    def test_gold_empty_prediction_non_empty(self):
        assert compute_f1("Paris", "") == 0.0

    def test_single_token_vs_single_token_match(self):
        assert compute_f1("france", "France") == 1.0

    def test_repeated_tokens_counter_intersection(self):
        # "a a b" vs "a b b": common Counter = {a:1, b:1} → 2 common
        # pred=3 tokens, gold=3 tokens → P=2/3, R=2/3, F1=2/3
        result = compute_f1("a a b", "a b b")
        expected = 2 / 3
        assert abs(result - expected) < 1e-9

    def test_returns_float(self):
        assert isinstance(compute_f1("x", "y"), float)


# ---------------------------------------------------------------------------
# compute_containment
# ---------------------------------------------------------------------------


class TestComputeContainment:
    def test_gold_contained_in_prediction(self):
        assert compute_containment("The capital of France is Paris", "Paris") == 1.0

    def test_gold_not_in_prediction(self):
        assert compute_containment("The capital of Germany is Berlin", "Paris") == 0.0

    def test_exact_match(self):
        assert compute_containment("Paris", "Paris") == 1.0

    def test_normalization_applies(self):
        # "the United Kingdom" normalizes to "united kingdom";
        # "United Kingdom" normalizes to "united kingdom" — contained
        assert compute_containment("The United Kingdom is a country", "United Kingdom") == 1.0

    def test_empty_gold_in_any_prediction(self):
        # normalize("") = "" which is always a substring of any string
        assert compute_containment("Paris", "") == 1.0

    def test_returns_float(self):
        assert isinstance(compute_containment("x", "y"), float)


# ---------------------------------------------------------------------------
# Real-world regression cases from 2WikiMultiHopQA
# ---------------------------------------------------------------------------


class TestRealWorldRegressionCases:
    def test_person_name_case(self):
        assert compute_em("John Quincy Adams", "john quincy adams") == 1.0

    def test_date_cross_format(self):
        assert compute_em("1932-08-16", "16 August 1932") == 1.0

    def test_article_removal_uk(self):
        assert compute_em("the United Kingdom", "United Kingdom") == 1.0

    def test_empty_prediction_vs_paris(self):
        assert compute_em("", "Paris") == 0.0
        assert compute_f1("", "Paris") == 0.0
        assert compute_containment("", "Paris") == 0.0

    def test_date_mdy_format_regression(self):
        # "September 7, 1868" is a common DBpedia date format
        assert compute_em("September 7, 1868", "1868-09-07") == 1.0

    def test_f1_partial_country_name(self):
        # "United States of America" vs "United States" — partial overlap
        result = compute_f1("United States of America", "United States")
        assert result > 0.0 and result < 1.0
