"""Evaluation metrics for 2WikiMultiHopQA benchmark.

Implements EM and F1 metrics for:
- Answer prediction
- Supporting facts prediction
- Evidence generation
- Joint metrics
"""

import re
import string
import unicodedata
from collections import Counter

# Month name -> number mapping for date parsing
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Regex patterns for common date formats
# ISO: 1932-08-16
_ISO_DATE = re.compile(r'^(\d{4})-(\d{1,2})-(\d{1,2})$')
# "16 August 1932" or "7 September 1868"
_DMY_DATE = re.compile(
    r'^(\d{1,2})\s+(' + '|'.join(_MONTH_MAP) + r')\s+(\d{4})$', re.IGNORECASE,
)
# "September 7, 1868" or "August 16, 1932"
_MDY_DATE = re.compile(
    r'^(' + '|'.join(_MONTH_MAP) + r')\s+(\d{1,2}),?\s+(\d{4})$', re.IGNORECASE,
)


def normalize_date(text: str) -> str | None:
    """Try to parse text as a date, return canonical YYYY-MM-DD or None.

    Handles three formats commonly seen in DBpedia answers and
    2WikiMultiHopQA gold answers:
      - ISO:  "1932-08-16"
      - DMY:  "16 August 1932"
      - MDY:  "September 7, 1868"

    Returns None if text is not a recognized date.
    """
    text = text.strip()

    m = _ISO_DATE.match(text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    m = _DMY_DATE.match(text)
    if m:
        day, month_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        return f"{year:04d}-{_MONTH_MAP[month_name]:02d}-{day:02d}"

    m = _MDY_DATE.match(text)
    if m:
        month_name, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        return f"{year:04d}-{_MONTH_MAP[month_name]:02d}-{day:02d}"

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Follows the official 2WikiMultiHopQA normalization:
    - Normalize dates to canonical YYYY-MM-DD format
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    # Unicode NFC normalization — prevent false mismatches from decomposed accents
    answer = unicodedata.normalize('NFC', answer)

    # Date normalization first (before punctuation removal destroys date hyphens)
    canonical_date = normalize_date(answer)
    if canonical_date is not None:
        return canonical_date

    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def normalize_whitespace(text: str) -> str:
        return ' '.join(text.split())

    answer = answer.lower()
    answer = remove_punctuation(answer)
    answer = remove_articles(answer)
    answer = normalize_whitespace(answer)

    return answer.strip()


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match score.

    Args:
        prediction: Predicted answer
        ground_truth: Gold answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score based on token overlap.

    Args:
        prediction: Predicted answer
        ground_truth: Gold answer

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_containment(prediction: str, ground_truth: str) -> float:
    """Check if gold answer appears anywhere in the prediction (upper bound)."""
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))
