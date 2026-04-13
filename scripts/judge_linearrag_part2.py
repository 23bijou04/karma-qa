"""
Judge script for LinearRAG part2 evaluation.
Uses rule-based normalization for clear cases, flags ambiguous for manual review.
"""
import json
import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "data/eval_1000q/judge_input/linearrag_part2.jsonl"
OUTPUT = PROJECT_ROOT / "data/eval_1000q/judge_output/linearrag_part2.jsonl"


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, normalize unicode, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s: str) -> set:
    return set(normalize(s).split())


def auto_judge(record: dict):
    """
    Returns True/False/'ambiguous'.
    True  = correct
    False = wrong
    'ambiguous' = needs manual review
    """
    pred = record.get("pred", "").strip()
    gold = record.get("gold", "").strip()
    aliases = record.get("aliases", [])

    # Empty pred → NO
    if not pred:
        return False

    pred_n = normalize(pred)
    gold_n = normalize(gold)

    # Exact match after normalization
    if pred_n == gold_n:
        return True

    # Alias match
    for alias in aliases:
        if normalize(alias) == pred_n:
            return True
        if normalize(alias) == gold_n and pred_n == gold_n:
            return True

    # Yes/No normalization
    yes_words = {"yes", "yeah", "yep", "yep", "true"}
    no_words = {"no", "nope", "nah", "false"}
    if gold_n in yes_words and pred_n in yes_words:
        return True
    if gold_n in no_words and pred_n in no_words:
        return True

    # Gold contained in pred (pred is more specific, e.g. "Munich" in "Nymphenburg Palace, Munich")
    if gold_n and gold_n in pred_n:
        return True

    # Pred contained in gold (pred is less specific) — only if pred is nearly all of gold
    if pred_n and pred_n in gold_n:
        # e.g. gold="Holy Roman Emperor" pred="Emperor" — too vague → ambiguous
        # Only auto-YES if pred covers most of gold tokens
        gold_toks = tokens(gold)
        pred_toks = tokens(pred)
        if len(pred_toks) >= len(gold_toks) * 0.8:
            return True

    # Date format normalization: handle "July 16, 2013" vs "2013-07-16" vs "16 July 2013"
    def extract_year(s):
        m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", s)
        return m.group(1) if m else None

    gold_year = extract_year(gold)
    pred_year = extract_year(pred)

    # Year-only match when gold has full date → flagged as ambiguous unless years match and gold is full date
    # Actually: if gold is a full date and pred is a year only → partial, mark as ambiguous
    gold_is_date = bool(re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", gold.lower()))
    pred_is_year_only = bool(re.fullmatch(r"\s*\d{4}\s*", pred.strip()))
    if gold_is_date and pred_is_year_only:
        return "ambiguous"

    # Both dates: normalize to year and see if years match, and see if month+day match
    if gold_year and pred_year and gold_year == pred_year:
        # Extract month numbers
        months = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12",
        }
        def get_month(s):
            sl = s.lower()
            for name, num in months.items():
                if name in sl:
                    return num
            m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}", s)
            if m:
                return m.group(1).zfill(2)
            return None
        def get_day(s):
            m = re.search(r"\b(\d{1,2})\b", s)
            return m.group(1).zfill(2) if m else None

        gm, pm = get_month(gold), get_month(pred)
        if gm and pm and gm == pm:
            return True
        if not gm and not pm:
            # Only year present in both
            return True

    # Nationality adjective ↔ country: "Italian" ↔ "Italy", "French" ↔ "France" etc.
    nationality_map = {
        "italian": "italy", "french": "france", "german": "germany",
        "spanish": "spain", "british": "united kingdom", "english": "england",
        "american": "united states", "australian": "australia", "dutch": "netherlands",
        "polish": "poland", "swedish": "sweden", "norwegian": "norway",
        "danish": "denmark", "finnish": "finland", "belgian": "belgium",
        "swiss": "switzerland", "austrian": "austria", "canadian": "canada",
        "japanese": "japan", "chinese": "china", "russian": "russia",
        "greek": "greece", "portuguese": "portugal", "hungarian": "hungary",
        "czech": "czech republic", "romanian": "romania", "turkish": "turkey",
        "iranian": "iran", "indian": "india", "scottish": "scotland",
        "welsh": "wales", "irish": "ireland", "mexican": "mexico",
        "brazilian": "brazil", "argentinian": "argentina", "argentine": "argentina",
        "ukrainian": "ukraine", "serbian": "serbia", "croatian": "croatia",
        "slovak": "slovakia", "slovenian": "slovenia", "bulgarian": "bulgaria",
        "lithuanian": "lithuania", "latvian": "latvia", "estonian": "estonia",
        "new zealand": "new zealand", "new zealander": "new zealand",
        "south african": "south africa", "nigerian": "nigeria",
        "egyptian": "egypt", "kenyan": "kenya", "ghanaian": "ghana",
        "algerian": "algeria", "moroccan": "morocco", "tunisian": "tunisia",
        "korean": "south korea", "south korean": "south korea",
        "north korean": "north korea", "vietnamese": "vietnam",
        "thai": "thailand", "philippine": "philippines", "filipino": "philippines",
        "indonesian": "indonesia", "malaysian": "malaysia", "singaporean": "singapore",
        "pakistani": "pakistan", "bangladeshi": "bangladesh", "sri lankan": "sri lanka",
        "nepalese": "nepal", "nepali": "nepal",
        "chilean": "chile", "colombian": "colombia", "peruvian": "peru",
        "venezuelan": "venezuela", "cuban": "cuba", "puerto rican": "puerto rico",
        "jamaican": "jamaica", "haitian": "haiti",
        "israeli": "israel", "lebanese": "lebanon", "jordanian": "jordan",
        "saudi": "saudi arabia", "saudi arabian": "saudi arabia",
        "emirati": "united arab emirates", "kuwaiti": "kuwait",
        "qatari": "qatar", "bahraini": "bahrain",
    }
    # reverse map: country → nationality
    reverse_nat = {v: k for k, v in nationality_map.items()}

    if gold_n in nationality_map and nationality_map[gold_n] == pred_n:
        return True
    if pred_n in nationality_map and nationality_map[pred_n] == gold_n:
        return True
    if gold_n in reverse_nat and reverse_nat[gold_n] == pred_n:
        return True
    if pred_n in reverse_nat and reverse_nat[pred_n] == gold_n:
        return True

    # "East Germany" vs "Germany" — related but distinct → ambiguous (handled manually)
    # "United States" vs "America" / "USA"
    us_variants = {"united states", "united states of america", "usa", "us", "america"}
    uk_variants = {"united kingdom", "great britain", "britain", "uk"}
    if gold_n in us_variants and pred_n in us_variants:
        return True
    if gold_n in uk_variants and pred_n in uk_variants:
        return True

    return "ambiguous"


def load_records():
    records = []
    with open(INPUT) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    records = load_records()
    print(f"Loaded {len(records)} records")

    ambiguous = []
    auto_results = {}

    for r in records:
        result = auto_judge(r)
        if result == "ambiguous":
            ambiguous.append(r)
        else:
            auto_results[r["id"]] = result

    print(f"Auto-judged: {len(auto_results)} ({sum(auto_results.values())} correct)")
    print(f"Ambiguous: {len(ambiguous)}")
    print("\nAmbiguous cases:")
    for r in ambiguous:
        print(f"  ID: {r['id']}")
        print(f"  Q:    {r['q']}")
        print(f"  Gold: {r['gold']}")
        print(f"  Pred: {r['pred']}")
        print(f"  Gemma: {r['gemma_judge']}")
        print()

    return records, auto_results, ambiguous


if __name__ == "__main__":
    main()
