#!/usr/bin/env python3
"""Retry Wikipedia fetches for entities that failed in Phase 2.

Reads wiki_pages_phase2.json, retries all entries with text=null (excluding
disambiguation pages), updates the file in place.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation_study_v3.wiki_utils import get_wikipedia_text  # noqa: E402


def _is_disambiguation_page(text: str) -> bool:
    first_500 = text[:500].lower()
    markers = ["may refer to", "can refer to", "commonly refers to", "(disambiguation)"]
    return any(m in first_500 for m in markers)


def main():
    pages_path = Path(__file__).resolve().parent.parent / "data/eval_1000q/wiki_pages_phase2.json"
    with open(pages_path) as f:
        pages = json.load(f)

    to_retry = [k for k, v in pages.items() if not v["text"] and not v["disambiguation"]]
    print(f"Retrying {len(to_retry)} entities with search fallback...")

    recovered = 0
    new_disambig = 0
    still_failed = 0

    for i, entity in enumerate(sorted(to_retry)):
        # Use get_wikipedia_text which tries exact title then search fallback
        text = get_wikipedia_text(entity, max_chars=8000)

        if text and _is_disambiguation_page(text):
            pages[entity] = {"text": None, "disambiguation": True}
            new_disambig += 1
        elif text:
            pages[entity] = {"text": text, "disambiguation": False}
            recovered += 1
        else:
            still_failed += 1

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(to_retry)}: "
                  f"{recovered} recovered, {new_disambig} disambig, {still_failed} failed")
        time.sleep(0.5)

    with open(pages_path, "w") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Recovered {recovered}, new disambiguation {new_disambig}, "
          f"still failed {still_failed}")
    print(f"Updated {pages_path}")

    with_text = sum(1 for v in pages.values() if v["text"])
    disambig = sum(1 for v in pages.values() if v["disambiguation"])
    no_text = sum(1 for v in pages.values() if not v["text"] and not v["disambiguation"])
    print(f"\nFinal: {with_text} with text, {disambig} disambiguation, {no_text} not found")


if __name__ == "__main__":
    main()
