#!/usr/bin/env python3
"""Wikidata deprecated/deleted claim checker for Phase 2 benchmark error detection."""

import json
import logging
from typing import Any

from scripts.wikidata_sparql_utils import (
    dbpedia_batch_qids,
    extract_year,
    is_date_like,
    wikidata_claim_query,
    wikidata_search_qid,
)

logger = logging.getLogger(__name__)

EVIDENCE_PROP_TO_PID: dict[str, Any] = {
    "director": "P57",
    "date of birth": "P569",
    "father": "P22",
    "date of death": "P570",
    "country of citizenship": "P27",
    "publication date": "P577",
    "place of birth": "P19",
    "spouse": "P26",
    "place of death": "P20",
    "mother": "P25",
    "country of origin": "P495",
    "country": ["P17", "P27"],
    "composer": "P86",
    "performer": "P175",
    "educated at": "P69",
    "place of burial": "P119",
    "inception": "P571",
    "employer": "P108",
    "award received": "P166",
    "child": "P40",
    "sibling": "P3373",
    "cause of death": "P509",
    "founded by": "P112",
    "producer": "P162",
    "publisher": "P123",
    "editor": "P98",
    "place of detention": "P2632",
}


def resolve_entity_qids(
    entity_names: list[str],
    uri_map: dict[str, str],
    cache_path: str,
) -> dict[str, str]:
    """Resolve entity names -> Wikidata QIDs; persist results in cache.

    1. Skip date-like values.
    2. DBpedia owl:sameAs batch queries (50 URIs/query, both directions).
    3. Wikidata wbsearchentities fallback for remaining names.
    """
    try:
        with open(cache_path) as f:
            cache: dict[str, str] = json.load(f)
    except FileNotFoundError:
        cache = {}

    missing = [n for n in entity_names if n not in cache and not is_date_like(n)]
    if not missing:
        return cache

    # DBpedia owl:sameAs path
    uri_to_name = {uri: name for name, uri in ((n, uri_map[n]) for n in missing if n in uri_map)}
    uris = list(uri_to_name.keys())
    total_batches = (len(uris) + 49) // 50

    for i in range(0, len(uris), 50):
        resolved = dbpedia_batch_qids(uris[i:i + 50])
        for uri, qid in resolved.items():
            if name := uri_to_name.get(uri):
                cache[name] = qid
        logger.info("DBpedia batch %d/%d: +%d QIDs", i // 50 + 1, total_batches, len(resolved))

    # Wikidata search fallback
    for name in (n for n in missing if n not in cache):
        if qid := wikidata_search_qid(name):
            cache[name] = qid

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    resolved_count = sum(1 for n in missing if n in cache)
    logger.info(
        "QID resolution: %d/%d non-date names resolved (%.0f%%)",
        resolved_count, len(missing),
        100 * resolved_count / max(len(missing), 1),
    )
    return cache


def _match_claim(obj_val: str, claims: list[tuple[str, str]], is_date: bool) -> str:
    """Classify one evidence value against Wikidata claims for an entity+property."""
    if is_date:
        target_year = extract_year(obj_val)
        if target_year is None:
            return "NOT_FOUND"
        for wd_val, rank in claims:
            if extract_year(wd_val) == target_year:
                return "DEPRECATED" if rank == "DeprecatedRank" else "CURRENT"
        return "NOT_FOUND"

    if not (obj_val.startswith("Q") and obj_val[1:].isdigit()):
        return "ENTITY_NOT_RESOLVED"

    # NOTE: Wikidata SPARQL auto-follows entity redirects, so wd_qid in results
    # may be a redirect target (e.g., Q200) while obj_val is the original QID
    # (e.g., Q100). This mismatch inflates NOT_FOUND counts for merged entities.
    for wd_val, rank in claims:
        wd_qid = wd_val.split("/")[-1] if "/" in wd_val else wd_val
        if wd_qid == obj_val:
            return "DEPRECATED" if rank == "DeprecatedRank" else "CURRENT"
    return "NOT_FOUND"


def check_claim_status(
    triples_by_pid: dict[str, list[tuple]],
    checkpoint_path: str,
) -> dict[tuple, str]:
    """Query Wikidata for evidence triples grouped by PID; supports checkpoint/resume.

    triples_by_pid: {PID: [(subj_qid, obj_val, question_id, is_date_prop), ...]}
    Returns: {(question_id, subj_qid, pid, obj_val): claim_status}
    """
    try:
        with open(checkpoint_path) as f:
            checkpoint: dict = json.load(f)
    except FileNotFoundError:
        checkpoint = {}

    results: dict[tuple, str] = {}

    for pid, triples in triples_by_pid.items():
        if pid in checkpoint:
            for key_str, status in checkpoint[pid].items():
                parts = key_str.split("|", 3)
                if len(parts) == 4:
                    results[tuple(parts)] = status  # type: ignore[arg-type]
            continue

        pid_results: dict[str, str] = {}
        found = wikidata_claim_query(pid, list({t[0] for t in triples}))

        for subj_qid, obj_val, question_id, is_date in triples:
            key = (question_id, subj_qid, pid, obj_val)
            key_str = "|".join(key)
            status = _match_claim(obj_val, found.get(subj_qid, []), is_date) if subj_qid in found else "NOT_FOUND"
            results[key] = status
            pid_results[key_str] = status

        checkpoint[pid] = pid_results
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info("PID %s done: %d triples checked", pid, len(triples))

    return results


def classify_errors(
    phase1_flags: list[dict],
    claim_results: dict[tuple, str],
) -> list[dict]:
    """Cross-reference Phase 1 text flags with Phase 2 Wikidata claim status.

    Taxonomy: WRONG_AT_CREATION | CLAIM_DELETED | DEPRECATED_SINCE | AMBIGUOUS
    """
    p1_by_qid = {f["question_id"]: f for f in phase1_flags}

    p2_by_qid: dict[str, list[dict]] = {}
    for (question_id, subj_qid, pid, obj_val), status in claim_results.items():
        if status in ("DEPRECATED", "NOT_FOUND"):
            p2_by_qid.setdefault(question_id, []).append(
                {"entity_qid": subj_qid, "property_pid": pid, "value_qid": obj_val, "claim_status": status}
            )

    merged: list[dict] = []
    for qid in set(p1_by_qid) | set(p2_by_qid):
        has_p1 = qid in p1_by_qid
        p2_entries = p2_by_qid.get(qid, [])
        statuses = {e["claim_status"] for e in p2_entries}

        if has_p1 and p2_entries:
            category = "WRONG_AT_CREATION"
        elif "NOT_FOUND" in statuses:
            category = "CLAIM_DELETED"
        elif "DEPRECATED" in statuses:
            category = "DEPRECATED_SINCE"
        else:
            category = "AMBIGUOUS"

        entry = {**(p1_by_qid.get(qid, {"question_id": qid})), "error_category": category}
        if p2_entries:
            entry["wikidata_claim_status"] = p2_entries
        merged.append(entry)

    return merged
