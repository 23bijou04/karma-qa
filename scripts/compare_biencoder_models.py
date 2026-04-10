"""Compare bi-encoder models for RAG retrieval quality.

Indexes the SAME Wikipedia chunks with two bi-encoder models, then measures:
1. Top-20 retrieval overlap (how similar are the candidate sets?)
2. Top-8 after cross-encoder reranking overlap (does the reranker converge?)
3. Answer presence in final context (does the gold answer appear?)

Usage:
    python scripts/compare_biencoder_models.py
"""

import logging
import sys
import time
from pathlib import Path

# -- sys.path guard for project imports (matches hybrid_batch_eval.py pattern) --
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
_parent = str(Path(PROJECT_ROOT).parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── 10 test questions (5 correct, 5 wrong from TEXT-route eval) ──────────
TEST_QUESTIONS = [
    {"q": "Who is Albert G. Simms's father-in-law?", "gold": "Mark Hanna"},
    {"q": "Where was the director of film Age-Old Friends born?", "gold": "Winnipeg"},
    {"q": "Why did John Middleton Murry's wife die?", "gold": "tuberculosis"},
    {"q": "Who is the mother of the director of film Atomised (Film)?", "gold": "Gisela Elsner"},
    {"q": "Are both villages, Mavrolithari and Ampelakia, Evros, located in the same country?",
     "gold": "yes"},
    {"q": "Do both films, The Wild Frontier (Film) and The Sundowners (1960 Film), have the "
         "directors who are from the same country?", "gold": "yes"},
    {"q": "What is the place of birth of Gwladys's husband?", "gold": "Gwynllwg"},
    {"q": "What nationality is Beatrice I, Countess Of Burgundy's husband?", "gold": "Germany"},
    {"q": "What nationality is the director of film Vortex (1976 Film)?", "gold": "Italy"},
    {"q": "Where was the composer of film Love Story 1999 born?", "gold": "Tamil"},
]

MODEL_A = "all-MiniLM-L6-v2"       # Current (small, 384-dim)
MODEL_B = "all-mpnet-base-v2"       # LinearRAG's model (larger, 768-dim)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_BIENCODER = 20
TOP_K_RERANKER = 8


def load_models():
    """Load both bi-encoders and the cross-encoder."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"Loading bi-encoder A: {MODEL_A}...")
    enc_a = SentenceTransformer(MODEL_A)
    print(f"  → {enc_a.get_sentence_embedding_dimension()}-dim")

    print(f"Loading bi-encoder B: {MODEL_B}...")
    enc_b = SentenceTransformer(MODEL_B)
    print(f"  → {enc_b.get_sentence_embedding_dimension()}-dim")

    print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}...")
    cross_enc = CrossEncoder(CROSS_ENCODER_MODEL)

    return enc_a, enc_b, cross_enc


def get_chunks_for_question(question: str):
    """Fetch Wikipedia pages and chunk them. Returns list of WikiChunk."""
    from dbpedia_qa.dbpedia import DBpediaEntityLinker

    from graphrag_wiki import link_entities_to_dbpedia, retrieve_and_chunk_wikipedia

    linker = DBpediaEntityLinker()
    entities = link_entities_to_dbpedia(question, linker)
    if not entities:
        return [], []

    chunks = retrieve_and_chunk_wikipedia(entities, question, verbose=False)
    return chunks, entities


def build_faiss_index(chunks, encoder):
    """Build a FAISS flat index from chunks using the given encoder."""
    import faiss

    texts = [c.text for c in chunks]
    embeddings = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity on normalized vectors
    index.add(embeddings)
    return index, embeddings


def retrieve_top_k(query, encoder, index, chunks, top_k=20):
    """Retrieve top-k chunks by cosine similarity."""
    q_emb = encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    fetch_k = min(top_k * 2, index.ntotal)
    scores, indices = index.search(q_emb, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((chunks[idx], float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def rerank_chunks(query, chunks_with_scores, cross_encoder, top_k=8):
    """Rerank using cross-encoder (same logic as rag/reranker.py)."""
    pairs = [(query, c.text) for c, _ in chunks_with_scores]
    chunk_objs = [c for c, _ in chunks_with_scores]

    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(chunk_objs, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


def answer_in_context(chunks, gold_answer):
    """Check if gold answer string appears in any chunk text."""
    gold_lower = gold_answer.lower()
    combined = " ".join(c.text for c in chunks).lower()
    return gold_lower in combined


def chunk_overlap(list_a, list_b):
    """Compute overlap ratio between two lists of chunks (by text identity)."""
    set_a = {c.text for c in list_a}
    set_b = {c.text for c in list_b}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)  # Jaccard similarity


def run_comparison():
    """Run the A/B comparison on all test questions."""
    enc_a, enc_b, cross_enc = load_models()

    print(f"\n{'='*80}")
    print(f"Comparing: {MODEL_A} vs {MODEL_B}")
    print(f"Cross-encoder (shared): {CROSS_ENCODER_MODEL}")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print(f"{'='*80}\n")

    results = []

    for i, tq in enumerate(TEST_QUESTIONS):
        question = tq["q"]
        gold = tq["gold"]
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {question}")
        print(f"  Gold answer: {gold}")

        # Step 1: Get chunks (shared between both models)
        chunks, entities = get_chunks_for_question(question)
        if not chunks:
            print("  ⚠ No chunks retrieved, skipping\n")
            continue
        print(f"  Chunks: {len(chunks)} from {len(set(c.page_title for c in chunks))} pages")

        # Step 2: Build FAISS index with each model
        t0 = time.time()
        index_a, _ = build_faiss_index(chunks, enc_a)
        time_a = time.time() - t0

        t0 = time.time()
        index_b, _ = build_faiss_index(chunks, enc_b)
        time_b = time.time() - t0

        # Step 3: Retrieve top-20 with each model
        top_a = retrieve_top_k(question, enc_a, index_a, chunks, TOP_K_BIENCODER)
        top_b = retrieve_top_k(question, enc_b, index_b, chunks, TOP_K_BIENCODER)

        chunks_a = [c for c, _ in top_a]
        chunks_b = [c for c, _ in top_b]

        overlap_20 = chunk_overlap(chunks_a, chunks_b)

        # Step 4: Rerank both with same cross-encoder → top-8
        reranked_a = rerank_chunks(question, top_a, cross_enc, TOP_K_RERANKER)
        reranked_b = rerank_chunks(question, top_b, cross_enc, TOP_K_RERANKER)

        overlap_8 = chunk_overlap(reranked_a, reranked_b)

        # Step 5: Check answer presence
        ans_in_top20_a = answer_in_context(chunks_a, gold)
        ans_in_top20_b = answer_in_context(chunks_b, gold)
        ans_in_reranked_a = answer_in_context(reranked_a, gold)
        ans_in_reranked_b = answer_in_context(reranked_b, gold)

        result = {
            "question": question,
            "gold": gold,
            "n_chunks": len(chunks),
            "encode_time_a": round(time_a, 3),
            "encode_time_b": round(time_b, 3),
            "top20_overlap": round(overlap_20, 3),
            "reranked_overlap": round(overlap_8, 3),
            "ans_top20_a": ans_in_top20_a,
            "ans_top20_b": ans_in_top20_b,
            "ans_reranked_a": ans_in_reranked_a,
            "ans_reranked_b": ans_in_reranked_b,
        }
        results.append(result)

        # Print per-question summary
        print(f"  Encode time:     {MODEL_A}: {time_a:.3f}s | {MODEL_B}: {time_b:.3f}s")
        print(f"  Top-20 overlap:  {overlap_20:.1%} (Jaccard)")
        print(f"  Top-8 overlap:   {overlap_8:.1%} (after reranking)")
        ans_a_str = "yes" if ans_in_top20_a else "NO"
        ans_b_str = "yes" if ans_in_top20_b else "NO"
        print(f"  Answer in top-20:  {MODEL_A}: {ans_a_str} | {MODEL_B}: {ans_b_str}")
        ans_ra_str = "yes" if ans_in_reranked_a else "NO"
        ans_rb_str = "yes" if ans_in_reranked_b else "NO"
        print(f"  Answer in top-8:   {MODEL_A}: {ans_ra_str} | {MODEL_B}: {ans_rb_str}")

        # Flag cases where models disagree
        if ans_in_reranked_a != ans_in_reranked_b:
            winner = MODEL_B if ans_in_reranked_b else MODEL_A
            print(f"  ** DIFFERENCE: {winner} found the answer, the other didn't **")
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    if not results:
        print("No results to summarize.")
        return

    n = len(results)
    avg_overlap_20 = sum(r["top20_overlap"] for r in results) / n
    avg_overlap_8 = sum(r["reranked_overlap"] for r in results) / n
    avg_time_a = sum(r["encode_time_a"] for r in results) / n
    avg_time_b = sum(r["encode_time_b"] for r in results) / n

    ans_top20_a_count = sum(1 for r in results if r["ans_top20_a"])
    ans_top20_b_count = sum(1 for r in results if r["ans_top20_b"])
    ans_reranked_a_count = sum(1 for r in results if r["ans_reranked_a"])
    ans_reranked_b_count = sum(1 for r in results if r["ans_reranked_b"])

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Questions evaluated: {n}")
    print(f"\nAvg encode time:      {MODEL_A}: {avg_time_a:.3f}s | {MODEL_B}: {avg_time_b:.3f}s")
    print(f"Avg top-20 overlap:   {avg_overlap_20:.1%}")
    print(f"Avg top-8 overlap:    {avg_overlap_8:.1%} (after cross-encoder reranking)")
    print(f"\nAnswer recall (top-20):  {MODEL_A}: {ans_top20_a_count}/{n} | "
          f"{MODEL_B}: {ans_top20_b_count}/{n}")
    print(f"Answer recall (top-8):   {MODEL_A}: {ans_reranked_a_count}/{n} | "
          f"{MODEL_B}: {ans_reranked_b_count}/{n}")

    diff_count = sum(1 for r in results if r["ans_reranked_a"] != r["ans_reranked_b"])
    print(f"\nQuestions where models disagree on answer presence: {diff_count}/{n}")

    if avg_overlap_8 > 0.8:
        print("\n→ High reranked overlap: cross-encoder is converging both models to similar "
              "results. Upgrading bi-encoder likely has minimal end-to-end impact.")
    elif diff_count == 0:
        print("\n→ No answer-presence differences: both models retrieve the answer equally well "
              "at this corpus scale. Upgrade is unlikely to help.")
    else:
        b_wins = sum(1 for r in results
                     if r["ans_reranked_b"] and not r["ans_reranked_a"])
        a_wins = sum(1 for r in results
                     if r["ans_reranked_a"] and not r["ans_reranked_b"])
        print(f"\n→ {MODEL_B} found answer where {MODEL_A} didn't: {b_wins} times")
        print(f"→ {MODEL_A} found answer where {MODEL_B} didn't: {a_wins} times")


if __name__ == "__main__":
    run_comparison()
