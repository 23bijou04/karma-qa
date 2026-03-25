#!/usr/bin/env python3
"""LogicRAGInstrumented — subclass of LogicRAG that captures all intermediate
data into self.last_trace without altering any logic or output.

The override of answer_question() calls the exact same inherited methods
in the same order as the original, wrapping each with timing + data capture.

Usage (after sys.path setup):
    from logicrag_instrumented import LogicRAGInstrumented
    rag = LogicRAGInstrumented(corpus_path)
    answer, contexts, rounds = rag.answer_question(question)
    print(rag.last_trace)       # full intermediate trace
    print(rag.last_token_delta) # {"prompt": N, "completion": N}
"""

import os
import sys
import time
from typing import List, Tuple

# Path setup — must happen before importing LogicRAG
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGICRAG_DIR = os.path.join(PROJECT_ROOT, "baselines", "LogicRAG")
sys.path.insert(0, LOGICRAG_DIR)

from src.models.logic_rag import LogicRAG  # noqa: E402
from src.utils.utils import TOKEN_COST  # noqa: E402


class LogicRAGInstrumented(LogicRAG):
    """LogicRAG with full intermediate trace capture.

    After each call to answer_question(), two attributes are populated:
        self.last_trace      — dict with warmup, dependency_sort, rounds, answer_gen
        self.last_token_delta — {"prompt": int, "completion": int} for this question
    """

    def __init__(self, corpus_path: str = None, cache_dir: str = "./cache",
                 filter_repeats: bool = False):
        super().__init__(corpus_path, cache_dir, filter_repeats)
        self.last_trace = None
        self.last_token_delta = None

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:
        """Override that mirrors parent logic exactly, adding trace capture.

        Returns the same (answer, last_contexts, round_count) tuple.
        """
        # Reset trace and token counters
        self.last_trace = None
        self.last_token_delta = None
        TOKEN_COST["prompt"] = 0
        TOKEN_COST["completion"] = 0
        t_total = time.time()

        info_summary = ""
        round_count = 0
        current_query = question
        last_contexts = []
        # Mirror filter_repeats tracking from parent
        retrieved_chunks_set = set() if self.filter_repeats else None
        round_traces = []

        # ============================================================
        # Stage 1: Warmup retrieval (same as parent lines 390-403)
        # ============================================================
        t0 = time.time()
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            new_contexts = self.retrieve(question)
        t_warmup_retrieval = time.time() - t0
        last_contexts = new_contexts

        t0 = time.time()
        info_summary = self.refine_summary_with_context(question, new_contexts, info_summary)
        t_warmup_summary = time.time() - t0

        t0 = time.time()
        analysis = self.warm_up_analysis(question, info_summary)
        t_warmup_analysis = time.time() - t0

        warmup_trace = {
            "query": question,
            "retrieved_chunks": new_contexts,
            "summary": info_summary,
            "analysis": analysis,
            "elapsed_sec": {
                "retrieval": round(t_warmup_retrieval, 3),
                "summary": round(t_warmup_summary, 3),
                "analysis": round(t_warmup_analysis, 3),
            },
        }

        # ============================================================
        # Early exit: warmup says can_answer (same as parent lines 405-411)
        # ============================================================
        if analysis["can_answer"]:
            t0 = time.time()
            answer = self.generate_answer(question, info_summary)
            t_answer_gen = time.time() - t0

            self.last_dependency_analysis = []
            self._assemble_trace(
                warmup=warmup_trace,
                dep_sort=None,
                round_traces=[],
                answer=answer,
                info_summary=info_summary,
                t_answer_gen=t_answer_gen,
                early_exit_warmup=True,
                t_total=t_total,
            )
            return answer, last_contexts, round_count

        # ============================================================
        # Stage 1.5: Dependency sort (same as parent lines 417-419)
        # ============================================================
        t0 = time.time()
        sorted_dependencies = self._sort_dependencies(analysis["dependencies"], question)
        t_dep_sort = time.time() - t0

        dep_sort_trace = {
            "raw_dependencies": analysis.get("dependencies", []),
            "sorted_dependencies": sorted_dependencies,
            "elapsed_sec": round(t_dep_sort, 3),
        }

        dependency_analysis_history = [{"sorted_dependencies": sorted_dependencies}]

        # ============================================================
        # Stage 2: Iterative rounds (same as parent lines 422-469)
        # ============================================================
        idx = 0

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1
            current_query = sorted_dependencies[idx]

            # Retrieval
            t0 = time.time()
            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                new_contexts = self.retrieve(current_query)
            t_round_retrieval = time.time() - t0
            last_contexts = new_contexts

            # Summary refinement
            t0 = time.time()
            info_summary = self.refine_summary_with_context(
                question, new_contexts, info_summary
            )
            t_round_summary = time.time() - t0

            # Dependency-aware analysis
            t0 = time.time()
            round_analysis = self.dependency_aware_rag(
                question, info_summary, sorted_dependencies, idx
            )
            t_round_analysis = time.time() - t0

            # Record round trace
            round_traces.append({
                "round": round_count,
                "query": current_query,
                "retrieved_chunks": new_contexts,
                "summary": info_summary,
                "analysis": round_analysis,
                "early_exit": bool(round_analysis.get("can_answer", False)),
                "elapsed_sec": {
                    "retrieval": round(t_round_retrieval, 3),
                    "summary": round(t_round_summary, 3),
                    "analysis": round(t_round_analysis, 3),
                },
            })

            # Mirror parent history tracking
            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": round_analysis,
            })

            if round_analysis.get("can_answer", False):
                t0 = time.time()
                answer = self.generate_answer(question, info_summary)
                t_answer_gen = time.time() - t0

                self.last_dependency_analysis = dependency_analysis_history
                self._assemble_trace(
                    warmup=warmup_trace,
                    dep_sort=dep_sort_trace,
                    round_traces=round_traces,
                    answer=answer,
                    info_summary=info_summary,
                    t_answer_gen=t_answer_gen,
                    early_exit_warmup=False,
                    t_total=t_total,
                )
                return answer, last_contexts, round_count
            else:
                idx += 1

        # ============================================================
        # Stage 3: Max rounds reached — generate best answer
        # ============================================================
        t0 = time.time()
        answer = self.generate_answer(question, info_summary)
        t_answer_gen = time.time() - t0

        self.last_dependency_analysis = dependency_analysis_history
        self._assemble_trace(
            warmup=warmup_trace,
            dep_sort=dep_sort_trace,
            round_traces=round_traces,
            answer=answer,
            info_summary=info_summary,
            t_answer_gen=t_answer_gen,
            early_exit_warmup=False,
            t_total=t_total,
        )
        return answer, last_contexts, round_count

    def _assemble_trace(self, *, warmup, dep_sort, round_traces, answer,
                        info_summary, t_answer_gen, early_exit_warmup, t_total):
        """Build self.last_trace and self.last_token_delta from captured data."""
        self.last_trace = {
            "warmup": warmup,
            "dependency_sort": dep_sort,
            "rounds": round_traces,
            "answer_gen": {
                "info_summary": info_summary,
                "answer": answer,
                "elapsed_sec": round(t_answer_gen, 3),
            },
            "early_exit_warmup": early_exit_warmup,
        }
        self.last_token_delta = {
            "prompt": TOKEN_COST["prompt"],
            "completion": TOKEN_COST["completion"],
        }
