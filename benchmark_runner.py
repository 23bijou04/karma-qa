"""
Run benchmark questions and evaluate results.
"""

import os
import sys
import re
import json
import time
import argparse

# Add parent directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_dir)
sys.path.insert(0, parent_dir)

from openai import OpenAI
from SPARQLWrapper import SPARQLWrapper, JSON
import requests

from dbpedia_qa.dbpedia.schema_discovery import SchemaDiscovery

# Configuration
GROQ_API_KEY = "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"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
EXTRACTION_MODEL = "llama-3.3-70b-versatile"
DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"


# Test questions (4 from sample + 6 additional diverse questions)
BENCHMARK_QUESTIONS = [
    # From sample file
    {
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison"
    },
    {
        "question": "What is the birthplace of the director of film It's a Wonderful World (1939 Film)?",
        "answer": "San Francisco",
        "type": "compositional"
    },
    {
        "question": "Who is the paternal grandfather of Eadred, King of England?",
        "answer": "Alfred the Great",
        "type": "inference"
    },
    {
        "question": "Which film has the director born first, The Millionaire (1927 Film) or Grand Prix (1934 Film)?",
        "answer": "Grand Prix (1934 film)",
        "type": "bridge_comparison"
    },
    # Additional questions
    {
        "question": "Who is the paternal grandfather of Prince William?",
        "answer": "Prince Philip",
        "type": "inference"
    },
    {
        "question": "Which film came out first, Blade Runner or The Terminator?",
        "answer": "Blade Runner",
        "type": "comparison"
    },
    {
        "question": "What is the capital of the country where Albert Einstein was born?",
        "answer": "Berlin",
        "type": "compositional"
    },
    {
        "question": "Who directed the film Inception?",
        "answer": "Christopher Nolan",
        "type": "simple"
    },
    {
        "question": "Were The Beatles and The Rolling Stones from the same country?",
        "answer": "yes",
        "type": "comparison"
    },
    {
        "question": "What is the birthplace of the author of Harry Potter?",
        "answer": "Yate",
        "type": "compositional"
    }
]


def extract_entity(client: OpenAI, question: str) -> str:
    """Extract entity from question."""
    user_msg = f"""Extract the STARTING entity we need to look up in a knowledge graph.

Rules:
- Return ONLY the entity name, no titles or descriptions
- Remove suffixes like "King of England", "President of USA" - just return the name
- For "Who is X of Y?" questions, extract Y (just the name)
- For "What is X of Y of Z?" questions, extract Z (the innermost entity)
- For comparison questions "X or Y", extract both separated by " AND "
- For "same X" comparisons, extract both entities separated by " AND "

Examples:
- "Who is the father of Barack Obama?" → Barack Obama
- "Who is the paternal grandfather of Prince William?" → Prince William
- "Who is the paternal grandfather of Eadred, King of England?" → Eadred
- "What is the birthplace of the director of film Inception?" → Inception
- "What is the birthplace of the director of film It's a Wonderful World (1939 Film)?" → It's a Wonderful World
- "Were Scott Derrickson and Ed Wood of the same nationality?" → Scott Derrickson AND Ed Wood
- "Which film came out first, A or B?" → A AND B
- "Which film has the director born first, The Millionaire (1927 Film) or Grand Prix (1934 Film)?" → The Millionaire AND Grand Prix

Question: {question}

Entity to look up:"""

    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=100
    )

    entity = response.choices[0].message.content.strip().strip('"\'.')
    entity = entity.split('\n')[0].strip()

    # Clean up common suffixes
    for suffix in [", King of England", ", President", ", Duke of", ", Queen of"]:
        if suffix in entity:
            entity = entity.split(suffix)[0]

    # Fallback extraction
    if not entity or len(entity) < 2:
        match = re.search(r'film\s+([^?]+?)(?:\?|$)', question, re.IGNORECASE)
        if match:
            entity = match.group(1).strip().rstrip('?')
        else:
            match = re.search(r'of\s+([A-Z][a-zA-Z\s\']+?)(?:\?|$)', question)
            if match:
                entity = match.group(1).strip()

    return entity


def run_pipeline(question: str, llm_client: OpenAI, schema_discovery: SchemaDiscovery,
                 sparql_client: SPARQLWrapper) -> dict:
    """Run full pipeline for a question."""
    result = {
        "question": question,
        "extracted_entity": None,
        "sparql_query": None,
        "kg_facts": [],
        "wikipedia_evidence": {},
        "answer": None,
        "error": None
    }

    try:
        # Step 1: Extract entity
        entity = extract_entity(llm_client, question)
        result["extracted_entity"] = entity

        # Handle multiple entities
        entities = [e.strip() for e in entity.split(" AND ")] if " AND " in entity else [entity]

        # Step 2: Discover schema
        schemas = []
        combined_schema_text = []

        for ent in entities:
            schema = schema_discovery.get_schema_for_entity(ent)
            schemas.append(schema)
            combined_schema_text.append(schema['schema_text'])

        schema_text = "\n\n---\n\n".join(combined_schema_text)

        # Step 3: Generate SPARQL
        sparql_prompt = f"""You are a SPARQL expert. Generate a query to answer the question.

SCHEMA INFORMATION:
{schema_text}

QUESTION: {question}

RULES:
1. Use the EXACT entity URIs shown above
2. For family relations like "grandfather", chain properties (e.g., father→father)
3. Prefer dbp: properties as they have better coverage
4. Use OPTIONAL for labels
5. Make sure query is complete and valid

Generate ONLY the SPARQL query, no explanation:"""

        response = llm_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": sparql_prompt}],
            temperature=0.0,
            max_tokens=1000
        )

        sparql_query = response.choices[0].message.content.strip()
        if "```" in sparql_query:
            lines = sparql_query.split("\n")
            sparql_query = "\n".join([l for l in lines if not l.strip().startswith("```")])

        result["sparql_query"] = sparql_query

        # Step 4: Execute SPARQL
        sparql_client.setQuery(sparql_query)
        kg_facts = []
        extracted_entities = []

        try:
            results = sparql_client.query().convert()
            bindings = results["results"]["bindings"]

            for row in bindings:
                for var, val in row.items():
                    value = val["value"]
                    if val["type"] == "uri":
                        entity_name = value.split("/")[-1].replace("_", " ")
                        extracted_entities.append(entity_name)
                        kg_facts.append(f"{var}: {entity_name}")
                    else:
                        kg_facts.append(f"{var}: {value}")
        except Exception as e:
            result["error"] = f"SPARQL error: {e}"

        result["kg_facts"] = kg_facts

        # Step 5: Fetch Wikipedia evidence
        all_entities = entities + extracted_entities
        all_entities = list(dict.fromkeys(all_entities))[:5]

        wikipedia_evidence = {}
        for ent in all_entities:
            ent_encoded = ent.replace(" ", "_").replace(",", "%2C")
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{ent_encoded}"
            try:
                resp = requests.get(url, headers={"User-Agent": "DBpediaQA/1.0"}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    wikipedia_evidence[ent] = data.get("extract", "")[:300]
            except:
                pass

        result["wikipedia_evidence"] = wikipedia_evidence

        # Step 6: Generate answer
        context = "KNOWLEDGE GRAPH FACTS:\n"
        for fact in kg_facts:
            context += f"  - {fact}\n"

        context += "\nWIKIPEDIA EVIDENCE:\n"
        for ent, text in wikipedia_evidence.items():
            context += f"\n[{ent}]:\n{text}\n"

        answer_prompt = f"""Based on the following context, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide your answer in this format:
ANSWER: [the direct answer - a name, place, yes/no, etc.]
REASONING: [brief explanation]"""

        response = llm_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.0,
            max_tokens=300
        )

        full_response = response.choices[0].message.content.strip()

        # Extract just the answer line
        answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', full_response, re.IGNORECASE)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        else:
            # Fallback to first line
            result["answer"] = full_response.split('\n')[0].strip()

    except Exception as e:
        result["error"] = str(e)

    return result


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    # Remove common prefixes
    for prefix in ["the ", "a ", "an "]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
    return answer


def check_answer(predicted: str, expected: str) -> bool:
    """Check if predicted answer matches expected."""
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    # Empty prediction is always wrong
    if not pred_norm or len(pred_norm) < 2:
        return False

    # Exact match
    if pred_norm == exp_norm:
        return True

    # Partial match (one contains the other)
    if exp_norm in pred_norm or pred_norm in exp_norm:
        return True

    # Handle yes/no
    if exp_norm in ["yes", "no"]:
        if exp_norm == "yes" and any(w in pred_norm for w in ["yes", "same", "both", "correct"]):
            return True
        if exp_norm == "no" and any(w in pred_norm for w in ["no", "not", "different"]):
            return True

    return False


def run_benchmark(questions: list, verbose: bool = True):
    """Run benchmark on questions."""
    print("=" * 70)
    print("  DBpedia GraphRAG Benchmark Runner")
    print("=" * 70)
    print(f"\nRunning {len(questions)} questions...")
    print(f"Model: {MODEL}")
    print(f"Extraction Model: {EXTRACTION_MODEL}")
    print()

    # Initialize clients
    llm_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    schema_discovery = SchemaDiscovery(DBPEDIA_ENDPOINT)
    sparql_client = SPARQLWrapper(DBPEDIA_ENDPOINT)
    sparql_client.addCustomHttpHeader("User-Agent", "DBpediaQA/1.0")
    sparql_client.setReturnFormat(JSON)

    results = []
    correct = 0

    for i, q in enumerate(questions):
        print(f"\n{'='*70}")
        print(f"Question {i+1}/{len(questions)}: {q['question']}")
        print(f"Expected: {q['answer']} | Type: {q['type']}")
        print("-" * 70)

        # Run pipeline
        start_time = time.time()
        result = run_pipeline(q['question'], llm_client, schema_discovery, sparql_client)
        elapsed = time.time() - start_time

        # Check answer
        is_correct = check_answer(result['answer'] or "", q['answer'])
        if is_correct:
            correct += 1

        result['expected'] = q['answer']
        result['type'] = q['type']
        result['is_correct'] = is_correct
        result['time'] = elapsed
        results.append(result)

        # Print result
        status = "✓" if is_correct else "✗"
        print(f"Entity: {result['extracted_entity']}")
        print(f"KG Facts: {result['kg_facts'][:3]}...")
        print(f"Answer: {result['answer']}")
        print(f"Result: {status} ({elapsed:.1f}s)")

        if result['error']:
            print(f"Error: {result['error']}")

        # Rate limiting
        time.sleep(1)

    # Summary
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    accuracy = correct / len(questions) * 100
    print(f"\nOverall: {correct}/{len(questions)} correct ({accuracy:.1f}%)")

    # By type
    by_type = {}
    for r in results:
        t = r['type']
        if t not in by_type:
            by_type[t] = {'correct': 0, 'total': 0}
        by_type[t]['total'] += 1
        if r['is_correct']:
            by_type[t]['correct'] += 1

    print("\nBy Question Type:")
    for t, stats in by_type.items():
        pct = stats['correct'] / stats['total'] * 100
        print(f"  {t}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    # Details
    print("\nDetailed Results:")
    print("-" * 70)
    for i, r in enumerate(results):
        status = "✓" if r['is_correct'] else "✗"
        print(f"{i+1}. [{status}] {r['question'][:50]}...")
        print(f"   Expected: {r['expected']} | Got: {r['answer']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run DBpedia GraphRAG Benchmark")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to run")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    questions = BENCHMARK_QUESTIONS[:args.limit]
    results = run_benchmark(questions)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
