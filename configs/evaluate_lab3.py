"""Lab 3 evaluation harness for the Lab 2 Gospel-of-Mark RAG system.

Run:
    python evaluate_lab3.py --chunk-size 200 --overlap 40 --top-k 3

The script imports the existing main.py so your Lab 2 retrieval pipeline remains
the source of truth. It evaluates all 25 questions, measures end-to-end answer
latency, records production-model token cost, and uses an LLM judge for open-ended
items.

No API keys are stored in this file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

import main as rag


SYSTEM_PROMPT = (
    "You are a retrieval-grounded question-answering assistant. "
    "You MUST answer using only the supplied context. "
    "Do not use outside knowledge. If the context does not clearly support "
    "an answer, reply exactly: 'I don't know based on the provided corpus.' "
    "Keep the answer concise."
)

JUDGE_PROMPT = """You are an evaluator for a retrieval-augmented QA system.

Grade the ACTUAL ANSWER against the QUESTION and EXPECTED ANSWER.
A score of 1 means the actual answer is substantively correct and answers
the question using information supported by the expected answer.
A score of 0 means it is incorrect, materially incomplete, contradictory,
or unsupported.

Return ONLY valid JSON in this exact shape:
{{"score": 0 or 1, "reason": "one concise sentence"}}

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

ACTUAL ANSWER:
{actual}
"""

# Default prices match the Lab 2 default model, gpt-4o-mini. Update these
# environment variables if your .env uses another model.
DEFAULT_OPENAI_INPUT_PRICE = 0.15
DEFAULT_OPENAI_OUTPUT_PRICE = 0.60

def load_tests() -> list[dict[str, Any]]:
    return json.loads((ROOT / "test_set.json").read_text(encoding="utf-8"))

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def closed_score(actual: str, test: dict[str, Any]) -> tuple[int, str]:
    normalized = normalize(actual)
    expected = normalize(test["expected_answer"])

    # First accept the full expected answer.
    if expected and expected in normalized:
        return 1, "The normalized expected answer was found in the response."

    # Then accept any explicitly supplied keyword/phrase.
    keywords = test.get("match_keywords", [])
    normalized_keywords = [normalize(k) for k in keywords if normalize(k)]
    hits = [kw for kw in normalized_keywords if kw in normalized]

    if hits:
        return 1, f"Accepted keyword/phrase match: {hits[0]!r}."

    return 0, (
        f"No expected phrase or keyword was found. "
        f"Expected={test['expected_answer']!r}; actual={actual[:250]!r}"
    )

def call_openai(messages: list[dict[str, str]], model: str) -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
    )
    text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return text, int(getattr(usage, "prompt_tokens", 0)), int(getattr(usage, "completion_tokens", 0))

def call_gemini(messages: list[dict[str, str]], model: str) -> tuple[str, int, int]:
    """Call Gemini through Google's OpenAI-compatible Chat Completions endpoint."""
    from openai import OpenAI
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env. Create a key in Google AI Studio.")
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
    )
    text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return text, int(getattr(usage, "prompt_tokens", 0)), int(getattr(usage, "completion_tokens", 0))

def call_anthropic(system: str, user: str, model: str) -> tuple[str, int, int]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "\n".join(
        block.text for block in response.content
        if getattr(block, "type", "") == "text"
    ).strip()
    usage = response.usage
    return text, int(getattr(usage, "input_tokens", 0)), int(getattr(usage, "output_tokens", 0))

def call_answer_model(question: str, context: str) -> tuple[str, int, int]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    user_content = (
        f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}"
        "\n\nAnswer from this context only."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return call_openai(messages, model)
    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        return call_gemini(messages, model)
    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        return call_anthropic(SYSTEM_PROMPT, user_content, model)
    raise RuntimeError("LLM_PROVIDER must be 'openai', 'gemini', or 'anthropic'.")

def call_judge(question: str, expected: str, actual: str) -> tuple[int, str, int, int]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    user = JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)
    messages = [
        {"role": "system", "content": "Return only valid JSON. Do not add markdown."},
        {"role": "user", "content": user},
    ]

    if provider == "openai":
        model = os.getenv("JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        raw, inp, out = call_openai(messages, model)
    elif provider == "gemini":
        model = os.getenv("JUDGE_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
        raw, inp, out = call_gemini(messages, model)
    else:
        model = os.getenv("JUDGE_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"))
        raw, inp, out = call_anthropic(
            "Return only valid JSON. Do not add markdown.",
            user,
            model,
        )

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        score = 1 if int(data.get("score", 0)) == 1 else 0
        reason = str(data.get("reason", "No reason supplied.")).strip()
    except Exception:
        match = re.search(r'"score"\s*:\s*([01])', raw)
        score = int(match.group(1)) if match else 0
        reason = "Judge response could not be parsed cleanly: " + raw[:200]
    return score, reason, inp, out

def price_for(provider: str, model: str) -> tuple[float, float]:
    if provider == "openai":
        in_price = float(os.getenv("OPENAI_INPUT_PRICE_PER_1M", DEFAULT_OPENAI_INPUT_PRICE))
        out_price = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_1M", DEFAULT_OPENAI_OUTPUT_PRICE))
        return in_price, out_price
    if provider == "gemini":
        # In the Gemini free tier, standard text input/output are priced at $0.
        # Set GEMINI_FREE_TIER=false when using a paid Gemini API project.
        free_tier = os.getenv("GEMINI_FREE_TIER", "true").strip().lower() in {"1", "true", "yes"}
        if free_tier:
            return 0.0, 0.0
        in_price = float(os.getenv("GEMINI_INPUT_PRICE_PER_1M", "0.75"))
        out_price = float(os.getenv("GEMINI_OUTPUT_PRICE_PER_1M", "3.75"))
        return in_price, out_price
    in_price = float(os.getenv("ANTHROPIC_INPUT_PRICE_PER_1M", "0.10"))
    out_price = float(os.getenv("ANTHROPIC_OUTPUT_PRICE_PER_1M", "0.50"))
    return in_price, out_price

def token_cost(input_tokens: int, output_tokens: int, provider: str, model: str) -> float:
    in_price, out_price = price_for(provider, model)
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price

def build_context(results: list[tuple[Any, float]]) -> str:
    return "\n\n".join(
        f"SOURCE: {chunk.source_file} | CHUNK: {chunk.chunk_id} | "
        f"REFERENCES: {', '.join(chunk.references)}\n{chunk.text}"
        for chunk, _ in results
    )

def classify_failure(retrieval_hit: bool, score: int, actual: str, test: dict[str, Any]) -> str | None:
    if score == 1:
        return None
    if not retrieval_hit:
        return "retrieval_miss"
    if actual.strip().lower().startswith("i don't know"):
        return "context_ignored"
    if test["type"] == "open":
        return "incomplete_or_incorrect_answer"
    return "answer_or_format_error"

def run_evaluation(chunk_size: int, overlap: int, top_k: int, include_heldout: bool = True) -> dict[str, Any]:
    tests = load_tests()
    if not include_heldout:
        tests = [t for t in tests if t["split"] == "development"]

    print("\n" + "=" * 60)
    print("LAB 3 EVALUATION")
    print("=" * 60)
    print(f"Chunk size : {chunk_size}")
    print(f"Overlap    : {overlap}")
    print(f"Top-k      : {top_k}")
    print(f"Questions  : {len(tests)}")

    chunks, vectors, embedding_model = rag.build_index(chunk_size, overlap)

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model_env = {
        "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
        "gemini": ("GEMINI_MODEL", "gemini-3.8-flash"),
        "anthropic": ("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
    }
    env_name, default_model = model_env.get(provider, ("OPENAI_MODEL", "gpt-4o-mini"))
    answer_model = os.getenv(env_name, default_model)

    results = []
    latencies = []
    total_system_cost = 0.0
    total_judge_cost = 0.0

    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test['question']}")
        start = time.perf_counter()

        retrieved = rag.retrieve(test["question"], chunks, vectors, embedding_model, top_k)
        retrieval_hit = any(test["expected_reference"] in chunk.references for chunk, _ in retrieved)
        context = build_context(retrieved)

        try:
            actual, input_tokens, output_tokens = call_answer_model(test["question"], context)
        except Exception as exc:
            actual = f"[ERROR calling answer model: {exc}]"
            input_tokens = output_tokens = 0

        answer_latency = time.perf_counter() - start
        latencies.append(answer_latency)

        system_cost = token_cost(input_tokens, output_tokens, provider, answer_model)
        total_system_cost += system_cost

        if test["type"] == "closed":
            score, reason = closed_score(actual, test)
            judge_input = judge_output = 0
        else:
            score, reason, judge_input, judge_output = call_judge(
                test["question"], test["expected_answer"], actual
            )
            judge_model = os.getenv("JUDGE_MODEL", answer_model)
            total_judge_cost += token_cost(judge_input, judge_output, provider, judge_model)

        failure = classify_failure(retrieval_hit, score, actual, test)

        item = {
            "id": test["id"],
            "question": test["question"],
            "split": test["split"],
            "type": test["type"],
            "expected_answer": test["expected_answer"],
            "expected_reference": test["expected_reference"],
            "retrieval_hit": retrieval_hit,
            "actual_answer": actual,
            "score": score,
            "reason": reason,
            "failure_bucket": failure,
            "latency_seconds": round(answer_latency, 4),
            "answer_input_tokens": input_tokens,
            "answer_output_tokens": output_tokens,
            "answer_cost_usd": round(system_cost, 8),
            "judge_input_tokens": judge_input,
            "judge_output_tokens": judge_output,
        }
        results.append(item)

        print(f"  Retrieval: {'HIT' if retrieval_hit else 'MISS'}")
        print(f"  Score    : {score}")
        print(f"  Latency  : {answer_latency:.3f}s")
        print(f"  Actual   : {actual[:250].replace(chr(10), ' ')}")
        if failure:
            print(f"  Failure  : {failure}")

    quality = sum(r["score"] for r in results) / len(results)
    median_latency = statistics.median(latencies)
    avg_system_cost = total_system_cost / len(results)
    failures = {}
    for r in results:
        if r["failure_bucket"]:
            failures[r["failure_bucket"]] = failures.get(r["failure_bucket"], 0) + 1

    output = {
        "configuration": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
        },
        "models": {
            "embedding_model": os.getenv("EMBEDDING_MODEL", rag.DEFAULT_EMBEDDING_MODEL),
            "answer_provider": provider,
            "answer_model": answer_model,
            "judge_model": os.getenv("JUDGE_MODEL", answer_model),
        },
        "metrics": {
            "question_count": len(results),
            "quality_score": round(quality, 6),
            "quality_percent": round(quality * 100, 2),
            "median_latency_seconds": round(median_latency, 4),
            "total_system_cost_usd": round(total_system_cost, 8),
            "system_cost_per_query_usd": round(avg_system_cost, 8),
            "total_judge_cost_usd": round(total_judge_cost, 8),
            "evaluation_cost_including_judge_usd": round(total_system_cost + total_judge_cost, 8),
        },
        "failure_buckets": failures,
        "results": results,
    }

    safe_name = f"chunk{chunk_size}_overlap{overlap}_topk{top_k}"
    out_file = ROOT / "results" / f"{safe_name}.json"
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n" + "-" * 60)
    print(f"QUALITY SCORE       : {quality:.1%}")
    print(f"MEDIAN LATENCY      : {median_latency:.3f}s")
    print(f"COST / QUERY        : ${avg_system_cost:.6f}")
    print(f"JUDGE COST / QUERY  : ${total_judge_cost / len(results):.6f}")
    print("FAILURE BUCKETS     :")
    for bucket, count in sorted(failures.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {bucket}: {count}")
    print(f"\nSaved: {out_file.relative_to(ROOT)}")

    return output

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        args.chunk_size,
        args.overlap,
        args.top_k,
        include_heldout=not args.development_only,
    )

if __name__ == "__main__":
    main()
