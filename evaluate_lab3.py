"""
Lab 3 - Evaluation and Optimization
Built around the Lab 2 Gospel of Mark RAG system.

Run:
python evaluate_lab3.py --chunk-size 200 --overlap 40 --top-k 3

Development only:
python evaluate_lab3.py --chunk-size 200 --overlap 40 --top-k 3 --development-only

Supports:
- Google Gemini
- OpenAI
- Anthropic
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

load_dotenv(ROOT / ".env", override=True)

import main as rag


# ==========================================
# PROMPTS
# ==========================================

SYSTEM_PROMPT = (
    "You are a retrieval-grounded question-answering assistant. "
    "You MUST answer using only the supplied context. Do not use outside knowledge. "
    "If the context does not clearly support an answer, reply exactly: "
    "'I don't know based on the provided corpus.' Keep the answer concise."
)


JUDGE_PROMPT = """You are an evaluator for a retrieval-augmented QA system.

Grade the ACTUAL ANSWER against the QUESTION and EXPECTED ANSWER.

Score 1 if the actual answer is substantively correct,
complete enough, and supported.

Score 0 if it is incorrect, materially incomplete,
contradictory, or unsupported.

Return ONLY valid JSON in this exact shape:

{{"score": 0 or 1, "reason": "one concise sentence"}}

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

ACTUAL ANSWER:
{actual}
"""


DEFAULT_OPENAI_INPUT_PRICE = 0.15
DEFAULT_OPENAI_OUTPUT_PRICE = 0.60


# ==========================================
# LOAD TEST DATA
# ==========================================

def load_tests() -> list[dict[str, Any]]:

    path = ROOT / "test_set.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing test dataset: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize(text: str) -> str:

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text.lower()
    )

    return re.sub(r"\s+", " ", text).strip()


# ==========================================
# CLOSED ANSWER SCORING
# ==========================================

def closed_score(
    actual: str,
    test: dict[str, Any]
) -> tuple[int, str]:

    answer = normalize(actual)
    expected = normalize(test["expected_answer"])

    if expected and expected in answer:

        return (
            1,
            "The normalized expected answer was found."
        )

    for keyword in test.get("match_keywords", []):

        phrase = normalize(keyword)

        if phrase and phrase in answer:

            return (
                1,
                f"Accepted keyword/phrase match: {keyword!r}."
            )

    return (
        0,
        f"No expected phrase or keyword was found. "
        f"Expected={test['expected_answer']!r}; "
        f"actual={actual[:250]!r}"
    )


# ==========================================
# VERSE REFERENCE MATCHING
# ==========================================

def expand_expected_references(text: str) -> set[str]:
    """
    Expand references such as:
    Mark 5:22-23, 35-43

    Into individual verse references.
    """

    match = re.match(
        r"^\s*Mark\s+(\d+):(.+?)\s*$",
        text
    )

    if not match:
        return {text.strip()}

    chapter = int(match.group(1))

    expanded = set()

    for part in match.group(2).split(","):

        part = part.strip()

        if not part:
            continue

        try:

            if "-" in part:

                first, last = [
                    int(value.strip())
                    for value in part.split("-", 1)
                ]

                expanded.update(
                    f"Mark {chapter}:{verse}"
                    for verse in range(first, last + 1)
                )

            else:

                expanded.add(
                    f"Mark {chapter}:{int(part)}"
                )

        except ValueError:
            continue

    return expanded


def retrieval_hit_for_test(
    test: dict[str, Any],
    retrieved: list[tuple[Any, float]]
) -> bool:

    expected = expand_expected_references(
        test["expected_reference"]
    )

    found = {
        reference
        for chunk, _score in retrieved
        for reference in chunk.references
    }

    return bool(expected.intersection(found))


# ==========================================
# OPENAI API
# ==========================================

def call_openai(
    messages: list[dict[str, str]],
    model: str
) -> tuple[str, int, int]:

    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing from .env"
        )

    client = OpenAI(api_key=key)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages
    )

    usage = response.usage

    return (
        (response.choices[0].message.content or "").strip(),
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0)
    )


# ==========================================
# GEMINI API
# ==========================================

def call_gemini(
    messages: list[dict[str, str]],
    model: str
) -> tuple[str, int, int]:

    """
    Uses Google's OpenAI-compatible Gemini endpoint.
    """

    from openai import OpenAI

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from .env"
        )

    client = OpenAI(
        api_key=key,
        base_url=(
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        )
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages
    )

    usage = response.usage

    return (
        (response.choices[0].message.content or "").strip(),
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0)
    )


# ==========================================
# ANTHROPIC API
# ==========================================

def call_anthropic(
    system: str,
    user: str,
    model: str
) -> tuple[str, int, int]:

    import anthropic

    key = os.getenv("ANTHROPIC_API_KEY")

    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing from .env"
        )

    client = anthropic.Anthropic(api_key=key)

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system=system,
        messages=[
            {"role": "user", "content": user}
        ]
    )

    text = "\n".join(
        block.text
        for block in response.content
        if getattr(block, "type", "") == "text"
    ).strip()

    usage = response.usage

    return (
        text,
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0)
    )


# ==========================================
# PROVIDER CONFIGURATION
# ==========================================

def provider_and_model() -> tuple[str, str]:

    provider = os.getenv(
        "LLM_PROVIDER",
        "gemini"
    ).strip().lower()

    settings = {

        "gemini": (
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        ),

        "openai": (
            "OPENAI_MODEL",
            "gpt-4o-mini"
        ),

        "anthropic": (
            "ANTHROPIC_MODEL",
            "claude-3-5-haiku-latest"
        )
    }

    if provider not in settings:

        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={provider!r}. "
            "Choose gemini, openai, or anthropic."
        )

    env_name, default_model = settings[provider]

    return (
        provider,
        os.getenv(env_name, default_model)
    )


def provider_call(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    anthropic_system: str
) -> tuple[str, int, int]:

    if provider == "gemini":

        return call_gemini(messages, model)

    if provider == "openai":

        return call_openai(messages, model)

    return call_anthropic(
        anthropic_system,
        messages[-1]["content"],
        model
    )


# ==========================================
# ANSWER GENERATION
# ==========================================

def call_answer_model(
    question: str,
    context: str
) -> tuple[str, int, int]:

    provider, model = provider_and_model()

    user = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        "Answer from this context only."
    )

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": user
        }

    ]

    return provider_call(
        provider,
        model,
        messages,
        SYSTEM_PROMPT
    )


# ==========================================
# LLM AS A JUDGE
# ==========================================

def call_judge(
    question: str,
    expected: str,
    actual: str
) -> tuple[int, str, int, int]:

    provider, default_model = provider_and_model()

    model = os.getenv(
        "JUDGE_MODEL",
        default_model
    )

    user = JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        actual=actual
    )

    messages = [

        {
            "role": "system",
            "content": "Return only valid JSON. Do not add markdown."
        },

        {
            "role": "user",
            "content": user
        }

    ]

    raw, input_tokens, output_tokens = provider_call(
        provider,
        model,
        messages,
        "Return only valid JSON. Do not add markdown."
    )

    cleaned = raw.strip()

    if cleaned.startswith("```"):

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned
        )

    try:

        result = json.loads(cleaned)

        score = (
            1 if int(result.get("score", 0)) == 1 else 0
        )

        reason = str(
            result.get("reason", "No reason supplied.")
        ).strip()

    except (ValueError, TypeError, json.JSONDecodeError):

        match = re.search(
            r'"score"\s*:\s*([01])',
            raw
        )

        score = int(match.group(1)) if match else 0

        reason = (
            "Judge response could not be parsed: "
            + raw[:200]
        )

    return (
        score,
        reason,
        input_tokens,
        output_tokens
    )


# ==========================================
# API COST CALCULATION
# ==========================================

def price_for(
    provider: str
) -> tuple[float, float]:

    if provider == "gemini":

        free = os.getenv(
            "GEMINI_FREE_TIER",
            "true"
        ).strip().lower() in {
            "1", "true", "yes"
        }

        if free:
            return 0.0, 0.0

        return (
            float(os.getenv(
                "GEMINI_INPUT_PRICE_PER_1M", "0"
            )),
            float(os.getenv(
                "GEMINI_OUTPUT_PRICE_PER_1M", "0"
            ))
        )

    if provider == "openai":

        return (
            float(os.getenv(
                "OPENAI_INPUT_PRICE_PER_1M",
                DEFAULT_OPENAI_INPUT_PRICE
            )),
            float(os.getenv(
                "OPENAI_OUTPUT_PRICE_PER_1M",
                DEFAULT_OPENAI_OUTPUT_PRICE
            ))
        )

    return (
        float(os.getenv(
            "ANTHROPIC_INPUT_PRICE_PER_1M", "0.10"
        )),
        float(os.getenv(
            "ANTHROPIC_OUTPUT_PRICE_PER_1M", "0.50"
        ))
    )


def token_cost(
    input_tokens: int,
    output_tokens: int,
    provider: str
) -> float:

    input_price, output_price = price_for(provider)

    return (
        input_tokens / 1_000_000 * input_price
        + output_tokens / 1_000_000 * output_price
    )


# ==========================================
# BUILD RETRIEVED CONTEXT
# ==========================================

def build_context(
    retrieved: list[tuple[Any, float]]
) -> str:

    return "\n\n".join(

        f"SOURCE: {chunk.source_file} | "
        f"CHUNK: {chunk.chunk_id} | "
        f"REFERENCES: {', '.join(chunk.references)}\n"
        f"{chunk.text}"

        for chunk, _score in retrieved

    )


# ==========================================
# FAILURE CLASSIFICATION
# ==========================================

def classify_failure(
    retrieval_hit: bool,
    score: int,
    actual: str,
    test: dict[str, Any]
) -> str | None:

    if score == 1:
        return None

    if not retrieval_hit:
        return "retrieval_miss"

    if actual.strip().lower().startswith("i don't know"):
        return "context_ignored"

    if test["type"] == "open":
        return "incomplete_or_incorrect_answer"

    return "answer_or_format_error"


def save_json(
    path: Path,
    data: dict[str, Any]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


# ==========================================
# MAIN EVALUATION
# ==========================================

def run_evaluation(
    chunk_size: int,
    overlap: int,
    top_k: int,
    include_heldout: bool = True
) -> dict[str, Any]:

    tests = load_tests()

    if not include_heldout:

        tests = [
            test
            for test in tests
            if test["split"] == "development"
        ]

    if not tests:
        raise RuntimeError("No test questions were selected.")

    provider, answer_model = provider_and_model()

    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is missing from .env.")

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")

    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing from .env.")

    judge_model = os.getenv(
        "JUDGE_MODEL",
        answer_model
    )

    print("\n" + "=" * 60)
    print("LAB 3 EVALUATION")
    print("=" * 60)

    print(f"Chunk size  : {chunk_size}")
    print(f"Overlap     : {overlap}")
    print(f"Top-k       : {top_k}")
    print(f"Questions   : {len(tests)}")
    print(f"LLM provider: {provider}")
    print(f"Answer model: {answer_model}")
    print(f"Judge model : {judge_model}")

    if provider == "gemini":
        print("Gemini API key: configured (value hidden)")

    # Build the original Lab 2 retrieval index.

    chunks, vectors, embedding_model = rag.build_index(
        chunk_size,
        overlap
    )

    results = []
    latencies = []

    system_cost_total = 0.0
    judge_cost_total = 0.0

    name = (
        f"chunk{chunk_size}_"
        f"overlap{overlap}_"
        f"topk{top_k}"
    )

    results_dir = ROOT / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    partial_path = results_dir / f"{name}_partial.json"

    # ======================================
    # PROCESS EACH QUESTION
    # ======================================

    for number, test in enumerate(tests, 1):

        print(
            f"\n[{number}/{len(tests)}] "
            f"{test['question']}"
        )

        start = time.perf_counter()

        # Retrieval

        retrieved = rag.retrieve(
            test["question"],
            chunks,
            vectors,
            embedding_model,
            top_k
        )

        retrieval_hit = retrieval_hit_for_test(
            test,
            retrieved
        )

        context = build_context(retrieved)

        # Answer generation

        try:

            actual, answer_input, answer_output = call_answer_model(
                test["question"],
                context
            )

        except Exception as exc:

            actual = (
                f"[ANSWER API ERROR: "
                f"{type(exc).__name__}: {exc}]"
            )

            answer_input = 0
            answer_output = 0

        latency = time.perf_counter() - start

        latencies.append(latency)

        system_cost = token_cost(
            answer_input,
            answer_output,
            provider
        )

        system_cost_total += system_cost

        judge_input = 0
        judge_output = 0

        # Scoring

        if actual.startswith("[ANSWER API ERROR:"):

            score = 0
            reason = actual

        elif test["type"] == "closed":

            score, reason = closed_score(
                actual,
                test
            )

        else:

            try:

                score, reason, judge_input, judge_output = call_judge(
                    test["question"],
                    test["expected_answer"],
                    actual
                )

            except Exception as exc:

                # Save completed questions before stopping.

                save_json(
                    partial_path,
                    {
                        "complete": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "configuration": {
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            "top_k": top_k
                        },
                        "results_so_far": results,
                        "interrupted_at_question": test["id"]
                    }
                )

                raise RuntimeError(
                    f"Judge API failed on question {test['id']}. "
                    f"Partial results saved to {partial_path}. "
                    f"Error: {exc}"
                ) from exc

            judge_cost_total += token_cost(
                judge_input,
                judge_output,
                provider
            )

        # Failure classification

        failure = classify_failure(
            retrieval_hit,
            score,
            actual,
            test
        )

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

            "latency_seconds": round(latency, 4),

            "answer_input_tokens": answer_input,

            "answer_output_tokens": answer_output,

            "answer_cost_usd": round(system_cost, 8),

            "judge_input_tokens": judge_input,

            "judge_output_tokens": judge_output

        }

        results.append(item)

        # Console output

        print(
            f"  Retrieval: "
            f"{'HIT' if retrieval_hit else 'MISS'}"
        )

        print(f"  Score    : {score}")

        print(f"  Latency  : {latency:.3f}s")

        print(
            f"  Actual   : "
            f"{actual[:250].replace(chr(10), ' ')}"
        )

        if failure:
            print(f"  Failure  : {failure}")

        # Save progress after each successful question.

        save_json(
            partial_path,
            {
                "complete": False,
                "configuration": {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "top_k": top_k
                },
                "results_so_far": results,
                "completed_count": len(results)
            }
        )

    # ======================================
    # CALCULATE FINAL METRICS
    # ======================================

    quality = (
        sum(item["score"] for item in results)
        / len(results)
    )

    median_latency = statistics.median(latencies)

    average_system_cost = (
        system_cost_total / len(results)
    )

    average_judge_cost = (
        judge_cost_total / len(results)
    )

    failures = {}

    for item in results:

        bucket = item["failure_bucket"]

        if bucket:

            failures[bucket] = (
                failures.get(bucket, 0) + 1
            )

    # ======================================
    # BUILD FINAL REPORT
    # ======================================

    output = {

        "complete": True,

        "configuration": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k
        },

        "models": {

            "embedding_model": os.getenv(
                "EMBEDDING_MODEL",
                rag.DEFAULT_EMBEDDING_MODEL
            ),

            "answer_provider": provider,

            "answer_model": answer_model,

            "judge_model": judge_model

        },

        "metrics": {

            "question_count": len(results),

            "quality_score": round(quality, 6),

            "quality_percent": round(
                quality * 100,
                2
            ),

            "median_latency_seconds": round(
                median_latency,
                4
            ),

            "total_system_cost_usd": round(
                system_cost_total,
                8
            ),

            "system_cost_per_query_usd": round(
                average_system_cost,
                8
            ),

            "total_judge_cost_usd": round(
                judge_cost_total,
                8
            ),

            "judge_cost_per_query_usd": round(
                average_judge_cost,
                8
            ),

            "evaluation_cost_including_judge_usd": round(
                system_cost_total + judge_cost_total,
                8
            )

        },

        "failure_buckets": failures,

        "results": results

    }

    final_path = results_dir / f"{name}.json"

    save_json(
        final_path,
        output
    )

    if partial_path.exists():
        partial_path.unlink()

    # ======================================
    # DISPLAY SUMMARY
    # ======================================

    print("\n" + "-" * 60)

    print(f"QUALITY SCORE       : {quality:.1%}")

    print(
        f"MEDIAN LATENCY      : "
        f"{median_latency:.3f}s"
    )

    print(
        f"COST / QUERY        : "
        f"${average_system_cost:.6f}"
    )

    print(
        f"JUDGE COST / QUERY  : "
        f"${average_judge_cost:.6f}"
    )

    print("\nFAILURE BUCKETS:")

    if failures:

        for bucket, count in sorted(
            failures.items(),
            key=lambda pair: (-pair[1], pair[0])
        ):

            print(f"  {bucket}: {count}")

    else:

        print("  None")

    if (
        provider == "gemini"
        and os.getenv("GEMINI_FREE_TIER", "true").lower()
        in {"1", "true", "yes"}
    ):

        print(
            "NOTE: Gemini free-tier API usage "
            "is reported at $0.00."
        )

    print(
        f"\nSaved: {final_path.relative_to(ROOT)}"
    )

    return output


# ==========================================
# COMMAND LINE
# ==========================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Lab 3 RAG evaluation harness"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=40
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3
    )

    parser.add_argument(
        "--development-only",
        action="store_true"
    )

    args = parser.parse_args()

    if args.chunk_size < 50:
        parser.error("--chunk-size must be at least 50.")

    if args.overlap < 0 or args.overlap >= args.chunk_size:
        parser.error(
            "--overlap must be non-negative and smaller than chunk size."
        )

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    run_evaluation(
        args.chunk_size,
        args.overlap,
        args.top_k,
        include_heldout=not args.development_only
    )


if __name__ == "__main__":
    main()