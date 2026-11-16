"""Lab 2 RAG demo: retrieval, evaluation, grounded answers, sources, and refusal."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CORPUS_FILE = ROOT / "corpus" / "mark.txt"
CACHE_DIR = ROOT / ".rag_cache"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 40
DEFAULT_TOP_K = 3
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MIN_SIMILARITY = 0.32

load_dotenv(ROOT / ".env")


@dataclass
class Verse:
    reference: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    text: str
    references: list[str]
    word_count: int


# Five known-answer retrieval tests. Correct references come from the Gospel of Mark.
TEST_QUESTIONS = [
    {
        "question": "Who baptized Jesus?",
        "expected_reference": "Mark 1:9",
        "expected_answer": "John the Baptist",
    },
    {
        "question": "What did Jesus say to the storm?",
        "expected_reference": "Mark 4:39",
        "expected_answer": "Peace, be still",
    },
    {
        "question": "How many loaves were available when Jesus fed the five thousand?",
        "expected_reference": "Mark 6:38",
        "expected_answer": "Five loaves",
    },
    {
        "question": "Who betrayed Jesus?",
        "expected_reference": "Mark 14:10",
        "expected_answer": "Judas Iscariot",
    },
    {
        "question": "What was the name of the blind man Jesus healed near Jericho?",
        "expected_reference": "Mark 10:46",
        "expected_answer": "Bartimaeus",
    },
]

REFUSAL_QUESTIONS = [
    "What is California Baptist University's current tuition for its MS in Computer Science program?",
    "Who won the 2026 FIFA World Cup?",
]


def ensure_corpus() -> None:
    if CORPUS_FILE.exists() and CORPUS_FILE.stat().st_size > 1000:
        return
    print("Corpus file is missing. Downloading the Gospel of Mark...")
    from download_corpus import main as download_main

    download_main()


def load_verses() -> list[Verse]:
    ensure_corpus()
    verses: list[Verse] = []
    pattern = re.compile(r"^Mark\s+(\d+):(\d+)\s*\|\s*(.+)$")
    for line_number, line in enumerate(CORPUS_FILE.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(
                f"Unexpected corpus format at {CORPUS_FILE.name}:{line_number}: {line[:80]}"
            )
        chapter, verse, text = match.groups()
        verses.append(Verse(f"Mark {chapter}:{verse}", text.strip()))
    if len(verses) < 600:
        raise ValueError(f"Only loaded {len(verses)} verses; expected the full Gospel of Mark.")
    return verses


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def create_chunks(
    verses: list[Verse], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap_words: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Create approximately chunk_size-word chunks with overlap measured in words."""
    if chunk_size < 50:
        raise ValueError("Use a chunk size of at least 50 words for this lab.")
    if overlap_words < 0 or overlap_words >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    chunks: list[Chunk] = []
    start = 0
    while start < len(verses):
        end = start
        count = 0
        while end < len(verses) and count < chunk_size:
            count += word_count(verses[end].text)
            end += 1

        selected = verses[start:end]
        text = "\n".join(f"{v.reference} | {v.text}" for v in selected)
        references = [v.reference for v in selected]
        chunks.append(
            Chunk(
                chunk_id=f"MARK-{chunk_size}-{len(chunks) + 1:03d}",
                source_file="corpus/mark.txt",
                text=text,
                references=references,
                word_count=word_count(text),
            )
        )

        if end >= len(verses):
            break

        # Back up whole verses until the next chunk includes approximately the requested overlap.
        next_start = end
        overlap_count = 0
        while next_start > start and overlap_count < overlap_words:
            next_start -= 1
            overlap_count += word_count(verses[next_start].text)
        start = next_start

    return chunks


def get_embedding_model() -> SentenceTransformer:
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def vectorize_chunks(chunks: list[Chunk], model: SentenceTransformer, chunk_size: int) -> np.ndarray:
    """Create embeddings and persist the NumPy vectors for reuse."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    signature_payload = model_name + "\n" + "\n---CHUNK---\n".join(chunk.text for chunk in chunks)
    signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"mark_vectors_{chunk_size}.npz"

    if cache_file.exists():
        try:
            cached = np.load(cache_file, allow_pickle=False)
            if str(cached["signature"].item()) == signature:
                vectors = cached["vectors"]
                if vectors.shape[0] == len(chunks):
                    print(f"Loaded saved vectors: {cache_file.relative_to(ROOT)}")
                    return vectors.astype(np.float32, copy=False)
        except Exception:
            pass

    print(f"Embedding {len(chunks)} chunks (first run downloads the embedding model if needed)...")
    vectors = model.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.savez_compressed(cache_file, vectors=vectors, signature=np.array(signature))
    print(f"Saved {len(vectors)} vectors to: {cache_file.relative_to(ROOT)}")
    return vectors


def build_index(chunk_size: int = DEFAULT_CHUNK_SIZE, overlap_words: int = DEFAULT_OVERLAP):
    verses = load_verses()
    chunks = create_chunks(verses, chunk_size, overlap_words)
    model = get_embedding_model()
    vectors = vectorize_chunks(chunks, model, chunk_size)
    print("\nINGESTION SUMMARY")
    print(f"Corpus file : {CORPUS_FILE.relative_to(ROOT)}")
    print(f"Verses      : {len(verses)}")
    print(f"Chunk size  : {chunk_size} words (target)")
    print(f"Overlap     : {overlap_words} words (approximate, whole-verse boundaries)")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector shape: {vectors.shape}")
    return chunks, vectors, model


def retrieve(query: str, chunks: list[Chunk], vectors: np.ndarray, model: SentenceTransformer, k: int = DEFAULT_TOP_K):
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][: min(k, len(chunks))]
    return [(chunks[int(i)], float(scores[int(i)])) for i in top_indices]


def show_retrieval(results: list[tuple[Chunk, float]], title: str = "TOP-K RETRIEVED CHUNKS") -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for rank, (chunk, score) in enumerate(results, 1):
        refs = ", ".join(chunk.references[:8])
        if len(chunk.references) > 8:
            refs += ", ..."
        snippet = chunk.text.replace("\n", " ")
        if len(snippet) > 420:
            snippet = snippet[:420] + "..."
        print(f"\n[{rank}] {chunk.source_file} | {chunk.chunk_id} | cosine similarity={score:.4f}")
        print(f"References: {refs}")
        print(f"Snippet: {snippet}")


def evaluate(chunks: list[Chunk], vectors: np.ndarray, model: SentenceTransformer, k: int = DEFAULT_TOP_K) -> tuple[int, int]:
    print("\nCHALLENGE 1 — RETRIEVAL QUALITY EVALUATION")
    print(f"Top-k = {k}; correct reference must appear in at least one of the top-{k} chunks.\n")
    hits = 0
    report = ["# Lab 2 Retrieval Evaluation", "", f"Top-k: {k}", ""]
    for number, test in enumerate(TEST_QUESTIONS, 1):
        results = retrieve(test["question"], chunks, vectors, model, k)
        hit = any(test["expected_reference"] in chunk.references for chunk, _ in results)
        hits += int(hit)
        status = "HIT" if hit else "MISS"
        print(f"TEST {number}: {test['question']}")
        print(f"Expected evidence: {test['expected_reference']} ({test['expected_answer']})")
        print(f"Result: {status}")
        for rank, (chunk, score) in enumerate(results, 1):
            refs = ", ".join(chunk.references)
            print(f"  Top {rank}: {chunk.chunk_id} | score={score:.4f} | {refs}")
            snippet = chunk.text.replace("\n", " ")[:260]
            print(f"    {snippet}{'...' if len(chunk.text) > 260 else ''}")
        print()
        report.extend([
            f"## Test {number}: {test['question']}",
            f"- Expected evidence: {test['expected_reference']} ({test['expected_answer']})",
            f"- Result: **{status}**",
            "- Retrieved chunks:",
        ])
        for rank, (chunk, score) in enumerate(results, 1):
            report.append(
                f"  - Top {rank}: `{chunk.source_file}` / `{chunk.chunk_id}`, similarity `{score:.4f}`, refs: {', '.join(chunk.references)}"
            )
            report.append(f"    - Snippet: {chunk.text.replace(chr(10), ' ')[:260]}")
        report.append("")
    print(f"HIT RATE: {hits}/5 = {hits / 5:.0%}")
    report.extend([f"**Hit rate: {hits}/5 = {hits / 5:.0%}**", ""])
    (ROOT / "evaluation_results.md").write_text("\n".join(report), encoding="utf-8")
    print("Saved detailed results to evaluation_results.md")
    return hits, len(TEST_QUESTIONS)


def min_similarity() -> float:
    try:
        return float(os.getenv("RAG_MIN_SIMILARITY", str(DEFAULT_MIN_SIMILARITY)))
    except ValueError:
        return DEFAULT_MIN_SIMILARITY


def call_llm(question: str, context: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    system_prompt = (
        "You are a retrieval-grounded question-answering assistant. You MUST answer using only the supplied context. "
        "Do not use outside knowledge, even if you know the answer. If the context does not clearly support an answer, "
        "reply exactly: 'I don't know based on the provided corpus.' Do not guess or fill gaps. "
        "Keep the answer concise. Do not make up source references; source citations are displayed by the application."
    )
    user_prompt = f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}\n\nAnswer from this context only."

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing from .env.")
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            max_tokens=400,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()

    if provider != "openai":
        raise RuntimeError("LLM_PROVIDER must be 'openai' or 'anthropic'.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env. Add your API key before using chat/refusal mode.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def answer_question(question: str, chunks: list[Chunk], vectors: np.ndarray, model: SentenceTransformer, k: int = DEFAULT_TOP_K) -> None:
    print(f"\nQUESTION: {question}")
    results = retrieve(question, chunks, vectors, model, k)
    show_retrieval(results, "RETRIEVED EVIDENCE")
    threshold = min_similarity()
    best_score = results[0][1] if results else -1.0

    if best_score < threshold:
        answer = "I don't know based on the provided corpus. The retrieved passages are not sufficiently relevant."
        print("\nGROUNDING CHECK: REFUSED (similarity below configured threshold)")
    else:
        context = "\n\n".join(
            f"SOURCE: {chunk.source_file} | CHUNK: {chunk.chunk_id} | REFERENCES: {', '.join(chunk.references)}\n{chunk.text}"
            for chunk, _ in results
        )
        try:
            answer = call_llm(question, context)
        except Exception as exc:
            print(f"\nCould not call the answer model: {exc}")
            print("Retrieval and source evidence still worked; configure .env to enable answer generation.")
            return
        if not answer:
            answer = "I don't know based on the provided corpus."
        print("\nGROUNDING CHECK: CONTEXT SENT TO MODEL; ANSWER MUST BE BASED ON THAT CONTEXT")

    print("\nANSWER")
    print("------")
    print(answer)
    print("\nSOURCES (for verification)")
    print("--------------------------")
    for chunk, score in results:
        refs = ", ".join(chunk.references)
        snippet = chunk.text.replace("\n", " ")[:240]
        print(f"- {chunk.source_file} | {chunk.chunk_id} | similarity={score:.4f}")
        print(f"  References: {refs}")
        print(f"  Quote: {snippet}{'...' if len(chunk.text) > 240 else ''}")


def compare_sizes(model: SentenceTransformer, overlap: int = DEFAULT_OVERLAP, k: int = DEFAULT_TOP_K) -> None:
    verses = load_verses()
    print("\nCHUNK SIZE COMPARISON")
    print("=====================")
    print("This runs the same five known-answer tests against two different chunk sizes.")
    for size in (200, 300):
        chunks = create_chunks(verses, size, min(overlap, size - 1))
        vectors = vectorize_chunks(chunks, model, size)
        hits = 0
        for test in TEST_QUESTIONS:
            results = retrieve(test["question"], chunks, vectors, model, k)
            hit = any(test["expected_reference"] in chunk.references for chunk, _ in results)
            hits += int(hit)
            print(f"size={size:3d} | {test['question']} | {'HIT' if hit else 'MISS'}")
        print(f"size={size}: {len(chunks)} chunks, overlap={overlap} words, hit rate={hits}/5 ({hits / 5:.0%})\n")


def interactive_chat(chunks: list[Chunk], vectors: np.ndarray, model: SentenceTransformer, k: int) -> None:
    print("\nBible Study RAG Assistant — Ask about the Gospel of Mark.")
    print("Type 'exit' or 'quit' to stop. Type 'search: your question' for retrieval only.\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue
        if question.lower().startswith("search:"):
            query = question[len("search:"):].strip()
            show_retrieval(retrieve(query, chunks, vectors, model, k))
        else:
            answer_question(question, chunks, vectors, model, k)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 2: Retrieval-Augmented Generation demo")
    parser.add_argument(
        "command", nargs="?", default="chat",
        choices=["chat", "search", "evaluate", "compare", "refusal"],
        help="chat (default), search, evaluate, compare chunk sizes, or refusal test",
    )
    parser.add_argument("query", nargs="*", help="Question used with the search command")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    chunks, vectors, model = build_index(args.chunk_size, args.overlap)

    if args.command == "evaluate":
        evaluate(chunks, vectors, model, args.top_k)
    elif args.command == "compare":
        compare_sizes(model, args.overlap, args.top_k)
    elif args.command == "search":
        query = " ".join(args.query).strip()
        if not query:
            parser.error('Provide a search question, e.g. python main.py search "Who baptized Jesus?"')
        show_retrieval(retrieve(query, chunks, vectors, model, args.top_k))
    elif args.command == "refusal":
        for question in REFUSAL_QUESTIONS:
            answer_question(question, chunks, vectors, model, args.top_k)
            print("\n" + "-" * 80)
    else:
        interactive_chat(chunks, vectors, model, args.top_k)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
