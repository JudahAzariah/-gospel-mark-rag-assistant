"""Public Streamlit front end for the Lab 2 Gospel-of-Mark RAG application.

Features: JSONL request logging, per-session hourly rate limit, graceful API
failure handling, and an operations/usage dashboard.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "requests.jsonl"
LOG_LOCK = threading.Lock()

st.set_page_config(page_title="Gospel of Mark RAG Assistant", page_icon="📖", layout="wide")

# Local runs read .env. On Streamlit Community Cloud, values are entered in
# App settings -> Secrets and copied into the process environment here.
load_dotenv(ROOT / ".env", override=False)
try:
    for _key in (
        "LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
        "RAG_MIN_SIMILARITY", "MAX_REQUESTS_PER_HOUR", "GEMINI_FREE_TIER",
        "GEMINI_INPUT_PRICE_PER_1M", "GEMINI_OUTPUT_PRICE_PER_1M",
        "OPENAI_INPUT_PRICE_PER_1M", "OPENAI_OUTPUT_PRICE_PER_1M",
        "ANTHROPIC_INPUT_PRICE_PER_1M", "ANTHROPIC_OUTPUT_PRICE_PER_1M",
    ):
        if _key in st.secrets and str(st.secrets[_key]):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    # st.secrets is absent during local runs unless configured.
    pass

# Import after environment/secrets are populated. main.py contains the existing
# chunking, embeddings, retrieval, and similarity-threshold functions.
import main as rag

SYSTEM_PROMPT = (
    "You are a retrieval-grounded question-answering assistant. Answer only "
    "from the supplied Gospel of Mark context. Do not use outside knowledge. "
    "If context does not support the answer, say: I don't know based on the "
    "provided corpus. Keep the answer concise."
)


def env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float = 0.0) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def append_log(record: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with LOG_LOCK:
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_logs() -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    records: list[dict[str, Any]] = []
    with LOG_LOCK:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        except json.JSONDecodeError:
            continue
    return records


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, with p in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int((p / 100.0) * len(ordered) + 0.999999))
    return float(ordered[min(rank - 1, len(ordered) - 1)])


def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    if provider == "gemini" and bool_env("GEMINI_FREE_TIER", True):
        return 0.0
    if provider == "gemini":
        in_price = env_float("GEMINI_INPUT_PRICE_PER_1M")
        out_price = env_float("GEMINI_OUTPUT_PRICE_PER_1M")
    elif provider == "openai":
        in_price = env_float("OPENAI_INPUT_PRICE_PER_1M", 0.15)
        out_price = env_float("OPENAI_OUTPUT_PRICE_PER_1M", 0.60)
    else:
        in_price = env_float("ANTHROPIC_INPUT_PRICE_PER_1M", 0.10)
        out_price = env_float("ANTHROPIC_OUTPUT_PRICE_PER_1M", 0.50)
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def provider_model() -> tuple[str, str]:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    models = {
        "gemini": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
    }
    if provider not in models:
        raise RuntimeError("Unsupported LLM_PROVIDER. Choose gemini, openai, or anthropic.")
    return provider, models[provider]


@st.cache_resource(show_spinner="Loading the Gospel of Mark and embedding model...")
def load_rag_index():
    return rag.build_index(
        int(os.getenv("RAG_CHUNK_SIZE", "200")),
        int(os.getenv("RAG_OVERLAP", "40")),
    )


def make_context(retrieved) -> str:
    return "\n\n".join(
        f"SOURCE: {chunk.source_file} | CHUNK: {chunk.chunk_id} | "
        f"REFERENCES: {', '.join(chunk.references)}\n{chunk.text}"
        for chunk, _score in retrieved
    )


def call_gemini(question: str, context: str, model: str) -> tuple[str, int, int]:
    from openai import OpenAI

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("The AI service is not configured. Please contact the app administrator.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=35.0,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}\n\nAnswer from this context only."},
        ],
    )
    usage = response.usage
    return (
        (response.choices[0].message.content or "").strip(),
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def call_openai(question: str, context: str, model: str) -> tuple[str, int, int]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("The AI service is not configured. Please contact the app administrator.")
    client = OpenAI(api_key=api_key, timeout=35.0, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}\n\nAnswer from this context only."},
        ],
    )
    usage = response.usage
    return (
        (response.choices[0].message.content or "").strip(),
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def call_anthropic(question: str, context: str, model: str) -> tuple[str, int, int]:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("The AI service is not configured. Please contact the app administrator.")
    client = anthropic.Anthropic(api_key=api_key, timeout=35.0, max_retries=0)
    response = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}\n\nAnswer from this context only."}],
    )
    text = "\n".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    usage = response.usage
    return text, int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def generate_answer(question: str, context: str, provider: str, model: str):
    if provider == "gemini":
        return call_gemini(question, context, model)
    if provider == "openai":
        return call_openai(question, context, model)
    return call_anthropic(question, context, model)


def session_fingerprint() -> str:
    raw = st.session_state.get("client_session_id", "unknown")
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:12]


def record_request(
    started: float,
    outcome: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    provider: str = "unknown",
    model: str = "unknown",
    error_type: str | None = None,
) -> dict[str, Any]:
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    cost = estimate_cost(provider, input_tokens, output_tokens) if provider != "unknown" else 0.0
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
        "session_hash": session_fingerprint(),
        "latency_ms": latency_ms,
        "prompt_tokens": int(input_tokens),
        "completion_tokens": int(output_tokens),
        "total_tokens": int(input_tokens + output_tokens),
        "outcome": outcome,
        "provider": provider,
        "model": model,
        "estimated_cost_usd": round(cost, 8),
        "error_type": error_type,
    }
    append_log(record)
    return record


def process_question(question: str, simulate_failure: bool) -> None:
    started = time.perf_counter()
    provider, model = provider_model()
    now = time.time()
    limit = env_int("MAX_REQUESTS_PER_HOUR", 5)
    history = st.session_state.get("request_times", [])
    history = [stamp for stamp in history if now - stamp < 3600]

    if len(history) >= limit:
        record_request(started, "rate_limited", provider=provider, model=model)
        st.error(f"Request limit reached: {limit} questions per hour for this browser session. Please try again later.")
        return

    history.append(now)
    st.session_state.request_times = history

    if simulate_failure:
        record_request(started, "simulated_api_failure", provider=provider, model=model, error_type="SimulatedProviderTimeout")
        st.error("The AI service is temporarily unavailable. Your request was not completed. Please try again later.")
        return

    try:
        chunks, vectors, embedding_model = load_rag_index()
        retrieved = rag.retrieve(question, chunks, vectors, embedding_model, int(os.getenv("RAG_TOP_K", "3")))
        best_score = retrieved[0][1] if retrieved else -1.0
        threshold = rag.min_similarity()

        if best_score < threshold:
            answer = "I don't know based on the provided corpus. The retrieved passages are not sufficiently relevant."
            record_request(started, "grounded_refusal", provider=provider, model=model)
            st.info(answer)
            return

        context = make_context(retrieved)
        answer, input_tokens, output_tokens = generate_answer(question, context, provider, model)
        if not answer:
            answer = "I don't know based on the provided corpus."
        record_request(started, "success", input_tokens, output_tokens, provider, model)

        st.markdown("### Answer")
        st.write(answer)
        with st.expander("Retrieved sources"):
            for chunk, score in retrieved:
                st.markdown(f"**{chunk.chunk_id}** — similarity `{score:.4f}`")
                st.caption(", ".join(chunk.references))
                st.write(chunk.text[:700])

    except Exception as exc:
        # Do not reveal stack traces, API keys, or raw provider error payloads.
        error_name = type(exc).__name__
        record_request(started, "api_error", provider=provider, model=model, error_type=error_name)
        st.error("The AI service is temporarily unavailable or its usage quota has been reached. Please try again later.")
        st.caption(f"Operational status: {error_name}. No credentials or stack trace are shown.")


# =========================================
# USER INTERFACE
# =========================================

st.title("📖 Gospel of Mark — RAG Assistant")
st.caption("Ask questions about the Gospel of Mark. Answers are grounded in retrieved passages.")

if "client_session_id" not in st.session_state:
    st.session_state.client_session_id = str(uuid.uuid4())
if "request_times" not in st.session_state:
    st.session_state.request_times = []

with st.sidebar:
    st.subheader("Application controls")
    st.write(f"Provider: **{os.getenv('LLM_PROVIDER', 'gemini')}**")
    st.write(f"Per-session limit: **{env_int('MAX_REQUESTS_PER_HOUR', 5)} requests/hour**")
    st.caption("Requests and operational metrics are logged. Questions and answers are not stored in the log.")

tab_ask, tab_ops = st.tabs(["Ask a question", "Operations & logs"])

with tab_ask:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area("Your question", placeholder="Who baptized Jesus?", max_chars=1000)
        simulate_failure = st.checkbox("Simulate temporary API outage (for graceful-degradation demo)")
        submitted = st.form_submit_button("Submit question", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            process_question(question.strip(), simulate_failure)

with tab_ops:
    st.subheader("Usage summary")
    records = read_logs()
    latencies = [float(r.get("latency_ms", 0)) for r in records]
    total_cost = sum(float(r.get("estimated_cost_usd", 0)) for r in records)
    request_count = len(records)
    success_count = sum(r.get("outcome") == "success" for r in records)
    error_count = sum(r.get("outcome") in {"api_error", "simulated_api_failure"} for r in records)
    limited_count = sum(r.get("outcome") == "rate_limited" for r in records)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests", request_count)
    c2.metric("Successful", success_count)
    c3.metric("p50 latency", f"{percentile(latencies, 50):.0f} ms")
    c4.metric("p95 latency", f"{percentile(latencies, 95):.0f} ms")

    c5, c6, c7 = st.columns(3)
    c5.metric("Estimated total cost", f"${total_cost:.6f}")
    c6.metric("API errors", error_count)
    c7.metric("Rate-limited", limited_count)

    st.caption("Metrics are calculated from this application's JSONL request log. Free-tier Gemini usage is estimated at $0; it is not a paid-tier cost comparison.")

    if records:
        st.dataframe(records[-50:][::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "Download JSONL request logs",
            data=LOG_FILE.read_bytes(),
            file_name="requests.jsonl",
            mime="application/jsonl",
        )
    else:
        st.info("No requests have been logged yet. Submit real questions in the Ask a question tab.")

st.divider()
st.caption("Lab 4 demonstration build — request guardrails, graceful failure handling, and observability.")
