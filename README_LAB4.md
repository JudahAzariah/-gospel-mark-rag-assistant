# Lab 4 — Shipping It: Deployment, Guardrails, and Observability

This adds a public Streamlit front end to the existing Lab 2 Gospel-of-Mark RAG project.

## Copy into the existing project

Copy these files/folders into the same folder as the existing `main.py` and `corpus/`:

- `streamlit_app.py`
- `requirements.txt` (merge with existing dependencies if needed)
- `.streamlit/config.toml`
- `.gitignore` (merge these entries into the existing file; do not overwrite useful entries)
- `logs/` (the app creates `logs/requests.jsonl` after requests)

Do not replace `main.py`, `corpus/`, or the Lab 3 scripts.

## Local setup

Activate your existing virtual environment, then install the requirements:

```bash
pip install -r requirements.txt
```

Configure `.env` locally (never commit it):

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FREE_TIER=true
MAX_REQUESTS_PER_HOUR=5
RAG_CHUNK_SIZE=200
RAG_OVERLAP=40
RAG_TOP_K=3
```

Run locally:

```bash
streamlit run streamlit_app.py
```

The app caches the embedding/index resources in memory for the Streamlit process, writes one JSON object per line to `logs/requests.jsonl`, and does not store the user's question or answer in the log.

## Feature evidence

### Challenge 1

1. Public URL: deploy `streamlit_app.py` using Streamlit Community Cloud.
2. Secrets: keep `.env` and `.streamlit/secrets.toml` out of Git. Add `GEMINI_API_KEY` in the app's Streamlit Secrets settings.
3. Structured logs: submit at least 20 real requests after deployment. Use Operations & logs to show/download `requests.jsonl`. Each record includes timestamp, latency, token counts, outcome, provider/model, and estimated cost.

### Challenge 2

1. Rate limit: default is 5 requests per browser session per hour. Submit 5 requests, then a sixth; it should be blocked and logged as `rate_limited`.
2. Graceful degradation: select “Simulate temporary API outage” and submit a question. The app displays a friendly error rather than a traceback and logs `simulated_api_failure`. Uncheck it to resume normal requests.
3. Usage summary: Operations & logs shows request count, p50/p95 latency, total estimated cost, success count, API errors, and rate-limited requests.

## Deploy with Streamlit Community Cloud

1. Put this file and the existing `main.py`, `corpus/mark.txt`, and dependencies in a GitHub repository.
2. Before pushing, confirm `.env`, `.venv/`, `.rag_cache/`, and `.streamlit/secrets.toml` are ignored and are not already tracked.
3. Push the code to GitHub.
4. Sign in at https://share.streamlit.io/ using GitHub and create an app.
5. Choose the repository, branch, and `streamlit_app.py` as the entrypoint.
6. In the app's Advanced settings / Secrets, set:

```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "YOUR_KEY"
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FREE_TIER = "true"
MAX_REQUESTS_PER_HOUR = "5"
RAG_CHUNK_SIZE = "200"
RAG_OVERLAP = "40"
RAG_TOP_K = "3"
```

Use a Gemini model that is currently enabled for your AI Studio project. If your Gemini free-tier quota is exhausted, the app will still start and demonstrate graceful error handling, but normal answer generation requires available API quota.

## Important logging limitation

The JSONL log is written to the app's local filesystem. Free hosting filesystems may be ephemeral and can reset after restarts/redeployments. Generate the 20 evidence requests after the app is deployed and stable, download the JSONL file from the Operations & logs tab, and keep a copy for your LMS submission. Avoid redeploying after collecting evidence unless you have downloaded the logs.

## Verify no secrets are in Git

Run locally:

```bash
git status --short
```

Also inspect tracked files:

```bash
git ls-files
```

`.env` and `.streamlit/secrets.toml` must not appear. If a secret was ever committed, remove it from Git history where appropriate and rotate the key.
