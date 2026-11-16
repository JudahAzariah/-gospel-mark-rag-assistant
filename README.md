# Lab 2 — Retrieval: Grounding a Model in Your Own Data

A terminal-based RAG demo built around the King James Version Gospel of Mark. It covers ingestion/chunking, embeddings, top-k similarity search, five-question retrieval evaluation, grounded LLM answers with local source citations, and out-of-corpus refusal tests.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`. Set `LLM_PROVIDER=openai` and add your OpenAI API key, or choose `LLM_PROVIDER=anthropic` and add your Anthropic API key. Never submit `.env` or include an API key in your video.

The first run downloads the public-domain KJV Gospel of Mark JSON from a public GitHub repository, converts it to `corpus/mark.txt`, and downloads the sentence-transformer embedding model. An internet connection is needed for first-time setup.

## Commands

```powershell
# Interactive grounded Q&A
python main.py

# Retrieve top-k chunks for an arbitrary query without generating an answer
python main.py search "What did Jesus say to the storm?"

# Run the five known-answer retrieval tests and save real results
python main.py evaluate

# Compare 200-word and 300-word target chunks
python main.py compare

# Run two out-of-corpus refusal tests
python main.py refusal

# Change k, chunk size, or overlap
python main.py search "Who baptized Jesus?" --top-k 5
python main.py chat --chunk-size 300 --overlap 50
```

## Implementation notes

- Target chunk size: **200 words**; overlap: **40 words**. Chunk boundaries respect verse boundaries, so real chunk sizes are approximate.
- A second evaluation runs at a target chunk size of 300 words to meet the requirement to try at least two configurations.
- `sentence-transformers/all-MiniLM-L6-v2` embeds every chunk. The vectors are normalized, persisted as NumPy arrays in `.rag_cache/`, and queried with dot-product cosine similarity.
- `evaluate` prints every test's retrieved chunks and calculates the observed hit rate; it writes the run's actual results to `evaluation_results.md`. Do not claim a hit rate until you have run it.
- The answering prompt restricts the model to retrieved context. An additional configurable cosine-similarity threshold refuses very low-relevance questions. Review both refusal outputs and adjust `RAG_MIN_SIMILARITY` only if the actual results justify it.
- Sources are printed with file name, chunk ID, verse references, similarity score, and a quote for verification.

## Public-data attribution

The text is the King James Version, public domain in the United States. The downloader tries the per-book JSON repository at `https://github.com/nolanbaxter/kjv-bible` and falls back to `https://github.com/thiagobodruk/bible`.
