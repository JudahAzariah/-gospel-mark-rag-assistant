# Lab 3 — Evaluation and Optimization

This folder extends the Lab 2 Gospel-of-Mark RAG system with a reproducible evaluation harness.

## Files

- `test_set.json` — 25 questions: 20 development questions + 5 held-out questions.
- `evaluate_lab3.py` — runs the RAG system against the test set and records quality, latency, token usage, cost, and failures.
- `failure_analysis.py` — converts a result JSON file into a failure-bucket report.
- `compare_configs.py` — runs four controlled configurations and creates a comparison table.
- `configs/` — experiment settings.
- `results/` — generated evaluation artifacts.

## Before running

1. Put these files in the same folder as your existing Lab 2 `main.py`.
2. Keep your existing `.env` in place. **Do not submit `.env`.**
3. Make sure your Lab 2 dependencies are installed:
   `pip install -r requirements.txt`

The evaluator uses the same `LLM_PROVIDER`, `OPENAI_MODEL`, `ANTHROPIC_MODEL`, and `EMBEDDING_MODEL` variables used by Lab 2.

## Cost settings

For the default Lab 2 model (`gpt-4o-mini`), the evaluator defaults to:
- input: $0.15 / 1M tokens
- output: $0.60 / 1M tokens

If you use another model, set these in `.env`:

```text
OPENAI_INPUT_PRICE_PER_1M=...
OPENAI_OUTPUT_PRICE_PER_1M=...
```

For Anthropic:

```text
ANTHROPIC_INPUT_PRICE_PER_1M=...
ANTHROPIC_OUTPUT_PRICE_PER_1M=...
```

Use the provider's current pricing page when setting these values.

## Challenge 1

Run the baseline evaluation:

```bash
python evaluate_lab3.py --chunk-size 200 --overlap 40 --top-k 3
```

This evaluates all 25 questions.

For development-only evaluation:

```bash
python evaluate_lab3.py --chunk-size 200 --overlap 40 --top-k 3 --development-only
```

Then generate failure analysis:

```bash
python failure_analysis.py results/chunk200_overlap40_topk3.json
```

## Challenge 2

Run:

```bash
python compare_configs.py
```

This evaluates:

1. Baseline: chunk 200, top-k 3
2. Chunk 300: chunk 300, top-k 3
3. Top-K 5: chunk 200, top-k 5
4. Top-K 2: chunk 200, top-k 2

The script changes one variable at a time.

The comparison is saved to:

```text
results/comparison.md
results/comparison.json
```

## Held-out evaluation

After choosing the configuration based on the development results, run it on all 25 questions. For example:

```bash
python evaluate_lab3.py --chunk-size 300 --overlap 40 --top-k 3
```

The last five questions are marked `heldout` in `test_set.json`.

## What to show in the video

1. `test_set.json`: 25 questions, 5 held out.
2. `evaluate_lab3.py`: automated scoring and latency/cost measurement.
3. `failure_analysis.py`: 2–4 named failure buckets with counts.
4. `compare_configs.py`: controlled experiments.
5. `results/comparison.md`: quality, median latency, and cost/query.
6. Final held-out run for the selected configuration.

Do not claim a required 10-percentage-point quality improvement or 30% latency/cost reduction unless your measured results actually satisfy the rubric.
