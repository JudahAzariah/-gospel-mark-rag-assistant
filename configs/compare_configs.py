"""Run multiple Lab 3 configurations and create a comparison table."""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CONFIGS = [
    {"name": "Baseline", "chunk_size": 200, "overlap": 40, "top_k": 3},
    {"name": "Chunk 300", "chunk_size": 300, "overlap": 40, "top_k": 3},
    {"name": "Top-K 5", "chunk_size": 200, "overlap": 40, "top_k": 5},
    {"name": "Top-K 2", "chunk_size": 200, "overlap": 40, "top_k": 2},
]

def run_one(cfg: dict) -> dict:
    cmd = [
        sys.executable, str(ROOT / "evaluate_lab3.py"),
        "--chunk-size", str(cfg["chunk_size"]),
        "--overlap", str(cfg["overlap"]),
        "--top-k", str(cfg["top_k"]),
        "--development-only",
    ]
    print("\n" + "#" * 70)
    print(f"RUNNING: {cfg['name']}")
    print("#" * 70)
    subprocess.run(cmd, check=True)
    filename = f"chunk{cfg['chunk_size']}_overlap{cfg['overlap']}_topk{cfg['top_k']}.json"
    return json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-heldout", action="store_true",
                        help="After development comparisons, rerun the best configuration on all 25 questions.")
    args = parser.parse_args()

    rows = []
    raw = []

    for cfg in CONFIGS:
        result = run_one(cfg)
        raw.append(result)
        m = result["metrics"]
        rows.append({
            "configuration": cfg["name"],
            "chunk_size": cfg["chunk_size"],
            "top_k": cfg["top_k"],
            "quality_percent": m["quality_percent"],
            "median_latency_seconds": m["median_latency_seconds"],
            "cost_per_query_usd": m["system_cost_per_query_usd"],
        })

    table = [
        "| Configuration | Chunk Size | Top-K | Quality Score | Median Latency | Cost / Query |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        table.append(
            f"| {r['configuration']} | {r['chunk_size']} | {r['top_k']} | "
            f"{r['quality_percent']:.1f}% | {r['median_latency_seconds']:.3f}s | "
            f"${r['cost_per_query_usd']:.6f} |"
        )

    baseline = rows[0]
    best_quality = max(rows, key=lambda r: r["quality_percent"])
    best_latency = min(rows, key=lambda r: r["median_latency_seconds"])
    quality_delta = best_quality["quality_percent"] - baseline["quality_percent"]
    latency_reduction = (
        (baseline["median_latency_seconds"] - best_latency["median_latency_seconds"])
        / baseline["median_latency_seconds"] * 100
    )

    md = "\n".join([
        "# Lab 3 Configuration Comparison",
        "",
        *table,
        "",
        "## Analysis",
        f"- Baseline quality: {baseline['quality_percent']:.1f}%",
        f"- Best quality configuration: {best_quality['configuration']} ({best_quality['quality_percent']:.1f}%)",
        f"- Quality improvement over baseline: {quality_delta:+.1f} percentage points",
        f"- Lowest-latency configuration: {best_latency['configuration']} ({best_latency['median_latency_seconds']:.3f}s)",
        f"- Latency reduction versus baseline: {latency_reduction:.1f}%",
        "",
        "Use the actual results in your video. Do not claim a 10-point or 30% improvement unless the measured data meets the rubric.",
        "",
    ])
    (ROOT / "results" / "comparison.md").write_text(md, encoding="utf-8")
    (ROOT / "results" / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(md)

    if args.final_heldout:
        best = best_quality
        print("\nRunning best-quality configuration on ALL 25 questions...")
        cmd = [
            sys.executable, str(ROOT / "evaluate_lab3.py"),
            "--chunk-size", str(best["chunk_size"]),
            "--overlap", "40",
            "--top-k", str(best["top_k"]),
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
