"""Print and save a concise failure-analysis report from an evaluation JSON."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LABELS = {
    "retrieval_miss": "Retrieval Miss",
    "context_ignored": "Right Context Retrieved but Model Ignored It",
    "incomplete_or_incorrect_answer": "Incomplete / Incorrect Open Answer",
    "answer_or_format_error": "Answer / Format Error",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", help="Path to a results/*.json file")
    args = parser.parse_args()

    path = Path(args.result_file)
    if not path.is_absolute():
        path = ROOT / path

    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]

    counts = {}
    for item in results:
        bucket = item.get("failure_bucket")
        if bucket:
            counts[bucket] = counts.get(bucket, 0) + 1

    total_failures = sum(counts.values())
    lines = [
        "# Failure Analysis",
        "",
        f"Configuration: chunk={data['configuration']['chunk_size']}, "
        f"overlap={data['configuration']['overlap']}, top-k={data['configuration']['top_k']}",
        "",
        "| Failure Bucket | Count | % of Failures |",
        "|---|---:|---:|",
    ]

    for bucket, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        pct = (count / total_failures * 100) if total_failures else 0
        lines.append(f"| {LABELS.get(bucket, bucket)} | {count} | {pct:.1f}% |")

    lines.extend(["", "## Failed Items", ""])
    for item in results:
        if item.get("failure_bucket"):
            lines.extend([
                f"### Q{item['id']}: {item['question']}",
                f"- Bucket: **{LABELS.get(item['failure_bucket'], item['failure_bucket'])}**",
                f"- Expected evidence: {item['expected_reference']}",
                f"- Expected answer: {item['expected_answer']}",
                f"- Actual answer: {item['actual_answer']}",
                f"- Judge / matcher reason: {item['reason']}",
                f"- Retrieval hit: {item['retrieval_hit']}",
                "",
            ])

    out = ROOT / "results" / "failure_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print(f"\nSaved: {out.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
