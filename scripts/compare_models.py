"""Model tier comparison (Tier 2): same query, same retrieval, synthesized
by Flash vs Pro, showing the actual quality/cost tradeoff rather than
asserting one exists.

Retrieval (search_corpus + retrieve) runs once and is shared between both
models, so the comparison isolates the synthesis model as the only
variable. Each model's output is then run through the same validate()
guardrail (Flash, matching the production configuration) so "quality" is
a measured number, grounded claims per model, not just a subjective read
of two paragraphs.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --query "your own research question"
    python scripts/compare_models.py --models gemini-3.7-flash,gemini-3.1-pro-preview
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.tools import (  # noqa: E402
    reset_state,
    retrieve,
    search_corpus,
    synthesize_with_model,
    validate,
)
from agent.trace import trace  # noqa: E402
from run_demo import HAPPY_QUERY  # noqa: E402


def run_one_model(question: str, retrieval_id: str, model: str) -> dict:
    trace.reset()
    y = synthesize_with_model(question, retrieval_id, model)
    v = validate(y["synthesis_id"], retrieval_id)
    entries = list(trace.entries)  # synthesize + validate entries for this model only
    return {
        "model": model,
        "claim_count": y["claim_count"],
        "supported_count": v["supported_count"],
        "unsupported_count": v["unsupported_count"],
        "final_markdown": v["final_markdown"],
        "cost_usd": sum(e.cost_usd for e in entries),
        "latency_ms": sum(e.latency_ms for e in entries),
        "input_tokens": sum(e.input_tokens for e in entries),
        "output_tokens": sum(e.output_tokens for e in entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=str, default=HAPPY_QUERY)
    parser.add_argument("--models", type=str, default="gemini-3.7-flash,gemini-3.1-pro-preview")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    print(f"Question: {args.query}\n")

    reset_state()
    search_result = search_corpus(args.query, top_k=5)
    paper_ids = [r["paper_id"] for r in search_result["results"]]
    retrieval = retrieve(paper_ids)
    retrieval_id = retrieval["retrieval_id"]
    print(f"Retrieved {len(paper_ids)} papers (shared across both models below).\n")

    results = [run_one_model(args.query, retrieval_id, m) for m in models]

    headers = ["model", "claims", "supported", "unsupported", "in/out tok", "cost", "latency"]
    rows = [
        [
            r["model"],
            str(r["claim_count"]),
            str(r["supported_count"]),
            str(r["unsupported_count"]),
            f"{r['input_tokens']}/{r['output_tokens']}",
            f"${r['cost_usd']:.5f}",
            f"{r['latency_ms']:.0f}ms",
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    if len(results) == 2:
        cheaper, pricier = sorted(results, key=lambda r: r["cost_usd"])
        cost_ratio = pricier["cost_usd"] / cheaper["cost_usd"] if cheaper["cost_usd"] else float("inf")
        print(
            f"\n{pricier['model']} cost {cost_ratio:.1f}x more than {cheaper['model']} "
            f"for this query."
        )

    for r in results:
        print(f"\n=== {r['model']} digest ===\n")
        print(r["final_markdown"])


if __name__ == "__main__":
    main()
