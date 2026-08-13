"""Research Triage Agent — CLI demo runner.

Usage:
    python run_demo.py --demo happy
    python run_demo.py --demo gap
    python run_demo.py --query "your own research question"
"""

from __future__ import annotations

import argparse
import asyncio
import time

from agent.agent import run_query
from agent.trace import trace

HAPPY_QUERY = (
    "What do studies say about the effect of chain-of-thought prompting on "
    "the correctness of LLM-generated code, compared to standard prompting?"
)

# Shares surface vocabulary with the corpus ("prompt engineering patterns",
# "the literature") so it embeds close enough to return real top-k results —
# but the outcome domain (agriculture) is entirely outside what a corpus
# about prompting and LLM code quality covers. search_corpus still returns
# its top 5 nearest neighbors (all genuinely about code/prompting, none
# about agriculture), and synthesize correctly declines to manufacture a
# claim about crop yields from them — this is the query that should trigger
# the guardrail rather than a naive confident answer.
GAP_QUERY = (
    "What does the literature say about the effect of prompt engineering "
    "patterns on crop yield prediction accuracy in precision agriculture?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", choices=["happy", "gap"], help="Run a prepared demo query")
    group.add_argument("--query", type=str, help="Run a freeform query")
    args = parser.parse_args()

    if args.demo == "happy":
        question = HAPPY_QUERY
    elif args.demo == "gap":
        question = GAP_QUERY
    else:
        question = args.query

    print(f"Question: {question}\n")

    start = time.perf_counter()
    answer = asyncio.run(run_query(question))
    wall_seconds = time.perf_counter() - start

    print("=== Trace ===")
    print(trace.render())
    print()
    print("=== Final Answer ===")
    print(answer)
    print()
    print(f"Wall clock time: {wall_seconds:.1f}s")


if __name__ == "__main__":
    main()
