"""Grounding eval runner (Tier 1.5).

Runs every golden question through the same four tools the agent uses at
runtime (search_corpus -> retrieve -> synthesize -> validate), then grades
each synthesized claim with an independent judge (eval/judge.py) and diffs
the judge's verdict against validate()'s verdict to compute:

  - unsupported-claim rate: claims validate() let through as "supported"
    that the judge says are not actually grounded (the validator's misses —
    what actually reaches a reader unflagged).
  - citation accuracy: of ALL synthesized claims, the fraction the judge
    confirms are genuinely grounded in their cited source text (measures
    synthesize()'s underlying quality, independent of whether validate()
    caught the rest).
  - false-refusal rate: claims validate() flagged as "unsupported" that the
    judge says WERE actually grounded (the validator's own false alarms —
    the failure mode people forget to check for in a guardrail).

Points at whatever corpus GCP_PROJECT_ID/BQ_DATASET/BQ_TABLE (see
agent/config.py) resolve to — nothing here is hardcoded to arXiv or to this
project's specific corpus, so this is runnable against any BigQuery table
matching the papers schema in setup/bigquery_schema.sql.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --questions path/to/other_questions.yaml
    python eval/run_eval.py --max-unsupported-rate 0.15 --max-false-refusal-rate 0.20
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from google.genai.errors import ClientError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.config import estimate_cost_usd, load_config  # noqa: E402
from agent.tools import (  # noqa: E402
    get_retrieval,
    get_synthesis,
    reset_state,
    retrieve,
    search_corpus,
    synthesize,
    validate,
)
from eval.judge import judge_claims  # noqa: E402

QUESTION_DELAY_SECONDS = 2.0  # be gentle with per-minute quota between questions
MAX_RETRIES = 4


@dataclass
class ClaimRecord:
    question_id: str
    claim_text: str
    paper_ids: list[str]
    validator_supported: bool
    judge_grounded: bool


@dataclass
class QuestionResult:
    question_id: str
    question: str
    expected_status: str
    claim_count: int
    supported_count: int
    unsupported_count: int
    answerability_match: bool
    cost_usd: float = 0.0
    claims: list[ClaimRecord] = field(default_factory=list)


def _with_retry(fn, *args):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = 15 * (2**attempt)
                print(f"    429 rate limited, backing off {wait}s...")
                time.sleep(wait)
                continue
            raise


def _run_pipeline(question: str):
    reset_state()
    s = search_corpus(question, top_k=5)
    paper_ids = [r["paper_id"] for r in s["results"]]
    r = retrieve(paper_ids)
    y = synthesize(question, r["retrieval_id"])
    v = validate(y["synthesis_id"], r["retrieval_id"])
    return r["retrieval_id"], y["synthesis_id"], v


def evaluate_question(q: dict) -> QuestionResult:
    retrieval_id, synthesis_id, validation = _with_retry(_run_pipeline, q["question"])

    synthesis = get_synthesis(synthesis_id)
    papers = get_retrieval(retrieval_id) or []
    papers_by_id = {p["paper_id"]: p for p in papers}
    claims = synthesis.claims if synthesis else []

    judge_verdicts, judge_in_tok, judge_out_tok = _with_retry(judge_claims, claims, papers_by_id)
    config = load_config()
    cost = estimate_cost_usd(config.model_judge, judge_in_tok, judge_out_tok)

    claim_supported: dict[int, bool] = validation.get("claim_supported", {})
    records = []
    for c in claims:
        jv = judge_verdicts.get(c.id)
        records.append(
            ClaimRecord(
                question_id=q["id"],
                claim_text=c.text,
                paper_ids=c.paper_ids,
                validator_supported=claim_supported.get(c.id, False),
                judge_grounded=jv.grounded if jv else False,
            )
        )

    supported_count = validation.get("supported_count", 0)
    expected = q["expected_status"]
    if expected == "answerable":
        answerability_match = supported_count > 0
    elif expected == "unanswerable":
        answerability_match = supported_count == 0
    else:  # partial — no hard expectation on count, always counts as a match
        answerability_match = True

    return QuestionResult(
        question_id=q["id"],
        question=q["question"],
        expected_status=expected,
        claim_count=len(claims),
        supported_count=supported_count,
        unsupported_count=validation.get("unsupported_count", 0),
        answerability_match=answerability_match,
        cost_usd=cost,
        claims=records,
    )


def compute_metrics(results: list[QuestionResult]) -> dict:
    all_claims = [c for r in results for c in r.claims]
    validator_said_supported = [c for c in all_claims if c.validator_supported]
    validator_said_unsupported = [c for c in all_claims if not c.validator_supported]

    unsupported_leaked = [c for c in validator_said_supported if not c.judge_grounded]
    false_refusals = [c for c in validator_said_unsupported if c.judge_grounded]
    judge_grounded_total = [c for c in all_claims if c.judge_grounded]

    unsupported_claim_rate = (
        len(unsupported_leaked) / len(validator_said_supported)
        if validator_said_supported
        else 0.0
    )
    false_refusal_rate = (
        len(false_refusals) / len(validator_said_unsupported)
        if validator_said_unsupported
        else 0.0
    )
    citation_accuracy = len(judge_grounded_total) / len(all_claims) if all_claims else 0.0

    answerability_matches = sum(1 for r in results if r.answerability_match)

    return {
        "total_questions": len(results),
        "total_claims": len(all_claims),
        "unsupported_claim_rate": round(unsupported_claim_rate, 4),
        "citation_accuracy": round(citation_accuracy, 4),
        "false_refusal_rate": round(false_refusal_rate, 4),
        "answerability_matches": answerability_matches,
        "answerability_match_rate": round(answerability_matches / len(results), 4)
        if results
        else 0.0,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
    }


def render_markdown(metrics: dict, results: list[QuestionResult], config) -> str:
    lines = [
        f"Model versions: orchestrator={config.model_orchestrator}, "
        f"synthesis={config.model_synthesis}, validation={config.model_validation}, "
        f"judge={config.model_judge}",
        f"Run date: {datetime.date.today().isoformat()}",
        f"Questions: {metrics['total_questions']}, total claims: {metrics['total_claims']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Unsupported-claim rate | {metrics['unsupported_claim_rate']:.1%} |",
        f"| Citation accuracy | {metrics['citation_accuracy']:.1%} |",
        f"| False-refusal rate | {metrics['false_refusal_rate']:.1%} |",
        f"| Answerability-category match | {metrics['answerability_matches']}/{metrics['total_questions']} "
        f"({metrics['answerability_match_rate']:.1%}) |",
        f"| Judge cost (this run) | ${metrics['total_cost_usd']:.4f} |",
        "",
        "| ID | Expected | Claims | Supported | Answerability match |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.question_id} | {r.expected_status} | {r.claim_count} | "
            f"{r.supported_count} | {'yes' if r.answerability_match else 'no'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parent / "golden_questions.yaml",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "results" / "latest.json"
    )
    parser.add_argument("--max-unsupported-rate", type=float, default=None)
    parser.add_argument("--max-false-refusal-rate", type=float, default=None)
    args = parser.parse_args()

    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
    config = load_config()

    results: list[QuestionResult] = []
    for i, q in enumerate(questions):
        print(f"[{i + 1}/{len(questions)}] {q['id']}: {q['question'][:70]}...")
        result = evaluate_question(q)
        print(
            f"    {result.claim_count} claims, {result.supported_count} supported "
            f"(expected={result.expected_status}, answerability_match={result.answerability_match})"
        )
        results.append(result)
        if i < len(questions) - 1:
            time.sleep(QUESTION_DELAY_SECONDS)

    metrics = compute_metrics(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "run_date": datetime.date.today().isoformat(),
                "models": {
                    "orchestrator": config.model_orchestrator,
                    "synthesis": config.model_synthesis,
                    "validation": config.model_validation,
                    "judge": config.model_judge,
                },
                "metrics": metrics,
                "questions": [
                    {
                        "id": r.question_id,
                        "expected_status": r.expected_status,
                        "claim_count": r.claim_count,
                        "supported_count": r.supported_count,
                        "unsupported_count": r.unsupported_count,
                        "answerability_match": r.answerability_match,
                        "claims": [
                            {
                                "text": c.claim_text,
                                "paper_ids": c.paper_ids,
                                "validator_supported": c.validator_supported,
                                "judge_grounded": c.judge_grounded,
                            }
                            for c in r.claims
                        ],
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(render_markdown(metrics, results, config))
    print()
    print(f"Full results written to {args.out}")

    failed = False
    if args.max_unsupported_rate is not None and metrics["unsupported_claim_rate"] > args.max_unsupported_rate:
        print(
            f"FAIL: unsupported-claim rate {metrics['unsupported_claim_rate']:.1%} "
            f"exceeds threshold {args.max_unsupported_rate:.1%}"
        )
        failed = True
    if args.max_false_refusal_rate is not None and metrics["false_refusal_rate"] > args.max_false_refusal_rate:
        print(
            f"FAIL: false-refusal rate {metrics['false_refusal_rate']:.1%} "
            f"exceeds threshold {args.max_false_refusal_rate:.1%}"
        )
        failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
