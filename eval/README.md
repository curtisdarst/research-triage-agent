# Eval harness

Reproducible grounding eval for the Research Triage Agent. This is what
makes the repo a contribution rather than a demo (see main
[README](../README.md)). Runs a fixed set of golden questions through the
same four tools the agent uses at runtime, then independently re-grades
every synthesized claim to score `validate()`'s own accuracy.

## What it measures

- **Unsupported-claim rate**: of the claims `validate()` let through as
  "supported," what fraction an independent judge says are not actually
  grounded in the cited source text. This is what actually reaches a
  reader unflagged, the guardrail's miss rate.
- **Citation accuracy**: of *all* synthesized claims (regardless of
  `validate()`'s verdict), what fraction the judge confirms are genuinely
  grounded. Measures `synthesize()`'s underlying quality independent of
  whether the guardrail catches the rest.
- **False-refusal rate**: of the claims `validate()` flagged as
  "unsupported," what fraction the judge says were actually well grounded.
  This is `validate()`'s own false-alarm rate, the failure mode people
  forget to check for when they build a guardrail (see main README "Known
  limitations": the validator is itself an LLM, and so is this judge).

The judge (`eval/judge.py`) is deliberately a different, stronger model
(`gemini-3.1-pro-preview` by default) than `validate()`
(`gemini-3.7-flash`), with an independently written prompt, graded blind
to `validate()`'s verdict. It is not the same check run twice.

## Golden questions

[`golden_questions.yaml`](golden_questions.yaml): 25 questions, each with
a human-assigned `expected_status` (`answerable` / `partial` /
`unanswerable`) and a `notes` field explaining *why*, written by reading
the actual corpus abstracts, never by running the agent and copying its
answer (that would make the eval circular). `unanswerable` questions were
additionally verified via `search_corpus` alone, confirming the retrieved
titles are genuinely unrelated, since embedding similarity stays
misleadingly high on shared phrasing like "prompt engineering patterns"
even for an unrelated domain.

## Running it

Points at whatever corpus `GCP_PROJECT_ID` / `BQ_DATASET` / `BQ_TABLE`
resolve to (see `.env` / `agent/config.py`); nothing in the harness is
hardcoded to arXiv or to this project's specific corpus. To eval your own
corpus, point those env vars at a BigQuery table matching the schema in
[`setup/bigquery_schema.sql`](../setup/bigquery_schema.sql), and write your
own `golden_questions.yaml` (or pass `--questions path/to/yours.yaml`).

```bash
python eval/run_eval.py
```

Writes full detail (every claim, both verdicts, judge reasoning) to
`eval/results/latest.json`, and prints a markdown summary table, the same
table format committed to the main README. Add
`--max-unsupported-rate 0.15 --max-false-refusal-rate 0.20` (or your own
thresholds) to make the run exit non-zero on regression, useful for
scripting a manual regression check before merging a prompt change to
`synthesize()` or `validate()`.

No CI gate is wired up. Running the eval before merging a prompt change is
a manual step for now, not an automated one.
