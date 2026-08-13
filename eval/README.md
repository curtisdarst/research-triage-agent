# Eval harness — not built yet (Tier 1.5)

This directory is a placeholder. Per this project's own build plan, the eval
harness is **required before this repository is made public**, because a
reproducible grounding eval is the project's actual claim to being a
contribution rather than a demo. It is intentionally not part of the Tier 1
interview-demo build.

What will live here:

- A version-controlled set of 20-30 golden questions against the demo
  corpus, each with an expected per-claim support status (some fully
  answerable, some deliberately not, some answerable only in part).
- A runner reporting three metrics:
  - **Unsupported-claim rate** — claims the validator should have caught but
    didn't.
  - **Citation accuracy** — cited paper_ids that actually support the claim.
  - **False-refusal rate** — claims the validator flagged as unsupported
    that were, in fact, supported. This is the failure mode of the
    validator itself, since it is also an LLM with its own error rate.
- A GitHub Actions workflow that runs the harness on every PR touching the
  synthesis or validation prompts, so a prompt change can't silently
  regress grounding.
- Support for pointing the harness at an arbitrary BigQuery corpus, not just
  the arXiv demo table, via the same `GCP_PROJECT_ID` / `BQ_DATASET` /
  `BQ_TABLE` environment variables `agent/config.py` already reads.

Results will be committed to the main README as a table with model version
and run date, since these numbers go stale as models change.
