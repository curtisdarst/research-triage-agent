# Eval harness

Reproducible grounding eval for the Research Triage Agent — this is what
makes the repo a contribution rather than a demo (see main
[README](../README.md)). Runs a fixed set of golden questions through the
same four tools the agent uses at runtime, then independently re-grades
every synthesized claim to score `validate()`'s own accuracy.

## What it measures

- **Unsupported-claim rate** — of the claims `validate()` let through as
  "supported," what fraction an independent judge says are not actually
  grounded in the cited source text. This is what actually reaches a
  reader unflagged — the guardrail's miss rate.
- **Citation accuracy** — of *all* synthesized claims (regardless of
  `validate()`'s verdict), what fraction the judge confirms are genuinely
  grounded. Measures `synthesize()`'s underlying quality independent of
  whether the guardrail catches the rest.
- **False-refusal rate** — of the claims `validate()` flagged as
  "unsupported," what fraction the judge says were actually well grounded.
  This is `validate()`'s own false-alarm rate — the failure mode people
  forget to check for when they build a guardrail (see main README "Known
  limitations": the validator is itself an LLM, and so is this judge).

The judge (`eval/judge.py`) is deliberately a different, stronger model
(`gemini-2.5-pro` by default) than `validate()` (`gemini-2.5-flash`), with
an independently written prompt, graded blind to `validate()`'s verdict —
it is not the same check run twice.

## Golden questions

[`golden_questions.yaml`](golden_questions.yaml) — 25 questions, each with
a human-assigned `expected_status` (`answerable` / `partial` /
`unanswerable`) and a `notes` field explaining *why*, written by reading
the actual corpus abstracts, never by running the agent and copying its
answer (that would make the eval circular). `unanswerable` questions were
additionally verified via `search_corpus` alone — confirming the retrieved
titles are genuinely unrelated, since embedding similarity stays
misleadingly high on shared phrasing like "prompt engineering patterns"
even for an unrelated domain.

## Running it

Points at whatever corpus `GCP_PROJECT_ID` / `BQ_DATASET` / `BQ_TABLE`
resolve to (see `.env` / `agent/config.py`) — nothing in the harness is
hardcoded to arXiv or to this project's specific corpus. To eval your own
corpus, point those env vars at a BigQuery table matching the schema in
[`setup/bigquery_schema.sql`](../setup/bigquery_schema.sql), and write your
own `golden_questions.yaml` (or pass `--questions path/to/yours.yaml`).

```bash
python eval/run_eval.py
```

Writes full detail (every claim, both verdicts, judge reasoning) to
`eval/results/latest.json`, and prints a markdown summary table — the same
table format committed to the main README. Add
`--max-unsupported-rate 0.15 --max-false-refusal-rate 0.20` (or your own
thresholds) to make the run exit non-zero on regression, which is what CI
uses.

## CI gate

[`.github/workflows/eval.yml`](../.github/workflows/eval.yml) runs this on
every PR touching `agent/**` or `eval/**`, so a prompt change to
`synthesize()` or `validate()` can't silently regress grounding.

Auth is Workload Identity Federation — no downloaded service-account key.
One-time setup (already done for this repo; documented here so it's
reproducible for a fork):

```bash
gcloud iam workload-identity-pools create github-actions \
  --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github-actions \
  --display-name="GitHub provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == 'YOUR_GITHUB_USERNAME'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

gcloud iam service-accounts create github-actions-eval \
  --display-name="GitHub Actions - eval harness (read-only BQ + Vertex AI user)"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-eval@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-eval@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-eval@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud iam service-accounts add-iam-policy-binding \
  github-actions-eval@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/YOUR_PROJECT_NUMBER/locations/global/workloadIdentityPools/github-actions/attribute.repository/YOUR_GITHUB_USERNAME/YOUR_REPO"
```

Then set these as **repository variables** (Settings → Secrets and
variables → Actions → Variables — not secrets, none of these values are
sensitive on their own): `GCP_WORKLOAD_IDENTITY_PROVIDER` (the provider's
full resource name), `GCP_EVAL_SERVICE_ACCOUNT`, `GCP_PROJECT_ID`,
`GCP_REGION`, `BQ_DATASET`, `BQ_TABLE`.

The `attribute-condition` above scopes the OIDC provider to your GitHub
account/org; the service-account IAM binding further scopes it to this one
specific repo. A workflow run from a fork or a different repo cannot
impersonate this service account.
