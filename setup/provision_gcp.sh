#!/usr/bin/env bash
# Research Triage Agent — one-time GCP provisioning.
# Idempotent: safe to re-run. Requires `gcloud auth login` and
# `gcloud auth application-default login` to already be done — this script
# never creates or downloads a service account key. Auth for the agent at
# runtime is Application Default Credentials (your own login), which is
# sufficient for local/demo use. See README "Production hardening" for the
# Workload Identity path used in a real deployment.
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID first (see .env.example)}"
: "${GCP_REGION:=us-central1}"
: "${BQ_DATASET:=research_triage}"

echo "Project:  $GCP_PROJECT_ID"
echo "Region:   $GCP_REGION"
echo "Dataset:  $BQ_DATASET"

gcloud config set project "$GCP_PROJECT_ID" --quiet

echo "Enabling required APIs (no-op if already enabled)..."
gcloud services enable \
  bigquery.googleapis.com \
  aiplatform.googleapis.com \
  --project "$GCP_PROJECT_ID"

echo "Creating BigQuery dataset (no-op if it already exists)..."
if ! bq --project_id="$GCP_PROJECT_ID" show "$BQ_DATASET" >/dev/null 2>&1; then
  bq --project_id="$GCP_PROJECT_ID" mk \
    --dataset \
    --location="$GCP_REGION" \
    --description="Research Triage Agent demo corpus" \
    "${GCP_PROJECT_ID}:${BQ_DATASET}"
else
  echo "  dataset ${BQ_DATASET} already exists, skipping"
fi

echo "Creating papers table (no-op if it already exists)..."
sed \
  -e "s/\${GCP_PROJECT_ID}/${GCP_PROJECT_ID}/g" \
  -e "s/\${BQ_DATASET}/${BQ_DATASET}/g" \
  "$(dirname "$0")/bigquery_schema.sql" \
  | bq --project_id="$GCP_PROJECT_ID" query --use_legacy_sql=false

echo "Done. Run ingest/ingest_arxiv.py next."
