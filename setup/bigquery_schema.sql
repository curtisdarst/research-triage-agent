-- Research Triage Agent — BigQuery schema
-- Run after `setup/provision_gcp.sh` (which creates the dataset). Idempotent:
-- safe to re-run via `bq query --use_legacy_sql=false < setup/bigquery_schema.sql`
-- with GCP_PROJECT_ID / BQ_DATASET substituted, or via provision_gcp.sh directly.

CREATE TABLE IF NOT EXISTS `${GCP_PROJECT_ID}.${BQ_DATASET}.papers` (
  id             STRING    NOT NULL OPTIONS (description = 'arXiv id, e.g. 2508.01234'),
  title          STRING    NOT NULL,
  authors        STRING    NOT NULL OPTIONS (description = 'Comma-separated author list'),
  published_date DATE      NOT NULL,
  abstract       STRING    NOT NULL,
  url            STRING    NOT NULL,
  embedding      ARRAY<FLOAT64> OPTIONS (description = 'gemini-embedding-001 output, 3072 dims'),
  ingested_at    TIMESTAMP NOT NULL
)
OPTIONS (
  description = 'arXiv abstracts ingested for the Research Triage Agent demo corpus.'
);

-- A vector index is deliberately NOT created here. BigQuery only populates a
-- vector index once the indexed table exceeds ~10 MB (below that,
-- VECTOR_SEARCH silently falls back to brute force with indexUnusedReasons =
-- BASE_TABLE_TOO_SMALL). At ~300-500 rows x 3072-dim FLOAT64 embeddings this
-- corpus sits at or under that line, so brute-force VECTOR_SEARCH is the
-- correct choice at this scale, not a missing optimization. If you grow the
-- corpus well past that threshold, uncomment and run:
--
-- CREATE VECTOR INDEX IF NOT EXISTS papers_embedding_idx
-- ON `${GCP_PROJECT_ID}.${BQ_DATASET}.papers`(embedding)
-- OPTIONS (index_type = 'IVF', distance_type = 'COSINE');
