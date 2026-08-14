"""Idempotent corpus ingest: arXiv -> embeddings -> BigQuery.

Usage:
    python ingest/ingest_arxiv.py [--max-results 400]

Safe to re-run: papers already present (by arXiv id) are updated in place
rather than duplicated, via a staging-table MERGE.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.cloud import bigquery
from google.genai import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.arxiv_client import fetch_papers  # noqa: E402

EMBED_BATCH_SIZE = 32
EMBED_INPUT_PRICE_PER_1M = 0.15  # gemini-embedding-001, USD, verified 2026-08-13


def _bq_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("authors", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("published_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("abstract", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def embed_abstracts(
    client: genai.Client, model: str, texts: list[str], on_progress=print
) -> tuple[list[list[float]], int]:
    """Returns (embedding vectors in input order, total input tokens billed)."""
    vectors: list[list[float]] = []
    total_tokens = 0
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        result = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors.extend(e.values for e in result.embeddings)
        total_tokens += sum(e.statistics.token_count for e in result.embeddings)
        on_progress(f"  embedded {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")
        time.sleep(0.2)
    return vectors, total_tokens


def upsert_papers(
    bq: bigquery.Client,
    project_id: str,
    dataset: str,
    rows: list[dict],
) -> int:
    """Idempotent upsert via a staging table + MERGE. Returns affected row count."""
    table_id = f"{project_id}.{dataset}.papers"
    staging_table_id = f"{project_id}.{dataset}.papers_staging"

    job_config = bigquery.LoadJobConfig(
        schema=_bq_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    load_job = bq.load_table_from_json(rows, staging_table_id, job_config=job_config)
    load_job.result()

    merge_sql = f"""
    MERGE `{table_id}` AS target
    USING `{staging_table_id}` AS source
    ON target.id = source.id
    WHEN MATCHED THEN UPDATE SET
      title = source.title,
      authors = source.authors,
      published_date = source.published_date,
      abstract = source.abstract,
      url = source.url,
      embedding = source.embedding,
      ingested_at = source.ingested_at
    WHEN NOT MATCHED THEN
      INSERT (id, title, authors, published_date, abstract, url, embedding, ingested_at)
      VALUES (source.id, source.title, source.authors, source.published_date,
              source.abstract, source.url, source.embedding, source.ingested_at)
    """
    merge_job = bq.query(merge_sql)
    merge_job.result()
    affected = merge_job.num_dml_affected_rows or 0

    bq.delete_table(staging_table_id, not_found_ok=True)
    return affected


def run_ingest(max_results: int, on_progress=print) -> dict:
    """Runs the full fetch -> embed -> upsert pipeline. Returns a summary dict.

    on_progress(str) is called with human-readable progress lines as the run
    proceeds (defaults to print for CLI use; web/main.py passes something
    that streams to the caller instead).
    """
    project_id = os.environ["GCP_PROJECT_ID"]
    region = os.environ.get("GCP_REGION", "us-central1")
    dataset = os.environ.get("BQ_DATASET", "research_triage")
    embedding_model = os.environ.get("MODEL_EMBEDDING", "gemini-embedding-001")

    on_progress(f"Fetching up to {max_results} papers from arXiv...")
    papers = fetch_papers(max_results=max_results)
    on_progress(f"Fetched {len(papers)} papers.")
    if not papers:
        return {"fetched": 0, "affected": 0, "total_rows": None, "cost_usd": 0.0, "tokens": 0}

    genai_client = genai.Client(vertexai=True, project=project_id, location=region)
    on_progress(f"Embedding {len(papers)} abstracts with {embedding_model}...")
    vectors, approx_tokens = embed_abstracts(
        genai_client, embedding_model, [p.abstract for p in papers], on_progress=on_progress
    )

    ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    rows = [
        {
            "id": p.id,
            "title": p.title,
            "authors": p.authors,
            "published_date": p.published_date,
            "abstract": p.abstract,
            "url": p.url,
            "embedding": vec,
            "ingested_at": ingested_at,
        }
        for p, vec in zip(papers, vectors)
    ]

    bq = bigquery.Client(project=project_id)
    on_progress(f"Upserting {len(rows)} rows into {dataset}.papers...")
    affected = upsert_papers(bq, project_id, dataset, rows)

    total_rows = next(
        iter(bq.query(f"SELECT COUNT(*) AS n FROM `{project_id}.{dataset}.papers`").result())
    )["n"]

    cost_usd = (approx_tokens / 1_000_000) * EMBED_INPUT_PRICE_PER_1M
    on_progress(
        f"Done. {affected} rows affected this run. Table now has {total_rows} papers. "
        f"Approx. embedding cost: ${cost_usd:.4f} (~{approx_tokens:,} tokens @ "
        f"${EMBED_INPUT_PRICE_PER_1M}/1M)"
    )
    return {
        "fetched": len(papers),
        "affected": affected,
        "total_rows": total_rows,
        "cost_usd": round(cost_usd, 4),
        "tokens": approx_tokens,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=400)
    args = parser.parse_args()
    run_ingest(args.max_results)


if __name__ == "__main__":
    main()
