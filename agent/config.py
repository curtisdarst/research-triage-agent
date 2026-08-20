"""Environment-driven configuration and measured-cost pricing table.

Model IDs and dataset locations are config, not hardcoded, so upgrading a
model or pointing at a different corpus is a one-line change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    project_id: str
    region: str
    model_region: str
    dataset: str
    table: str
    model_orchestrator: str
    model_synthesis: str
    model_validation: str
    model_embedding: str
    model_judge: str


def load_config() -> Config:
    project_id = os.environ["GCP_PROJECT_ID"]
    region = os.environ.get("GCP_REGION", "us-central1")
    # Separate from `region`: BigQuery and Cloud Run stay pinned to
    # GCP_REGION, but gemini-3.7-flash and gemini-3.1-pro-preview both
    # 404 in us-central1 for this project and only work via the "global"
    # Vertex endpoint (confirmed by live-testing every region tried before
    # picking these as the defaults). If a future model needs a specific
    # region instead of global, override this env var, don't repoint
    # GCP_REGION, that would also move BigQuery/Cloud Run.
    model_region = os.environ.get("MODEL_REGION", "global")

    # ADK's internal google-genai client (used for the orchestrator agent's
    # own LLM calls) builds its client from these standard env vars rather
    # than from anything in our own Config — our explicit
    # genai.Client(vertexai=True, ...) calls in agent/tools.py don't cover
    # ADK's internal orchestrator model, so this keeps both paths on Vertex
    # instead of ADK's client silently falling back to the Gemini Developer
    # API (which needs an API key we don't have/want).
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", model_region)

    return Config(
        project_id=project_id,
        region=region,
        model_region=model_region,
        dataset=os.environ.get("BQ_DATASET", "research_triage"),
        table=os.environ.get("BQ_TABLE", "papers"),
        model_orchestrator=os.environ.get("MODEL_ORCHESTRATOR", "gemini-3.7-flash"),
        model_synthesis=os.environ.get("MODEL_SYNTHESIS", "gemini-3.1-pro-preview"),
        model_validation=os.environ.get("MODEL_VALIDATION", "gemini-3.7-flash"),
        model_embedding=os.environ.get("MODEL_EMBEDDING", "gemini-embedding-001"),
        # Deliberately Pro by default, even though validate() uses Flash —
        # the eval judge is meant to be a meaningfully stronger, slower,
        # more careful second opinion, not a repeat of the same check.
        # gemini-3.1-pro-preview is preview-tier (no stable Gemini 3.x Pro
        # exists yet), a deliberate exception to stable-only, see README.
        model_judge=os.environ.get("MODEL_JUDGE", "gemini-3.1-pro-preview"),
    )


# USD per 1M tokens. Verified against ai.google.dev/gemini-api/docs/pricing
# on 2026-08-13 — re-check before trusting for budgeting, prices drift.
PRICING_PER_1M_TOKENS = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-embedding-001": {"input": 0.15, "output": 0.0},
    # Verified against ai.google.dev/gemini-api/docs/pricing on 2026-08-15.
    # Promotional rate through 2026-12-31; rises to $1.50 / $7.50 on
    # 2027-01-01 — update this before trusting cost figures past that date.
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75},
    # Verified against ai.google.dev/gemini-api/docs/pricing on 2026-08-15.
    # <=200k token prompts; rises to $4.00 / $18.00 above 200k, not modeled
    # here since this project's prompts never approach that (small corpus,
    # short retrieval payloads).
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_PER_1M_TOKENS.get(model)
    if rates is None:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]


def billed_output_tokens(usage_metadata) -> int:
    """candidates_token_count alone undercounts real cost.

    2.5-series models emit internal "thinking" tokens by default
    (usage_metadata.thoughts_token_count) that are invisible in the
    response text but billed at the same per-token output rate as visible
    output (confirmed against a real GCP billing export: the "Thinking
    Text Output" SKU price matches the standard output SKU price exactly).
    For a trivial one-word answer, thinking tokens outnumbered visible
    output tokens 135:1 in one measured case. Every cost figure in this
    project counted only candidates_token_count before this was caught;
    call this everywhere output tokens are read for cost accounting.
    """
    if usage_metadata is None:
        return 0
    return (usage_metadata.candidates_token_count or 0) + (
        usage_metadata.thoughts_token_count or 0
    )
