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

    # ADK's internal google-genai client (used for the orchestrator agent's
    # own LLM calls) builds its client from these standard env vars rather
    # than from anything in our own Config — our explicit
    # genai.Client(vertexai=True, ...) calls in agent/tools.py don't cover
    # ADK's internal orchestrator model, so this keeps both paths on Vertex
    # instead of ADK's client silently falling back to the Gemini Developer
    # API (which needs an API key we don't have/want).
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", region)

    return Config(
        project_id=project_id,
        region=region,
        dataset=os.environ.get("BQ_DATASET", "research_triage"),
        table=os.environ.get("BQ_TABLE", "papers"),
        model_orchestrator=os.environ.get("MODEL_ORCHESTRATOR", "gemini-2.5-flash"),
        model_synthesis=os.environ.get("MODEL_SYNTHESIS", "gemini-2.5-pro"),
        model_validation=os.environ.get("MODEL_VALIDATION", "gemini-2.5-flash"),
        model_embedding=os.environ.get("MODEL_EMBEDDING", "gemini-embedding-001"),
        # Deliberately Pro by default, even though validate() uses Flash —
        # the eval judge is meant to be a meaningfully stronger, slower,
        # more careful second opinion, not a repeat of the same check.
        model_judge=os.environ.get("MODEL_JUDGE", "gemini-2.5-pro"),
    )


# USD per 1M tokens. Verified against ai.google.dev/gemini-api/docs/pricing
# on 2026-08-13 — re-check before trusting for budgeting, prices drift.
PRICING_PER_1M_TOKENS = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-embedding-001": {"input": 0.15, "output": 0.0},
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
