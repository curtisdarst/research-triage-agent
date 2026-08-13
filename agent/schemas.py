"""Structured-output schemas for the synthesis and validation LLM calls.

Passed as `response_schema` to `generate_content`, which makes the SDK
enforce and auto-parse JSON output (`response.parsed`) instead of us
hand-parsing free text — the claim/citation structure is exactly what the
guardrail checks against, so it needs to be reliable, not regex'd out of
prose.
"""

from __future__ import annotations

from pydantic import BaseModel


class Claim(BaseModel):
    id: int
    text: str
    paper_ids: list[str]


class SynthesisResult(BaseModel):
    claims: list[Claim]


class ClaimValidation(BaseModel):
    id: int
    supported: bool
    reasoning: str


class ValidationResult(BaseModel):
    results: list[ClaimValidation]
