"""Structured-output schema for the eval judge's independent grading pass."""

from __future__ import annotations

from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    claim_id: int
    grounded: bool
    reasoning: str


class JudgeResult(BaseModel):
    verdicts: list[JudgeVerdict]
