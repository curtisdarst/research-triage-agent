"""Independent grading pass used only by the eval harness.

validate() (agent/tools.py) is the production guardrail: Flash, fast, cheap,
runs on every query. This judge is a separate, deliberately stronger check
used only to score validate()'s own accuracy against the golden question
set — it is not part of the agent's runtime path. Grading each claim
independently (blind to validate's verdict) and then diffing against
validate's verdict is what makes unsupported-claim rate and false-refusal
rate measurable at all, rather than just re-asserting whatever validate
already said.

This is still an LLM judging LLM output, so it inherits the same class of
error the whole project is honest about elsewhere (see README "Known
limitations") — a stronger model and an independently-written prompt reduce
but do not eliminate that.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from agent.config import billed_output_tokens, load_config
from agent.schemas import Claim
from eval.schemas import JudgeResult, JudgeVerdict

_config = load_config()
_genai: genai.Client | None = None


def _genai_client() -> genai.Client:
    global _genai
    if _genai is None:
        _genai = genai.Client(
            vertexai=True, project=_config.project_id, location=_config.model_region
        )
    return _genai


_JUDGE_INSTRUCTION = """You are an independent auditor reviewing another \
system's fact-checking work. You will be given a set of claims, each with \
the paper_ids it cites and the full abstract text of each cited paper. For \
each claim, decide for yourself — from scratch, ignoring any other opinion \
— whether the claim's specific factual content is genuinely and directly \
stated by the cited abstract text.

Mark a claim NOT grounded if:
- The cited text is topically related but does not state the specific fact \
claimed.
- The claim generalizes, overstates, or draws an inference beyond what the \
text explicitly says.
- A cited paper_id has no corresponding source text at all (nothing to \
ground it in).

Be strict. A plausible-sounding claim that is not explicitly backed by the \
provided text is not grounded, no matter how reasonable it sounds.
"""


def judge_claims(
    claims: list[Claim], papers_by_id: dict[str, dict]
) -> tuple[dict[int, JudgeVerdict], int, int]:
    """Independently grades each claim. Returns (verdicts by claim id, input_tokens, output_tokens)."""
    if not claims:
        return {}, 0, 0

    claims_block = "\n\n".join(
        f"claim_id: {c.id}\nclaim_text: {c.text}\ncited_paper_ids: {c.paper_ids}\n"
        + "\n".join(
            f"--- source text for {pid} ---\n{papers_by_id[pid]['abstract']}"
            for pid in c.paper_ids
            if pid in papers_by_id
        )
        for c in claims
    )

    response = _genai_client().models.generate_content(
        model=_config.model_judge,
        contents=claims_block,
        config=types.GenerateContentConfig(
            system_instruction=_JUDGE_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=JudgeResult,
            temperature=0.0,
        ),
    )
    result: JudgeResult = response.parsed
    verdicts = {v.claim_id: v for v in result.verdicts}
    in_tok = response.usage_metadata.prompt_token_count or 0
    out_tok = billed_output_tokens(response.usage_metadata)
    return verdicts, in_tok, out_tok
