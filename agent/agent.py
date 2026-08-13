"""ADK agent definition: one orchestrator, four explicit named tools.

The orchestrator model is deliberately cheap (Flash) — it only sequences
tool calls, it does not do the synthesis or validation reasoning itself.
Those happen inside the synthesize/validate tools, which make their own
Gemini calls to the tiered models. See README "Why tiered models".

Tool call order is enforced via instruction, matching the pattern ADK's own
samples use for sequencing (see contributing/samples/core/hello_world in
google/adk-python). The final-answer rule below is what makes the guardrail
visible: the orchestrator is told to return validate()'s output verbatim,
so the guardrail's "Could not verify" section is a property of the code
(assembled deterministically inside validate — see agent/tools.py), not
something we're trusting the orchestrator LLM to faithfully preserve.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner

from agent.config import load_config
from agent.tools import reset_state, retrieve, search_corpus, synthesize, validate

_config = load_config()

INSTRUCTION = """You answer research questions by calling four tools, in \
this exact order, once each per question:

1. search_corpus(query) — find candidate papers.
2. retrieve(paper_ids) — pull full text for the papers search_corpus found. \
Pass ALL paper_ids from search_corpus's results, even if the list is empty.
3. synthesize(question, retrieval_id) — using the retrieval_id retrieve() \
gave you, write a claim-by-claim digest.
4. validate(synthesis_id, retrieval_id) — using the synthesis_id synthesize() \
gave you, check every claim against its source text and build the final answer.

Always call all four tools, in order, exactly once each, even if \
search_corpus finds nothing — retrieve/synthesize/validate all handle empty \
input and will produce an honest "could not verify" answer through the same \
path, which is the behavior we want to show.

After validate() returns, your final response MUST be exactly the \
'final_markdown' field from validate()'s output, verbatim — no extra \
commentary before or after it, no re-wrapping in code fences, no \
summarizing. validate() already assembled the complete, correctly \
formatted answer, including the "Could not verify" section if there is one. \
Do not soften, omit, or rephrase that section — reporting the gap honestly \
is the point.
"""

root_agent = Agent(
    name="research_triage_agent",
    model=_config.model_orchestrator,
    description=(
        "Answers research questions from a paper corpus with per-claim "
        "citations, and flags any claim it cannot verify against the "
        "retrieved source text rather than guessing."
    ),
    instruction=INSTRUCTION,
    tools=[search_corpus, retrieve, synthesize, validate],
)


async def run_query(question: str) -> str:
    """Runs one question through the agent and returns its final text answer."""
    reset_state()
    runner = InMemoryRunner(agent=root_agent, app_name="research_triage_agent")
    events = await runner.run_debug(question, quiet=True)

    for event in reversed(events):
        if not event.content or not event.content.parts:
            continue
        text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
        if text.strip():
            return text

    # Fallback: the orchestrator is instructed to echo validate()'s
    # final_markdown verbatim as its own final turn, but that last
    # inference call occasionally doesn't come through (rate limiting on a
    # fresh project's default quota, mainly — see README "Known
    # limitations"). validate() already produced the complete, correctly
    # formatted answer as a tool result, so read it directly from the event
    # stream instead of returning nothing.
    for event in reversed(events):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            fr = getattr(part, "function_response", None)
            if fr and fr.name == "validate" and fr.response:
                markdown = fr.response.get("final_markdown")
                if markdown:
                    return markdown

    return "(no final response produced — validate() may not have completed; see trace above)"
