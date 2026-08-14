"""The four named tools: search_corpus, retrieve, synthesize, validate.

Design note on handles vs. payloads: `retrieve` and `synthesize` return a
short opaque id (`retrieval_id`, `synthesis_id`) rather than making the
orchestrator LLM copy full abstract text or claim lists through its own
context to pass into the next tool call. The full payloads are kept
server-side in an in-process store. This avoids relying on the model to
faithfully retype large JSON blobs between tool calls — a real source of
transcription drift — and keeps each tool call's arguments small. It also
means tool call arguments/results stay a fair reflection of orchestration
cost rather than being inflated by re-quoted document text.

`validate` is the guardrail. It is the only tool that checks each claim's
literal text against the literal retrieved source text and reports
supported/unsupported per claim — this is deliberately different from
critique-style validators (e.g. ADK's llm_auditor sample): it verifies
against specific retrieved evidence with per-claim provenance, it does not
silently rewrite the digest, and it assembles the final answer itself so
the guardrail's output can't be paraphrased away by the orchestrator.
"""

from __future__ import annotations

import time
import uuid

from google import genai
from google.cloud import bigquery
from google.genai import types

from agent.config import estimate_cost_usd, load_config
from agent.schemas import SynthesisResult, ValidationResult
from agent.trace import trace

_config = load_config()
_bq: bigquery.Client | None = None
_genai: genai.Client | None = None
_store: dict[str, object] = {}


def _bq_client() -> bigquery.Client:
    global _bq
    if _bq is None:
        _bq = bigquery.Client(project=_config.project_id)
    return _bq


def _genai_client() -> genai.Client:
    global _genai
    if _genai is None:
        _genai = genai.Client(
            vertexai=True, project=_config.project_id, location=_config.region
        )
    return _genai


def reset_state() -> None:
    """Clears the handle store and trace. Call once per demo query."""
    _store.clear()
    trace.reset()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def get_synthesis(synthesis_id: str) -> SynthesisResult | None:
    """Eval-only accessor: the full SynthesisResult (with claim text/paper_ids), not just the summary synthesize() returns."""
    return _store.get(synthesis_id)  # type: ignore[return-value]


def get_retrieval(retrieval_id: str) -> list[dict] | None:
    """Eval-only accessor: the full retrieved papers (with abstract text), not just the preview retrieve() returns."""
    return _store.get(retrieval_id)  # type: ignore[return-value]


def search_corpus(query: str, top_k: int = 5) -> dict:
    """Searches the paper corpus for abstracts most relevant to a research question.

    Args:
        query (str): The research question or topic to search for.
        top_k (int): Number of top matching papers to return. Defaults to 5.

    Returns:
        dict: status, and either 'results' (list of {paper_id, title, score})
            ordered most to least relevant, or 'error_message'.
    """
    start = time.perf_counter()
    embed_resp = _genai_client().models.embed_content(
        model=_config.model_embedding,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = embed_resp.embeddings[0].values
    embed_tokens = embed_resp.embeddings[0].statistics.token_count

    table_id = f"{_config.project_id}.{_config.dataset}.{_config.table}"
    sql = f"""
    SELECT base.id AS paper_id, base.title AS title, distance
    FROM VECTOR_SEARCH(
      TABLE `{table_id}`,
      'embedding',
      (SELECT @query_embedding AS embedding),
      top_k => @top_k,
      distance_type => 'COSINE'
    )
    ORDER BY distance ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_vector),
            bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
        ]
    )
    rows = list(_bq_client().query(sql, job_config=job_config).result())
    results = [
        {"paper_id": r["paper_id"], "title": r["title"], "score": round(1 - r["distance"], 4)}
        for r in rows
    ]

    latency_ms = (time.perf_counter() - start) * 1000
    cost = estimate_cost_usd(_config.model_embedding, embed_tokens, 0)
    trace.record(
        tool="search_corpus",
        input_summary=f"query={query!r} top_k={top_k}",
        output_summary=f"{len(results)} results",
        model=_config.model_embedding,
        input_tokens=embed_tokens,
        output_tokens=0,
        latency_ms=latency_ms,
        cost_usd=cost,
    )
    return {"status": "success", "results": results}


def retrieve(paper_ids: list[str]) -> dict:
    """Retrieves full abstract text and metadata for a set of paper ids.

    Args:
        paper_ids (list[str]): arXiv paper ids, e.g. from search_corpus results.

    Returns:
        dict: status, 'retrieval_id' (pass this to synthesize), and 'papers'
            (list of {paper_id, title, authors, published_date, url,
            abstract_preview} — a short preview only; synthesize reads the
            full text server-side via retrieval_id), or 'error_message'.
    """
    start = time.perf_counter()
    table_id = f"{_config.project_id}.{_config.dataset}.{_config.table}"
    sql = f"""
    SELECT id, title, authors, published_date, abstract, url
    FROM `{table_id}`
    WHERE id IN UNNEST(@ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", paper_ids)]
    )
    rows = list(_bq_client().query(sql, job_config=job_config).result())
    papers = [
        {
            "paper_id": r["id"],
            "title": r["title"],
            "authors": r["authors"],
            "published_date": r["published_date"].isoformat(),
            "abstract": r["abstract"],
            "url": r["url"],
        }
        for r in rows
    ]

    retrieval_id = _new_id("r")
    _store[retrieval_id] = papers

    latency_ms = (time.perf_counter() - start) * 1000
    trace.record(
        tool="retrieve",
        input_summary=f"paper_ids={paper_ids}",
        output_summary=f"retrieval_id={retrieval_id}, {len(papers)} papers",
        model=None,
        input_tokens=0,
        output_tokens=0,
        latency_ms=latency_ms,
        cost_usd=0.0,
    )
    return {
        "status": "success",
        "retrieval_id": retrieval_id,
        "papers": [
            {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "authors": p["authors"],
                "published_date": p["published_date"],
                "url": p["url"],
                "abstract_preview": p["abstract"][:150],
            }
            for p in papers
        ],
    }


_SYNTHESIS_INSTRUCTION = """You are a research digest writer. Given a research \
question and a set of retrieved paper abstracts, write a digest answering the \
question as a set of discrete factual claims.

Rules:
- Every claim must be attributable to one or more of the retrieved abstracts.
- Every claim MUST cite paper_ids from the retrieved set only — never invent \
an id.
- Break the answer into multiple atomic claims rather than one long paragraph; \
each claim should be checkable as true or false against a single abstract's text.
- Write naturally and helpfully, the way a research assistant would summarize \
a literature set for a question. Do not add meta-commentary about \
confidence or hedging — a separate verification step checks each claim \
against the source text, that is not your job here.
- If the retrieved abstracts do not substantively address what the question \
is actually asking about — for example the question is about one subject \
and the retrieved papers are about a different, unrelated subject — output \
ZERO claims. Do not write a claim that just notes the mismatch, and do not \
write claims describing what the abstracts cover instead if that isn't what \
was asked. An empty claim list is itself the correct, honest output here; a \
downstream step already turns that into a clear "could not verify" answer, \
so you do not need to explain the mismatch yourself. Only write claims that \
are actually responsive to the question asked.
"""


def synthesize(question: str, retrieval_id: str) -> dict:
    """Writes a claim-by-claim digest answering the question from retrieved papers.

    Args:
        question (str): The original research question.
        retrieval_id (str): The retrieval_id returned by retrieve().

    Returns:
        dict: status, 'synthesis_id' (pass this to validate), and
            'claim_count', or 'error_message' if retrieval_id is unknown.
    """
    papers = _store.get(retrieval_id)
    if papers is None:
        return {"status": "error", "error_message": f"unknown retrieval_id {retrieval_id!r}"}

    if not papers:
        synthesis_id = _new_id("y")
        _store[synthesis_id] = SynthesisResult(claims=[])
        trace.record(
            tool="synthesize", input_summary=f"question={question!r} retrieval_id={retrieval_id}",
            output_summary=f"synthesis_id={synthesis_id}, 0 claims (nothing retrieved)",
            model=None, input_tokens=0, output_tokens=0, latency_ms=0.0, cost_usd=0.0,
        )
        return {"status": "success", "synthesis_id": synthesis_id, "claim_count": 0}

    start = time.perf_counter()
    sources_block = "\n\n".join(
        f"paper_id: {p['paper_id']}\ntitle: {p['title']}\nabstract: {p['abstract']}"
        for p in papers
    )
    prompt = f"Research question: {question}\n\nRetrieved abstracts:\n\n{sources_block}"

    response = _genai_client().models.generate_content(
        model=_config.model_synthesis,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYNTHESIS_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SynthesisResult,
            temperature=0.3,
        ),
    )
    result: SynthesisResult = response.parsed

    synthesis_id = _new_id("y")
    _store[synthesis_id] = result

    latency_ms = (time.perf_counter() - start) * 1000
    in_tok = response.usage_metadata.prompt_token_count or 0
    out_tok = response.usage_metadata.candidates_token_count or 0
    cost = estimate_cost_usd(_config.model_synthesis, in_tok, out_tok)
    trace.record(
        tool="synthesize",
        input_summary=f"question={question!r} retrieval_id={retrieval_id}",
        output_summary=f"synthesis_id={synthesis_id}, {len(result.claims)} claims",
        model=_config.model_synthesis,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        cost_usd=cost,
    )
    return {"status": "success", "synthesis_id": synthesis_id, "claim_count": len(result.claims)}


_VALIDATION_INSTRUCTION = """You are a strict grounding checker. For each \
claim, you are given the claim's text and the full abstract text of every \
paper_id it cites. Decide whether the claim's specific factual content is \
actually stated or directly and specifically supported by that abstract \
text — not by outside knowledge, not by plausibility, not by topical \
relevance alone.

Mark a claim unsupported if:
- The cited abstract does not contain the specific fact claimed, even if it \
is topically related.
- A paper_id it cites is not in the provided set at all.
- The claim overstates, generalizes beyond, or misattributes what the \
abstract actually says.

Mark it supported only when the abstract text directly backs the claim.
"""


def validate(synthesis_id: str, retrieval_id: str) -> dict:
    """Checks each synthesized claim against its cited source text and builds the final answer.

    If any claim is unsupported, the final answer explicitly separates
    supported findings from a "Could not verify" section rather than
    silently dropping or rewriting the unsupported claims.

    Args:
        synthesis_id (str): The synthesis_id returned by synthesize().
        retrieval_id (str): The retrieval_id returned by retrieve().

    Returns:
        dict: status, 'final_markdown' (the complete final answer — return
            this verbatim as your response), 'supported_count',
            'unsupported_count', or 'error_message'.
    """
    result: SynthesisResult | None = _store.get(synthesis_id)  # type: ignore[assignment]
    papers: list[dict] | None = _store.get(retrieval_id)  # type: ignore[assignment]
    if result is None or papers is None:
        return {
            "status": "error",
            "error_message": f"unknown synthesis_id/retrieval_id ({synthesis_id!r}, {retrieval_id!r})",
        }

    if not result.claims:
        markdown = (
            "## Could not verify\n\n"
            "No claims could be synthesized from the retrieved papers for this "
            "question. The corpus does not appear to contain relevant coverage.\n"
        )
        trace.record(
            tool="validate", input_summary=f"synthesis_id={synthesis_id}",
            output_summary="0 claims to validate (empty synthesis)",
            model=None, input_tokens=0, output_tokens=0, latency_ms=0.0, cost_usd=0.0,
        )
        return {"status": "success", "final_markdown": markdown, "supported_count": 0, "unsupported_count": 0}

    papers_by_id = {p["paper_id"]: p for p in papers}
    start = time.perf_counter()
    claims_block = "\n\n".join(
        f"claim_id: {c.id}\nclaim_text: {c.text}\ncited_paper_ids: {c.paper_ids}\n"
        + "\n".join(
            f"--- source text for {pid} ---\n{papers_by_id[pid]['abstract']}"
            for pid in c.paper_ids
            if pid in papers_by_id
        )
        for c in result.claims
    )

    response = _genai_client().models.generate_content(
        model=_config.model_validation,
        contents=claims_block,
        config=types.GenerateContentConfig(
            system_instruction=_VALIDATION_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ValidationResult,
            temperature=0.0,
        ),
    )
    validation: ValidationResult = response.parsed
    verdicts = {v.id: v for v in validation.results}

    supported_lines, unsupported_lines = [], []
    claim_supported: dict[int, bool] = {}
    for c in result.claims:
        verdict = verdicts.get(c.id)
        # Link to the paper's real arXiv URL when we actually retrieved it,
        # so a reader can click through and verify the claim themselves.
        # A cited id with no URL to link to (never retrieved) is left as
        # plain bracketed text, it's flagged as a fabrication signal below,
        # not something to make clickable.
        cite = " ".join(
            f"[{pid}]({papers_by_id[pid]['url']})" if pid in papers_by_id else f"[{pid}]"
            for pid in c.paper_ids
        )
        unknown_ids = [pid for pid in c.paper_ids if pid not in papers_by_id]
        if unknown_ids:
            unsupported_lines.append(
                f"- {c.text} (cited {cite}, but {unknown_ids} were never retrieved)"
            )
            claim_supported[c.id] = False
        elif verdict is not None and verdict.supported:
            supported_lines.append(f"- {c.text} {cite}")
            claim_supported[c.id] = True
        else:
            reason = verdict.reasoning if verdict else "not checked"
            unsupported_lines.append(f"- {c.text} (cited {cite}: {reason})")
            claim_supported[c.id] = False

    parts = []
    if supported_lines:
        parts.append("## Findings\n\n" + "\n".join(supported_lines))
    if unsupported_lines:
        parts.append(
            "## Could not verify\n\n"
            "The following claims were not supported by the retrieved source "
            "text and are reported as a gap rather than included above:\n\n"
            + "\n".join(unsupported_lines)
        )
    if not supported_lines:
        parts.insert(
            0,
            "The corpus does not contain enough directly supporting evidence "
            "to answer this question with traceable citations.",
        )
    markdown = "\n\n".join(parts) + "\n"

    latency_ms = (time.perf_counter() - start) * 1000
    in_tok = response.usage_metadata.prompt_token_count or 0
    out_tok = response.usage_metadata.candidates_token_count or 0
    cost = estimate_cost_usd(_config.model_validation, in_tok, out_tok)
    trace.record(
        tool="validate",
        input_summary=f"synthesis_id={synthesis_id} retrieval_id={retrieval_id}",
        output_summary=f"{len(supported_lines)} supported, {len(unsupported_lines)} unsupported",
        model=_config.model_validation,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        cost_usd=cost,
    )
    return {
        "status": "success",
        "final_markdown": markdown,
        "supported_count": len(supported_lines),
        "unsupported_count": len(unsupported_lines),
        # Per-claim verdicts, keyed by claim id — not needed by the
        # orchestrator (which only uses final_markdown) but read directly by
        # eval/run_eval.py to score validate()'s accuracy against an
        # independent judge.
        "claim_supported": claim_supported,
    }
