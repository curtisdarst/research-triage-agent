"""Minimal Cloud Run web front end (Tier 2).

Same agent as run_demo.py, over HTTP instead of a CLI. Kept deliberately
small: one FastAPI app, one static HTML page, no build step, no frontend
framework.

Concurrency note: agent/tools.py and agent/trace.py hold per-query state in
module-level singletons (the retrieval/synthesis handle store, the trace
recorder), reset at the start of each run_query() call. That's fine for a
single-request CLI process, but two concurrent requests in the same
process would corrupt each other's state. This app is deployed with Cloud
Run `--concurrency=1` (one request per container instance; concurrent load
gets separate instances, each a fresh process) as the primary guard, and
_QUERY_LOCK below as defense in depth in case that ever changes. Neither
is a real fix, request-scoped state instead of module globals is the
actual fix, and is out of scope for this demo. See README "Production
hardening".
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import bigquery
from google.genai.errors import ClientError
from pydantic import BaseModel

from agent.agent import run_query
from agent.config import load_config
from agent.trace import trace
from ingest.arxiv_client import SEARCH_QUERY
from ingest.ingest_arxiv import clear_corpus, run_ingest
from run_demo import GAP_QUERY, HAPPY_QUERY

app = FastAPI(title="Research Triage Agent")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

_QUERY_LOCK = asyncio.Lock()
# Separate from _QUERY_LOCK: ingest doesn't touch agent/tools.py's or
# agent/trace.py's module-level state at all (it writes straight to
# BigQuery), so it isn't blocked by an in-flight query. It gets its own
# lock purely to stop two concurrent ingests from double-fetching arXiv
# and double-billing embeddings, not for agent-state safety.
_INGEST_LOCK = asyncio.Lock()
MAX_INGEST_RESULTS = 2000  # bounds worst-case wall clock and cost per request


class AskRequest(BaseModel):
    mode: str  # "happy" | "gap" | "custom"
    question: str | None = None


class IngestRequest(BaseModel):
    max_results: int = 400
    search_query: str | None = None  # None = use the built-in default


@app.get("/")
def index() -> FileResponse:
    return FileResponse("web/static/index.html")


@app.get("/status")
def status() -> dict:
    # Not /healthz: that path is intercepted by Google's frontend before
    # reaching the container on Cloud Run, confirmed by testing (every
    # other path, including a near-identical one, reaches this app
    # correctly; /healthz alone returns a Google-branded 404 HTML page
    # instead of this app's JSON 404).
    return {"status": "ok"}


@app.post("/api/ask")
async def ask(req: AskRequest) -> JSONResponse:
    if req.mode == "happy":
        question = HAPPY_QUERY
    elif req.mode == "gap":
        question = GAP_QUERY
    elif req.mode == "custom" and req.question and req.question.strip():
        question = req.question.strip()
    else:
        return JSONResponse({"error": "invalid request"}, status_code=400)

    async with _QUERY_LOCK:
        start = time.perf_counter()
        try:
            answer = await run_query(question)
        except ClientError as e:
            if e.code == 429:
                return JSONResponse(
                    {
                        "error": "Rate limited by Gemini/Vertex (429 RESOURCE_EXHAUSTED). "
                        "This project's quota is easy to hit with back-to-back queries. "
                        "Wait a bit and try again. See README 'Known limitations'."
                    },
                    status_code=429,
                )
            return JSONResponse({"error": f"Gemini/Vertex API error: {e}"}, status_code=502)
        except Exception as e:  # noqa: BLE001 - surfaced to the caller as JSON, not a bare 500 page
            return JSONResponse({"error": f"Unexpected server error: {e}"}, status_code=500)
        wall_seconds = time.perf_counter() - start

        entries = [
            {
                "tool": e.tool,
                "model": e.model,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "latency_ms": round(e.latency_ms),
                "cost_usd": round(e.cost_usd, 5),
                "output_summary": e.output_summary,
            }
            for e in trace.entries
        ]
        in_tok, out_tok = trace.total_tokens()

    return JSONResponse(
        {
            "question": question,
            "answer": answer,
            "trace": entries,
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
            "total_cost_usd": round(trace.total_cost_usd(), 5),
            "wall_seconds": round(wall_seconds, 1),
        }
    )


@app.post("/api/ingest", response_model=None)
async def ingest(req: IngestRequest) -> StreamingResponse | JSONResponse:
    """Streams progress as newline-delimited SSE events (fetch/embed/upsert
    lines from run_ingest's on_progress, as they happen) rather than
    blocking silently for the whole run. The frontend reads this with a
    plain fetch() + stream reader, not EventSource, since EventSource
    can't send a POST body and search_query needs one.

    Event shapes: {"type": "progress", "line": str},
    {"type": "done", ...same fields the old JSON response had...},
    {"type": "error", "error": str}.
    """
    if req.max_results < 1 or req.max_results > MAX_INGEST_RESULTS:
        return JSONResponse(
            {"error": f"max_results must be between 1 and {MAX_INGEST_RESULTS}"},
            status_code=400,
        )

    if _INGEST_LOCK.locked():
        return JSONResponse(
            {"error": "An ingest is already running. Wait for it to finish before starting another."},
            status_code=409,
        )

    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def on_progress(line: str) -> None:
            # Called from the to_thread worker thread below, not the event
            # loop thread, so the queue needs a threadsafe handoff.
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"type": "progress", "line": line}))

        async def run() -> None:
            async with _INGEST_LOCK:
                start = time.perf_counter()
                try:
                    # run_ingest is synchronous (requests, time.sleep, sync
                    # genai calls) and can take minutes for a large corpus;
                    # to_thread keeps it off the event loop rather than
                    # blocking the whole app, including /status.
                    result = await asyncio.to_thread(
                        run_ingest,
                        req.max_results,
                        search_query=req.search_query,
                        on_progress=on_progress,
                    )
                    wall_seconds = time.perf_counter() - start
                    await queue.put(
                        json.dumps(
                            {
                                "type": "done",
                                "fetched": result["fetched"],
                                "affected": result["affected"],
                                "total_rows": result["total_rows"],
                                "cost_usd": result["cost_usd"],
                                "tokens": result["tokens"],
                                "wall_seconds": round(wall_seconds, 1),
                            }
                        )
                    )
                except ClientError as e:
                    if e.code == 429:
                        msg = "Rate limited by Gemini/Vertex (429 RESOURCE_EXHAUSTED) while embedding. Wait a bit and try again."
                    else:
                        msg = f"Gemini/Vertex API error: {e}"
                    await queue.put(json.dumps({"type": "error", "error": msg}))
                except Exception as e:  # noqa: BLE001 - surfaced to the caller as an SSE event, not a bare 500 page
                    await queue.put(json.dumps({"type": "error", "error": f"Ingest failed: {e}"}))
                finally:
                    await queue.put(None)  # sentinel: stop iterating below

        task = asyncio.create_task(run())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {item}\n\n"
        await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/ingest/default-query")
def default_query() -> JSONResponse:
    """The built-in arXiv search query, so the web UI can prefill an
    editable field with it instead of duplicating it in JS."""
    return JSONResponse({"search_query": SEARCH_QUERY})


@app.get("/api/admin/corpus-count")
def corpus_count() -> JSONResponse:
    config = load_config()
    bq = bigquery.Client(project=config.project_id)
    table_id = f"{config.project_id}.{config.dataset}.{config.table}"
    n = next(iter(bq.query(f"SELECT COUNT(*) AS n FROM `{table_id}`").result()))["n"]
    return JSONResponse({"count": n})


@app.post("/api/admin/clear-corpus")
async def admin_clear_corpus() -> JSONResponse:
    """Deletes every row from the papers table. No separate confirmation
    here beyond Cloud Run's own auth check, this endpoint trusts the
    caller (the web UI gates it behind a JS confirm() showing the current
    row count; the CLI's --clear flag has its own text-confirmation
    prompt) rather than re-implementing confirmation server-side too."""
    if _INGEST_LOCK.locked():
        return JSONResponse(
            {"error": "An ingest is currently running. Wait for it to finish first."},
            status_code=409,
        )

    async with _INGEST_LOCK:
        try:
            result = await asyncio.to_thread(clear_corpus, lambda _line: None)
        except Exception as e:  # noqa: BLE001 - surfaced to the caller as JSON, not a bare 500 page
            return JSONResponse({"error": f"Clear failed: {e}"}, status_code=500)

    return JSONResponse({"deleted_rows": result["deleted_rows"]})
