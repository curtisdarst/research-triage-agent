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
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent import run_query
from agent.trace import trace
from run_demo import GAP_QUERY, HAPPY_QUERY

app = FastAPI(title="Research Triage Agent")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

_QUERY_LOCK = asyncio.Lock()


class AskRequest(BaseModel):
    mode: str  # "happy" | "gap" | "custom"
    question: str | None = None


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
        answer = await run_query(question)
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
