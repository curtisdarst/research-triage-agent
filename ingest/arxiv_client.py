"""Thin client for the arXiv API (export.arxiv.org) — no auth required.

Query is scoped to this project's demo topic: the effect of prompt
engineering patterns on LLM-generated code quality (the corpus backing
the dissertation-adjacent demo, see README "Topic choice"). Categories
cs.SE (software engineering), cs.CL (computational linguistics), and
cs.AI carry the relevant literature.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlencode

import feedparser
import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
PAGE_SIZE = 100
REQUEST_DELAY_SECONDS = 3.0  # arXiv API etiquette: no more than 1 req / 3s

SEARCH_QUERY = (
    "(cat:cs.SE OR cat:cs.CL OR cat:cs.AI) AND ("
    # Every disjunct is deliberately anchored to a code-related term.
    # Earlier versions of this query allowed general prompt-engineering
    # terms to stand alone, which pulled in unrelated domains (e.g.
    # clinical/medical LLM prompting papers) purely on phrase overlap —
    # this corpus is specifically about code quality, not prompting in
    # general.
    '(abs:"prompt engineering" AND abs:"code") OR '
    '(abs:"prompt pattern" AND abs:"code") OR '
    '(abs:"chain-of-thought" AND abs:"code generation") OR '
    '(abs:"chain of thought" AND abs:"code generation") OR '
    '(abs:"few-shot prompting" AND abs:"code generation") OR '
    '(abs:"in-context learning" AND abs:"code generation") OR '
    '(abs:"code generation" AND abs:"large language model") OR '
    '(abs:"code generation" AND abs:"prompt") OR '
    '(abs:"code quality" AND abs:"language model") OR '
    'abs:"LLM-generated code" OR '
    '(abs:"cyclomatic complexity" AND abs:"language model") OR '
    '(abs:"defect density" AND abs:"code")'
    ")"
)


@dataclass
class Paper:
    id: str
    title: str
    authors: str
    published_date: str  # YYYY-MM-DD
    abstract: str
    url: str


def _arxiv_id(entry_id_url: str) -> str:
    # entry.id looks like http://arxiv.org/abs/2508.01234v2 -> keep 2508.01234
    tail = entry_id_url.rsplit("/", 1)[-1]
    return tail.split("v")[0] if "v" in tail.rsplit(".", 1)[-1] else tail


def fetch_papers(max_results: int = 400) -> list[Paper]:
    """Fetch up to max_results papers matching SEARCH_QUERY, newest first."""
    papers: list[Paper] = []
    start = 0
    while len(papers) < max_results:
        batch_size = min(PAGE_SIZE, max_results - len(papers))
        params = {
            "search_query": SEARCH_QUERY,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)

        if not feed.entries:
            break

        for entry in feed.entries:
            authors = ", ".join(a.name for a in getattr(entry, "authors", []))
            published_date = entry.published[:10]  # ISO date prefix
            papers.append(
                Paper(
                    id=_arxiv_id(entry.id),
                    title=" ".join(entry.title.split()),
                    authors=authors,
                    published_date=published_date,
                    abstract=" ".join(entry.summary.split()),
                    url=entry.id.replace("http://", "https://"),
                )
            )

        start += batch_size
        if len(feed.entries) < batch_size:
            break  # exhausted results
        time.sleep(REQUEST_DELAY_SECONDS)

    return papers
