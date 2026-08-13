"""Per-run trace recorder.

Every tool call and every underlying Gemini call is recorded here so the
demo can print the exact sequence of steps, the model used at each step,
token counts, and latency — the "Agent Observability" talking point made
concrete rather than asserted.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TraceEntry:
    tool: str
    input_summary: str
    output_summary: str
    model: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


@dataclass
class TraceRecorder:
    entries: list[TraceEntry] = field(default_factory=list)

    def reset(self) -> None:
        self.entries.clear()

    def record(
        self,
        tool: str,
        input_summary: str,
        output_summary: str,
        model: str | None,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        self.entries.append(
            TraceEntry(
                tool=tool,
                input_summary=input_summary,
                output_summary=output_summary,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            )
        )

    @contextmanager
    def timed(self):
        start = time.perf_counter()
        holder: dict = {}
        yield holder
        holder["latency_ms"] = (time.perf_counter() - start) * 1000

    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(e.input_tokens for e in self.entries),
            sum(e.output_tokens for e in self.entries),
        )

    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def render(self) -> str:
        if not self.entries:
            return "(no trace entries)"

        headers = ["#", "tool", "model", "in_tok", "out_tok", "latency_ms", "cost_usd", "output"]
        rows = []
        for i, e in enumerate(self.entries, 1):
            rows.append(
                [
                    str(i),
                    e.tool,
                    e.model or "-",
                    str(e.input_tokens),
                    str(e.output_tokens),
                    f"{e.latency_ms:.0f}",
                    f"{e.cost_usd:.5f}",
                    e.output_summary[:60],
                ]
            )

        widths = [
            max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
        ]
        lines = []
        lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        lines.append("  ".join("-" * w for w in widths))
        for r in rows:
            lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))

        in_tok, out_tok = self.total_tokens()
        lines.append("")
        lines.append(
            f"Totals: {in_tok} input tokens, {out_tok} output tokens, "
            f"${self.total_cost_usd():.5f}"
        )
        return "\n".join(lines)


trace = TraceRecorder()
