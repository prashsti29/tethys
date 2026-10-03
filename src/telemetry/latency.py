"""
Latency instrumentation for the voice pipeline.

Timestamps every stage (VAD end → STT final → LLM first token → TTS first byte
→ playback start) and reports P50/P95 time-to-first-audio against budget.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Latency budget (milliseconds)
LATENCY_BUDGET_MS: Dict[str, float] = {
    "vad_end→stt_final": 150.0,
    "stt_final→llm_first_token": 250.0,
    "llm_first_token→tts_first_byte": 100.0,
    "tts_first_byte→playback_start": 100.0,
    "vad_end→playback_start": 800.0,  # total end-to-end
}

# Ordered list of stage names for reporting
STAGE_ORDER = [
    "vad_end",
    "stt_final",
    "llm_first_token",
    "tts_first_byte",
    "playback_start",
]


@dataclass
class LatencyTracer:
    """Per-turn latency tracer.  Call ``record(stage)`` at each pipeline boundary."""

    turn_id: int = 0
    _stamps: Dict[str, float] = field(default_factory=dict)

    # ── public API ──────────────────────────────────────────────

    def record(self, stage: str) -> None:
        """Stamp the current time for *stage*.  First call wins (no overwrites)."""
        if stage not in self._stamps:
            self._stamps[stage] = time.perf_counter()

    def elapsed_ms(self, start: str, end: str) -> Optional[float]:
        """Return milliseconds between two recorded stages, or None."""
        s = self._stamps.get(start)
        e = self._stamps.get(end)
        if s is None or e is None:
            return None
        return (e - s) * 1000.0

    def report(self) -> Dict[str, Optional[float]]:
        """Return a dict of per-segment deltas (ms) plus total end-to-end."""
        segments: Dict[str, Optional[float]] = {}
        for i in range(len(STAGE_ORDER) - 1):
            key = f"{STAGE_ORDER[i]}→{STAGE_ORDER[i + 1]}"
            segments[key] = self.elapsed_ms(STAGE_ORDER[i], STAGE_ORDER[i + 1])

        # Total
        segments["vad_end→playback_start"] = self.elapsed_ms("vad_end", "playback_start")
        return segments

    def log_report(self) -> None:
        """Log a human-readable latency breakdown at INFO level."""
        segments = self.report()
        parts: List[str] = []
        for key, ms in segments.items():
            budget = LATENCY_BUDGET_MS.get(key)
            if ms is not None:
                flag = ""
                if budget and ms > budget:
                    flag = " ⚠ OVER BUDGET"
                parts.append(f"  {key}: {ms:7.1f} ms (budget {budget or '—'} ms){flag}")
            else:
                parts.append(f"  {key}: —")

        header = f"[Latency] Turn #{self.turn_id}"
        logger.info(f"\n{header}\n" + "\n".join(parts))

    def reset(self) -> None:
        self._stamps.clear()


class LatencyAggregator:
    """Accumulates per-stage latencies across turns and computes percentiles."""

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._turn_count: int = 0

    def ingest(self, tracer: LatencyTracer) -> None:
        """Ingest one completed tracer's report into the aggregator."""
        self._turn_count += 1
        for key, ms in tracer.report().items():
            if ms is not None:
                self._history[key].append(ms)

    def percentile(self, segment: str, p: float) -> Optional[float]:
        """Return the *p*-th percentile (0–100) for *segment*, or None."""
        vals = self._history.get(segment)
        if not vals:
            return None
        return float(np.percentile(vals, p))

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Return {segment: {p50, p95, budget, over_budget}} for all segments."""
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for seg in self._history:
            p50 = self.percentile(seg, 50)
            p95 = self.percentile(seg, 95)
            budget = LATENCY_BUDGET_MS.get(seg)
            out[seg] = {
                "p50_ms": round(p50, 1) if p50 is not None else None,
                "p95_ms": round(p95, 1) if p95 is not None else None,
                "budget_ms": budget,
                "p95_over_budget": (p95 > budget) if (p95 is not None and budget) else None,
            }
        return out

    def log_summary(self) -> None:
        """Log aggregate latency summary."""
        if self._turn_count == 0:
            return
        lines = [f"[Latency Aggregator] {self._turn_count} turns"]
        for seg, stats in self.summary().items():
            flag = " ⚠ OVER" if stats.get("p95_over_budget") else ""
            lines.append(
                f"  {seg}: P50={stats['p50_ms']} ms  P95={stats['p95_ms']} ms  "
                f"(budget {stats['budget_ms'] or '—'} ms){flag}"
            )
        logger.info("\n".join(lines))
