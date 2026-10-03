"""Unit tests for src.telemetry.latency — LatencyTracer & LatencyAggregator."""

import time
import pytest
from src.telemetry.latency import LatencyTracer, LatencyAggregator, STAGE_ORDER


class TestLatencyTracer:
    """Tests for the per-turn LatencyTracer."""

    def test_record_and_elapsed(self):
        tracer = LatencyTracer(turn_id=1)
        tracer.record("vad_end")
        time.sleep(0.01)  # ~10 ms
        tracer.record("stt_final")

        ms = tracer.elapsed_ms("vad_end", "stt_final")
        assert ms is not None
        assert ms >= 5.0  # should be ≥ 10 ms but allow jitter

    def test_first_call_wins(self):
        tracer = LatencyTracer(turn_id=2)
        tracer.record("vad_end")
        first_stamp = tracer._stamps["vad_end"]
        time.sleep(0.01)
        tracer.record("vad_end")  # should NOT overwrite
        assert tracer._stamps["vad_end"] == first_stamp

    def test_elapsed_returns_none_for_missing_stage(self):
        tracer = LatencyTracer(turn_id=3)
        tracer.record("vad_end")
        assert tracer.elapsed_ms("vad_end", "stt_final") is None
        assert tracer.elapsed_ms("nonexistent", "vad_end") is None

    def test_report_all_stages(self):
        tracer = LatencyTracer(turn_id=4)
        for stage in STAGE_ORDER:
            tracer.record(stage)
            time.sleep(0.005)

        report = tracer.report()
        # Should have N-1 segment keys + 1 total key
        assert "vad_end→stt_final" in report
        assert "vad_end→playback_start" in report

        # All values should be non-negative
        for key, val in report.items():
            assert val is not None
            assert val >= 0.0

    def test_report_with_partial_stages(self):
        tracer = LatencyTracer(turn_id=5)
        tracer.record("vad_end")
        tracer.record("stt_final")
        # Skip later stages

        report = tracer.report()
        assert report["vad_end→stt_final"] is not None
        assert report["stt_final→llm_first_token"] is None  # not stamped
        assert report["vad_end→playback_start"] is None  # not stamped

    def test_reset_clears_stamps(self):
        tracer = LatencyTracer(turn_id=6)
        tracer.record("vad_end")
        tracer.reset()
        assert tracer.elapsed_ms("vad_end", "vad_end") is None

    def test_log_report_does_not_raise(self):
        """log_report should work even with incomplete data."""
        tracer = LatencyTracer(turn_id=7)
        tracer.record("vad_end")
        tracer.log_report()  # should not raise


class TestLatencyAggregator:
    """Tests for the cross-turn LatencyAggregator."""

    def _make_tracer(self, turn_id: int, delay_s: float = 0.005) -> LatencyTracer:
        tracer = LatencyTracer(turn_id=turn_id)
        for stage in STAGE_ORDER:
            tracer.record(stage)
            time.sleep(delay_s)
        return tracer

    def test_ingest_and_percentile(self):
        agg = LatencyAggregator()
        for i in range(5):
            agg.ingest(self._make_tracer(i + 1))

        p50 = agg.percentile("vad_end→stt_final", 50)
        p95 = agg.percentile("vad_end→stt_final", 95)
        assert p50 is not None
        assert p95 is not None
        assert p95 >= p50

    def test_percentile_returns_none_for_empty(self):
        agg = LatencyAggregator()
        assert agg.percentile("vad_end→stt_final", 50) is None

    def test_summary_keys(self):
        agg = LatencyAggregator()
        agg.ingest(self._make_tracer(1))
        summary = agg.summary()

        assert "vad_end→stt_final" in summary
        entry = summary["vad_end→stt_final"]
        assert "p50_ms" in entry
        assert "p95_ms" in entry
        assert "budget_ms" in entry
        assert "p95_over_budget" in entry

    def test_log_summary_does_not_raise(self):
        agg = LatencyAggregator()
        agg.log_summary()  # empty — should not raise
        agg.ingest(self._make_tracer(1))
        agg.log_summary()  # with data — should not raise
