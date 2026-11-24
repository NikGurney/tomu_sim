"""Tests for readable candidate and halt traceability."""

from tomu.config import DEFAULT_CONFIG
from tomu.inference import Mode, backward_inference
from tomu.scenarios import build_variant
from tomu.trace import InferenceTrace


def test_trace_records_candidates_confidence_and_halt() -> None:
    scenario = build_variant("trusted", DEFAULT_CONFIG)
    trace = InferenceTrace()
    backward_inference(
        scenario.lewm.agents[scenario.target], scenario.behavior, "contents",
        Mode(), DEFAULT_CONFIG.s1_default, DEFAULT_CONFIG, trace=trace,
    )
    transcript = trace.transcript()
    assert [entry.candidate for entry in trace.entries] == ["chocolate", "popcorn"]
    assert "confidence" in transcript
    assert "halt:" in transcript
