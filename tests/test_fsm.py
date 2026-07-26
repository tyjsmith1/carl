"""FSM data-type tests. The transition dispatch lives in Pipeline; those
transitions are exercised by tests/test_pipeline_e2e_mock.py."""

from __future__ import annotations

from carl.pipeline.fsm import PipelineEvent, State


def test_states_enumerate() -> None:
    assert {s.value for s in State} == {"IDLE", "RECORDING", "THINKING", "SPEAKING", "ERROR"}


def test_event_construction() -> None:
    e = PipelineEvent(type="wake_detected", payload={"keyword": "hey_carl"})
    assert e.type == "wake_detected"
    assert e.payload["keyword"] == "hey_carl"


def test_event_default_payload_is_empty() -> None:
    e = PipelineEvent(type="tick")
    assert e.payload == {}
