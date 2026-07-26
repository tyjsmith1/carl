"""Config schema + loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from carl.config import Config, deep_merge
from carl.config.schema import DeepgramConfig, OpenWakeWordConfig


def test_defaults_all_mock() -> None:
    cfg = Config()
    assert cfg.providers.wake_word == "mock"
    assert cfg.providers.vad == "mock"
    assert cfg.providers.stt == "mock"
    assert cfg.providers.reasoner == "mock"
    assert cfg.providers.tts == "mock"
    assert cfg.providers.search == "mock"


def test_load_canonical_yaml() -> None:
    cfg = Config.load("config/carl.yaml")
    assert cfg.version == 1
    assert cfg.storage.sqlite_path == "~/.carl/carl.db"
    # Real-provider blocks should be populated even when unused.
    assert cfg.stt.deepgram is not None
    assert cfg.stt.deepgram.model == "nova-3"


def test_local_override_merges(tmp_path: Path) -> None:
    base = tmp_path / "carl.yaml"
    base.write_text(yaml.safe_dump({
        "providers": {"reasoner": "mock"},
        "conversation": {"history_turns": 10},
    }))
    local = tmp_path / "carl.local.yaml"
    local.write_text(yaml.safe_dump({
        "conversation": {"history_turns": 5},   # deep-merged, not replaced
    }))
    cfg = Config.load(base)
    assert cfg.conversation.history_turns == 5
    assert cfg.providers.reasoner == "mock"


def test_deep_merge_preserves_disjoint_keys() -> None:
    merged = deep_merge(
        {"a": {"x": 1, "y": 2}, "b": 10},
        {"a": {"y": 3, "z": 4}},
    )
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}, "b": 10}


def test_extra_keys_rejected() -> None:
    with pytest.raises(Exception):  # noqa: B017 — pydantic raises ValidationError, subclass
        Config.model_validate({"nonsense_top_level_key": 1})


def test_openwakeword_config_defaults() -> None:
    """OpenWakeWordConfig has all-defaults construction (unlike Porcupine which
    required a keyword_path); the pre-trained hey_jarvis phrase is the MVP default."""
    cfg = OpenWakeWordConfig()
    assert cfg.model_name == "hey_jarvis"
    assert cfg.model_path is None
    assert 0.0 <= cfg.threshold <= 1.0
    assert cfg.inference_framework == "onnx"


def test_deepgram_config_defaults() -> None:
    dg = DeepgramConfig()
    assert dg.model == "nova-3"
    assert dg.language == "en-US"
    assert dg.endpointing_ms == 300
    assert dg.utterance_end_ms == 1000
    assert dg.interim_results is True
