"""Stage factory: maps config.providers.* → concrete implementation instances.

Adding a new implementation:
1. Write the class implementing the stage's Protocol (in stages/<stage>/<name>.py).
2. Add the name to the corresponding Literal in carl/config/schema.py.
3. Register it in the _*_IMPLS dict below.

Real (non-mock) implementations arrive in Milestone 2+. Their names are
registered here as sentinels that raise NotImplementedError on instantiation
until the real class lands — so misconfiguration fails at startup with a
useful message rather than mysteriously later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from carl.config.schema import Config
from carl.stages.reasoner import EchoReasoner, Reasoner
from carl.stages.stt import STT, CannedTranscriptSTT
from carl.stages.tts import TTS, PrintTTS
from carl.stages.vad import VAD, NoopVAD
from carl.stages.wake_word import StdinTriggeredWakeWord, WakeWord


def _not_yet(stage: str, name: str) -> Callable[[Any], Any]:
    def _factory(config: Any) -> Any:
        raise NotImplementedError(
            f"{stage} provider '{name}' is not implemented yet. "
            f"Set providers.{stage}: mock in your config, or land the real impl."
        )
    return _factory


def _openwakeword_factory(config: Any) -> WakeWord:
    # Lazy import — heavy deps (tflite/onnx, model download).
    from carl.stages.wake_word.openwakeword import OpenWakeWord
    return OpenWakeWord(config)


_WAKE_WORD_IMPLS: dict[str, Callable[[Any], WakeWord]] = {
    "mock": StdinTriggeredWakeWord,
    "openwakeword": _openwakeword_factory,
}

_VAD_IMPLS: dict[str, Callable[[Any], VAD]] = {
    "mock": NoopVAD,
    "silero": _not_yet("vad", "silero"),                     # Milestone 2
    "webrtc": _not_yet("vad", "webrtc"),
}

_STT_IMPLS: dict[str, Callable[[Any], STT]] = {
    "mock": CannedTranscriptSTT,
    "deepgram": _not_yet("stt", "deepgram"),                 # Milestone 2
    "whisper_api": _not_yet("stt", "whisper_api"),
    "whisper_local": _not_yet("stt", "whisper_local"),
}

_REASONER_IMPLS: dict[str, Callable[[Any], Reasoner]] = {
    "mock": EchoReasoner,
    "claude_sonnet": _not_yet("reasoner", "claude_sonnet"),  # Milestone 2
    "claude_opus": _not_yet("reasoner", "claude_opus"),
    "claude_haiku": _not_yet("reasoner", "claude_haiku"),
}

_TTS_IMPLS: dict[str, Callable[[Any], TTS]] = {
    "mock": PrintTTS,
    "openai": _not_yet("tts", "openai"),                     # Milestone 2
    "elevenlabs": _not_yet("tts", "elevenlabs"),
    "piper": _not_yet("tts", "piper"),
}


@dataclass(frozen=True, slots=True)
class Stages:
    wake_word: WakeWord
    vad: VAD
    stt: STT
    reasoner: Reasoner
    tts: TTS


def build_stages(config: Config) -> Stages:
    """Instantiate the stage impls named in config.providers.*.
    Each impl is passed its own config subsection (or None if not set)."""

    def _sub(section: Any, name: str) -> Any:
        # e.g. section=config.wake_word, name="porcupine" → config.wake_word.porcupine
        # For "mock", we pass None — mocks don't need config.
        if name == "mock":
            return None
        return getattr(section, name, None)

    return Stages(
        wake_word=_WAKE_WORD_IMPLS[config.providers.wake_word](
            _sub(config.wake_word, config.providers.wake_word)
        ),
        vad=_VAD_IMPLS[config.providers.vad](
            _sub(config.vad, config.providers.vad)
        ),
        stt=_STT_IMPLS[config.providers.stt](
            _sub(config.stt, config.providers.stt)
        ),
        reasoner=_REASONER_IMPLS[config.providers.reasoner](
            _sub(config.reasoner, config.providers.reasoner)
        ),
        tts=_TTS_IMPLS[config.providers.tts](
            _sub(config.tts, config.providers.tts)
        ),
    )


__all__ = ["Stages", "build_stages"]
