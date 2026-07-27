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


def _silero_vad_factory(config: Any) -> VAD:
    from carl.stages.vad.silero import build_utterance_end_vad
    return build_utterance_end_vad(config)


def _deepgram_stt_factory(config: Any) -> STT:
    from carl.stages.stt.deepgram import DeepgramSTT
    return DeepgramSTT(config)


def _claude_reasoner_factory(config: Any) -> Reasoner:
    from carl.stages.reasoner.claude import ClaudeReasoner
    return ClaudeReasoner(config)


def _openai_tts_factory(config: Any) -> TTS:
    from carl.stages.tts.openai import OpenAITTS
    return OpenAITTS(config)


def _elevenlabs_tts_factory(config: Any) -> TTS:
    from carl.stages.tts.elevenlabs import ElevenLabsTTS
    return ElevenLabsTTS(config)


_WAKE_WORD_IMPLS: dict[str, Callable[[Any], WakeWord]] = {
    "mock": StdinTriggeredWakeWord,
    "openwakeword": _openwakeword_factory,
}

_VAD_IMPLS: dict[str, Callable[[Any], VAD]] = {
    "mock": NoopVAD,
    "silero": _silero_vad_factory,
    "webrtc": _not_yet("vad", "webrtc"),
}

_STT_IMPLS: dict[str, Callable[[Any], STT]] = {
    "mock": CannedTranscriptSTT,
    "deepgram": _deepgram_stt_factory,
    "whisper_api": _not_yet("stt", "whisper_api"),
    "whisper_local": _not_yet("stt", "whisper_local"),
}

_REASONER_IMPLS: dict[str, Callable[[Any], Reasoner]] = {
    "mock": EchoReasoner,
    "claude_sonnet": _claude_reasoner_factory,
    "claude_opus": _claude_reasoner_factory,
    "claude_haiku": _claude_reasoner_factory,
}

_TTS_IMPLS: dict[str, Callable[[Any], TTS]] = {
    "mock": PrintTTS,
    "openai": _openai_tts_factory,
    "elevenlabs": _elevenlabs_tts_factory,
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

    def _sub(stage_name: str, section: Any, provider_name: str) -> Any:
        """Return the config block for the selected provider, or raise a clear
        error if the block is missing (rather than let the impl crash later
        with AttributeError on a None config)."""
        if provider_name == "mock":
            return None
        sub = getattr(section, provider_name, None)
        if sub is None:
            raise ValueError(
                f"providers.{stage_name} is '{provider_name}' but "
                f"{stage_name}.{provider_name} block is missing from config. "
                f"Add it to config/carl.yaml (or carl.local.yaml)."
            )
        return sub

    return Stages(
        wake_word=_WAKE_WORD_IMPLS[config.providers.wake_word](
            _sub("wake_word", config.wake_word, config.providers.wake_word)
        ),
        vad=_VAD_IMPLS[config.providers.vad](
            _sub("vad", config.vad, config.providers.vad)
        ),
        stt=_STT_IMPLS[config.providers.stt](
            _sub("stt", config.stt, config.providers.stt)
        ),
        reasoner=_REASONER_IMPLS[config.providers.reasoner](
            _sub("reasoner", config.reasoner, config.providers.reasoner)
        ),
        tts=_TTS_IMPLS[config.providers.tts](
            _sub("tts", config.tts, config.providers.tts)
        ),
    )


__all__ = ["Stages", "build_stages"]
