"""Silero VAD stage. See docs/specs/pipeline.md.

Wraps the `silero-vad` package's `VADIterator` for real-time speech-start /
speech-end detection on 16 kHz mono int16 frames.

Two use cases share this class with different tunings (constructed via factory
helpers):
- Utterance-end VAD (RECORDING → THINKING trigger): `min_silence_duration_ms=700`
  so mid-sentence pauses don't cut the utterance short.
- Barge-in VAD (SPEAKING → RECORDING trigger, Milestone 5): 150 ms silence,
  fast enough to feel responsive when interrupting the assistant.

Note on floats: Silero wants float32 in [-1, 1]. Router hands us int16 bytes;
we convert per-frame with np.frombuffer + divide by 32768. Silero's fixed
chunk size at 16 kHz is 512 samples, which matches router.FRAME_SIZE exactly.
"""

from __future__ import annotations

import numpy as np
import structlog

from carl.config.schema import SileroConfig
from carl.stages.vad.base import SpeechEnd, SpeechStart, VADEvent

log = structlog.get_logger()


class SileroVAD:
    """Wraps silero-vad's VADIterator. Instantiate once per role
    (utterance-end vs barge-in) — VADIterator is stateful across calls."""

    def __init__(self, silero_config: SileroConfig, *, min_silence_ms: int):
        # Lazy imports — silero_vad pulls torch, several hundred MB.
        from silero_vad import VADIterator, load_silero_vad

        self._model = load_silero_vad(onnx=False)
        self._iter = VADIterator(
            self._model,
            threshold=silero_config.threshold,
            sampling_rate=silero_config.sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=silero_config.speech_pad_ms,
        )
        self.sample_rate = silero_config.sample_rate
        self.frame_length = 512   # Silero requirement at 16 kHz

        # Warm the model with a throwaway inference so the first real
        # frame doesn't pay JIT / graph-init cost.
        silence = np.zeros(self.frame_length, dtype=np.float32)
        self._iter(silence, return_seconds=True)
        self._iter.reset_states()

        log.info("silero_vad_loaded",
                 threshold=silero_config.threshold,
                 min_silence_ms=min_silence_ms,
                 speech_pad_ms=silero_config.speech_pad_ms)

    async def process(self, pcm_i16: bytes) -> VADEvent | None:
        # Convert int16 bytes → float32 in [-1, 1]. Router always hands us
        # exactly 512-sample frames = 1024 bytes; if that ever changes, the
        # frombuffer will still work but Silero may reject non-512 sizes.
        pcm = np.frombuffer(pcm_i16, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._iter(pcm, return_seconds=True)
        if result is None:
            return None
        if "start" in result:
            return SpeechStart(timestamp=float(result["start"]))
        if "end" in result:
            return SpeechEnd(timestamp=float(result["end"]))
        return None

    def reset(self) -> None:
        """Reset VAD state between utterances/sessions so state from a
        prior turn doesn't leak into the next one."""
        self._iter.reset_states()


def build_utterance_end_vad(config: SileroConfig) -> SileroVAD:
    return SileroVAD(config, min_silence_ms=config.min_silence_ms_utterance_end)


def build_bargein_vad(config: SileroConfig) -> SileroVAD:
    """Milestone 5 will actually use this. Milestone 2 doesn't."""
    return SileroVAD(config, min_silence_ms=config.min_silence_ms_bargein)
