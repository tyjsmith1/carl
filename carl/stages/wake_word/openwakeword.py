"""OpenWakeWord stage. See docs/specs/pipeline.md.

Audio-driven wake-word engine. The audio router feeds this via `feed()` on
the asyncio dispatch loop (not the audio thread — see router.py). We buffer
router-native 512-sample (32 ms) frames until we have 1280 samples (80 ms),
which is openwakeword's prediction window, run inference, and fire the
wake callback if a score crosses `threshold`.

Debouncing: OpenWakeWord may score above threshold on multiple consecutive
80 ms predictions for a single utterance. We suppress refires within
`DEBOUNCE_SECONDS` of the last firing to avoid double-triggering the pipeline.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import structlog

from carl.stages.wake_word.base import OnWakeCallback

log = structlog.get_logger()


class OpenWakeWord:
    SAMPLE_RATE = 16000
    PREDICT_SAMPLES = 1280      # 80 ms at 16 kHz
    PREDICT_BYTES = PREDICT_SAMPLES * 2   # int16 = 2 bytes
    DEBOUNCE_SECONDS = 1.5      # min gap between wake events

    def __init__(self, config: Any):
        # Lazy import — openwakeword pulls tflite-runtime / onnxruntime and
        # downloads model files on first init; don't pay this at import time.
        import openwakeword
        from openwakeword.model import Model

        # Idempotent — no-op if the pretrained models are already cached.
        openwakeword.utils.download_models()

        model_arg = config.model_path or config.model_name
        self._model = Model(
            wakeword_models=[model_arg],
            inference_framework=config.inference_framework,
        )
        self._threshold = config.threshold
        self._configured_name = config.model_name

        self._on_wake: OnWakeCallback | None = None
        self._buffer = bytearray()
        self._last_fired_at = 0.0

        log.info("openwakeword_loaded",
                 model=config.model_name,
                 threshold=config.threshold,
                 framework=config.inference_framework)

    async def start(self, on_wake: OnWakeCallback) -> None:
        self._on_wake = on_wake
        # Human-visible line so the user knows Carl is actually listening.
        phrase = self._configured_name.replace("_", " ")
        print(f'IDLE. listening for "{phrase}"... (Ctrl-C to quit)', flush=True)

    async def stop(self) -> None:
        # Nothing to clean up — no native handles, no threads owned by us.
        return

    def wants_audio(self) -> bool:
        return True

    def feed(self, pcm_i16: bytes) -> None:
        """Called on the asyncio dispatch loop from AudioRouter. Buffer to
        80 ms windows, run inference, fire wake on threshold crossing."""
        self._buffer.extend(pcm_i16)
        while len(self._buffer) >= self.PREDICT_BYTES:
            chunk = bytes(self._buffer[: self.PREDICT_BYTES])
            del self._buffer[: self.PREDICT_BYTES]
            arr = np.frombuffer(chunk, dtype=np.int16)
            scores = self._model.predict(arr)
            if not scores:
                continue
            # OpenWakeWord returns {model_key: score}; pick the max in case
            # multiple models are loaded (future-proofing).
            key, score = max(scores.items(), key=lambda kv: kv[1])
            if score > self._threshold:
                self._fire_wake(key, float(score))

    def _fire_wake(self, key: str, score: float) -> None:
        now = time.monotonic()
        if now - self._last_fired_at < self.DEBOUNCE_SECONDS:
            return
        self._last_fired_at = now

        log.info("wake_detected", model=key, score=round(score, 3))
        if self._on_wake is None:
            return

        # We're on the asyncio loop (dispatch task) — safe to create_task
        # directly. The task is fire-and-forget because AudioRouter's dispatch
        # continues immediately with the next frame.
        asyncio.create_task(self._on_wake({"keyword": key, "score": score}))
