# Pipeline Spec (Audio + Wake + VAD + STT + TTS)

> **Verification note**: SDK details in this spec were verified against public docs. Porcupine and Silero VAD sections cite the training-data snapshot with `[VERIFY]` markers where a specific claim (version number, tier limit) should be re-confirmed at build time. Deepgram, OpenAI TTS, and Tavily sections are verified against live docs as of 2026-07-25.

## Purpose

The audio pipeline is everything that isn't the LLM: how sound gets from the microphone into Carl's ear, and how Carl's response gets back out. Five stages, each behind an interface, each with a chosen provider.

## Data flow

```
                    ┌─────────────────────────────────────┐
                    │   sounddevice InputStream           │
                    │   16 kHz, int16, mono, blocksize=512│
                    │   Full-duplex; runs continuously    │
                    └────────────┬────────────────────────┘
                                 │ raw PCM frames
                                 ▼
              ┌───────────────────────────────────────────┐
              │   Audio Router (owned by Pipeline FSM)    │
              │   Routes PCM to different consumers per   │
              │   FSM state.                              │
              └────┬──────────────┬──────────────┬────────┘
                   │IDLE          │RECORDING     │SPEAKING
                   ▼              ▼              ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │  Porcupine   │ │  Silero VAD  │ │  Silero VAD  │
           │  wake-word   │ │  + STT       │ │  (bargein)   │
           │  + 500ms ring│ │  streamer    │ │              │
           │  pre-buffer  │ │              │ │              │
           └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                  │                │                │
       wake_detected      speech_end/          speech_start
       (with pre-audio)   final transcript     → bargein signal
                  │                │                │
                  └────────────────┴────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │      Pipeline FSM queue      │
                    └──────────────┬───────────────┘
                                   │
                          (reasoner turn)
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   TTS streamer               │
                    │   sentence chunks in         │
                    │   → PCM chunks out           │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │   Playback Queue             │
                    │   (asyncio.Queue of PCM)     │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  sounddevice OutputStream    │
                    │  24 kHz, int16, mono         │
                    │  .abort() on barge-in        │
                    └──────────────────────────────┘
```

**Key concept: the audio router.** One InputStream runs continuously (opening/closing the mic on every state transition is fragile and adds latency). A single router callback dispatches PCM frames to whichever consumer(s) the current FSM state needs. In `IDLE`, frames go to Porcupine plus a ring buffer. In `RECORDING`, they go to Silero VAD plus the STT streamer. In `SPEAKING`, they go to Silero VAD only (for barge-in detection).

## Sample rate contract

Everything upstream of the router runs at **16 kHz mono int16**:
- Porcupine requires 16 kHz mono int16 (frame length exposed as `porcupine.frame_length`, typically 512 samples). `[VERIFY]`
- Silero VAD requires 16 kHz mono float32 at exactly 512 samples per frame. Router does int16→float32 conversion.
- Deepgram Nova-3 accepts 16 kHz mono PCM natively. No resampling needed.

**TTS output is 24 kHz mono int16.** Runs on a separate `sounddevice.OutputStream`; no conversion required if we request `response_format=pcm` from OpenAI TTS.

Rationale: 16 kHz is the lingua franca of on-device voice models (VAD, wake word, STT), and 24 kHz matches OpenAI TTS's native rate. Two rates but zero resampling on the hot path.

## The audio router

```python
class AudioRouter:
    """Single owner of the mic. Dispatches frames to registered consumers
    based on FSM state."""

    FRAME_SIZE = 512   # samples @ 16kHz = 32ms

    def __init__(self, event_queue, ring_buffer_ms=500):
        self._event_queue = event_queue
        self._ring = collections.deque(maxlen=ring_buffer_ms // 32)  # int16 frames
        self._consumers: dict[str, Callable[[bytes], None]] = {}
        self._active_set: set[str] = set()

    def register(self, name: str, consumer: Callable[[bytes], None]):
        self._consumers[name] = consumer

    def activate(self, *names: str):
        self._active_set = set(names)

    def _on_frame(self, indata, frames, time_info, status):
        # sounddevice callback thread. Convert once, dispatch to active.
        pcm_i16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        self._ring.append(pcm_i16)
        for name in self._active_set:
            self._consumers[name](pcm_i16)

    def replay_ring(self, consumer: Callable[[bytes], None]):
        """Replay ring buffer to a new consumer. Used to prepend the
        ~500ms before wake-word into the STT stream so the wake phrase
        and utterance start aren't cut off."""
        for frame in list(self._ring):
            consumer(frame)
```

The `sounddevice.InputStream` runs at `blocksize=FRAME_SIZE`, so each callback delivers exactly one frame — no re-chunking on the consumer side.

## Provider interfaces

Every stage has an abstract interface. Implementations register via config (`providers:` block).

### Wake word

```python
class WakeWord(Protocol):
    sample_rate: int
    frame_length: int
    async def process(self, pcm_i16: bytes) -> bool: ...
    async def close(self) -> None: ...
```

**OpenWakeWord implementation** (chosen impl — swapped in for Porcupine, which now gates signup behind corporate email):

- Package: `openwakeword` on PyPI (`pip install openwakeword`). Apache-2.0 code; pre-trained models under CC BY-NC-SA 4.0 (fine for personal non-commercial use).
- Init: `Model(wakeword_models=["hey_jarvis"], inference_framework="tflite")`. First run downloads the model files (~few MB) via `openwakeword.utils.download_models()`.
- Audio contract: **16 kHz mono PCM int16**, 80 ms frames (1280 samples per prediction). Note: this is 2.5× Porcupine's 512-sample frame — the router chunk math changes, and we buffer 2-3 router frames per wake prediction.
- Per prediction: `scores = model.predict(pcm_int16_bytes)` returns `{model_name: score}` in `[0.0, 1.0]`. Wake fires when `scores[model_name] > threshold`.
- Cleanup: no explicit `.close()` needed — pure Python + tflite/onnx runtime.

**Wake phrase choice for MVP: `hey_jarvis`** (pre-trained, ships with the package). Custom "hey_carl" model can be trained via OpenWakeWord's Colab notebook (~1 hour of synthetic-sample generation via Piper TTS + fine-tuning). Tracked in `docs/BACKLOG.md` under "Post-Milestone-6 / v2+" and Milestone 2 callouts.

**Threading:** thread-safe per instance. Router callback thread can call `predict` directly. Model inference is ~1-2 ms per frame on a modern laptop CPU — well within audio-callback budget.

**Latency:** wake-detect floor is 80 ms (one frame) + inference (~2 ms) ≈ 82 ms. Vs. Porcupine's ~32 ms. In practice: imperceptible for MVP; if it ever feels laggy, drop `threshold` to reduce false rejects at the cost of more false accepts.

### VAD

```python
class VAD(Protocol):
    sample_rate: int
    frame_length: int   # must be 512 for Silero at 16kHz
    async def process(self, pcm_i16: bytes) -> VADEvent | None:
        """Returns None most calls. Emits SpeechStart/SpeechEnd events."""

@dataclass
class SpeechStart: timestamp: float
@dataclass
class SpeechEnd:   timestamp: float
VADEvent = SpeechStart | SpeechEnd
```

**Silero VAD implementation** (chosen impl):
- Package: `silero-vad` on PyPI (`pip install silero-vad`), currently 5.x line.
- Load: `model = load_silero_vad(onnx=False)` (or `onnx=True` for lighter deploy).
- Wrapper: `VADIterator(model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=700, speech_pad_ms=100)`.
- Per frame: convert int16 → float32 in [-1, 1], call `it(chunk, return_seconds=True)`.
  Returns `None`, `{'start': t}`, or `{'end': t}`.
- Two instances needed:
  - **Utterance-end VAD** for `RECORDING` state: `min_silence_duration_ms=700` (don't cut on mid-sentence pauses).
  - **Barge-in VAD** for `SPEAKING` state: `min_silence_duration_ms=150` (want fast interruption detection).
- Reset state between conversations: `it.reset_states()`.

Fixed 512-sample chunk at 16 kHz is a hard requirement — router already guarantees this.

### STT

```python
class STT(Protocol):
    async def open_stream(self, on_transcript: Callable[[Transcript], Awaitable[None]]) -> STTSession: ...

class STTSession(Protocol):
    async def send(self, pcm_i16: bytes) -> None: ...
    async def close(self) -> str:
        """Return final transcript."""

@dataclass
class Transcript:
    text: str
    is_final: bool
    confidence: float | None
```

**Deepgram Nova-3 streaming implementation** (chosen impl):

- Package: `deepgram-sdk` (Python), current v7.6.0 as of 2026-07-22 [pypi.org/project/deepgram-sdk/]. Requires Python ≥ 3.10.
- Install: `pip install deepgram-sdk`.
- Client: `DeepgramClient()` (reads `DEEPGRAM_API_KEY` from env by default).
- **Live connection surface (SDK v7+):** `client.listen.v2.connect()` used as a context manager. This is a change from earlier v3.x/v4.x namespaces (`listen.asynclive.v("1")`, etc.); those older paths are gone.
- Event registration: `connection.on(EventType.OPEN | MESSAGE | CLOSE | ERROR, handler)` — imports from `deepgram.core.events`. Individual event classes (transcript, utterance_end) surface on the `MESSAGE` handler's payload.
- Model config: `model="nova-3"` (or `nova-3-general`, alias). English supported natively: `language="en-US"`.
- Reference code shape (adapted from the SDK README):

```python
from deepgram import DeepgramClient
from deepgram.core.events import EventType

client = DeepgramClient()

with client.listen.v2.connect(
    model="nova-3",
    language="en-US",
    encoding="linear16",
    sample_rate=16000,
    interim_results=True,
    vad_events=True,
    utterance_end_ms=1000,
    smart_format=True,
) as conn:
    conn.on(EventType.MESSAGE, on_transcript)   # inspect message.type for Transcript vs UtteranceEnd
    conn.on(EventType.ERROR,   on_error)
    conn.start_listening()
    # push audio: conn.send(pcm_bytes)  from the audio router
    # cleanup on context exit
```

- **Endpointing vs. UtteranceEnd** (both server-side, distinct mechanisms):
  - `endpointing=<ms>` — trailing silence threshold before Deepgram marks a transcript with `speech_final=true`. **Default 10ms** (very aggressive; will finalize on tiny pauses). For voice-assistant use we want ~300ms to avoid mid-sentence cuts.
  - `utterance_end_ms=<ms>` — separate configuration. When enabled together with `interim_results=true` and `vad_events=true`, produces a distinct `UtteranceEnd` event ~N ms after speech stops. Typical value 1000ms.
  - We use **both**: `endpointing=300` for tight in-sentence finalization; `utterance_end_ms=1000` for the clean end-of-turn signal.
- We ALSO run Silero VAD in parallel; whichever fires end-of-utterance first wins. Belt-and-suspenders in case Deepgram's stream stalls.
- Prepend the router ring-buffer (~500ms pre-wake audio) to the stream immediately after opening, so the wake phrase and beginning of utterance are captured.

Keepalive: send silence frames continuously while in `RECORDING` (sounddevice buffer naturally produces them). If audio stops flowing entirely (bug on our side), Deepgram may drop the socket. If dropped mid-utterance, log `stt_error` and transition to ERROR.

**Cost** (as of 2026-07-25 [deepgram.com/pricing]):
- Nova-3 streaming (monolingual): **$0.0048/min** pay-as-you-go, $0.0042/min on Growth plan.
- Smart formatting: **free** (included).
- Diarization / redaction: +$0.0020/min if enabled (we don't enable either).
- **Free tier: $200 credit for new accounts** — effectively unlimited for personal use for the first several months.

**Alternative model — Flux (worth flagging):** Deepgram's newer `flux-general-en` model is purpose-built for voice-agent applications (much lower latency, integrated turn-detection). It may be a better fit than Nova-3 for Carl once tested. Kept out of MVP because we already picked Nova-3 and switching is a one-line config change (`model="flux-general-en"`). Consider benchmarking during or after Milestone 2.

### TTS

```python
class TTS(Protocol):
    async def speak(
        self,
        text: str,
        chunk_out: Callable[[bytes], Awaitable[None]],
        cancel: asyncio.Event,
    ) -> None:
        """Synthesize text, pushing PCM chunks to chunk_out. Respects
        cancel event for mid-utterance interruption."""
```

**OpenAI TTS implementation** (chosen impl):
- Client: `AsyncOpenAI()`; endpoint `client.audio.speech.with_streaming_response.create(...)`.
- Model: `gpt-4o-mini-tts` (supports voice `instructions` param + streaming).
- Voice: `nova` (bright, natural; alt: `sage`, `coral`).
- Format: `response_format="pcm"` — raw 16-bit signed LE mono at **24 kHz**. Critical for mid-word cancellation (every byte is a valid cut point).
- Instructions: optional free-form tone directive from `config/prompts/tts_style.txt` (e.g. "Warm, conversational, unhurried.").
- Streaming pattern:

```python
async def speak(self, text, chunk_out, cancel):
    async with self.client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=self.voice,
        input=text,
        response_format="pcm",
        instructions=self.instructions,
    ) as response:
        async for chunk in response.iter_bytes(chunk_size=4096):
            if cancel.is_set():
                break   # exiting the context manager closes the HTTP stream
            await chunk_out(chunk)
```

**Two-level cancellation** required for barge-in (per TTS research brief):
1. Set `cancel` → breaks out of iterator, closes HTTP stream (stops delivery).
2. Drain playback queue AND call `output_stream.abort()` on sounddevice → flushes the ~100-200ms of already-buffered audio out of the OS audio buffer.

Without both, you'll get a "delayed stop" where Carl keeps talking for a beat after the user interrupts.

**Cost note**: closing the HTTP stream stops *delivery* but does not necessarily stop server-side generation — assume full billing for the requested text. Mitigate by chunking long replies at sentence boundaries and issuing per-sentence TTS calls (also cuts time-to-first-audio).

## Playback layer

```python
class Playback:
    def __init__(self, sample_rate=24000):
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype='int16',
            callback=self._on_output,
        )
        self._stream.start()

    async def push(self, pcm: bytes):
        await self._queue.put(pcm)

    async def flush(self):
        # Drain queue + abort OS buffer.
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except asyncio.QueueEmpty: break
        self._stream.abort()   # discards OS-level buffered audio, then restarts

    def _on_output(self, outdata, frames, time_info, status):
        # Pull from queue; if empty, output silence.
        # ...
```

The `.abort()` call is what actually cuts the sound. `.stop()` drains gracefully — we don't want that on barge-in.

## Latency budget (end-to-end)

For "hey Carl, when was Chicago founded":

| Segment | Budget | Notes |
|---|---|---|
| Wake detection | ~50ms | Porcupine one frame + inference |
| STT streaming through utterance | (parallel) | Streams during speech |
| STT final after speech_end | ~200ms | Deepgram flushes remaining |
| LLM first token | ~400ms | Sonnet 4.6 typical |
| First sentence assembled | +200-500ms | ~10-15 tokens |
| TTS first byte | ~340ms | OpenAI TTS pcm mode |
| **Total speech_end → first spoken syllable** | **~1.0-1.5s** | Achievable target |

If we blow this budget, first suspect: TTS format (mp3/opus adds decoding latency; PCM is direct) or LLM streaming setup (must use `client.messages.stream()`, not `.create()`).

## Failure modes

| Failure | Recovery |
|---|---|
| Mic device disappears mid-run | InputStream callback raises → router logs, retries open with backoff, ERROR state announces if permanent |
| Speaker device disappears | Same pattern for OutputStream |
| Porcupine `.ppn` expired (free-tier 30-day) | Startup check: try init, catch expiration error, print clear "regenerate .ppn" message and exit |
| Silero VAD first-inference warmup lag | Do a throwaway inference at startup; warm the model before first user turn |
| Deepgram WebSocket disconnect mid-utterance | STT session logs error, pipeline transitions RECORDING → ERROR → IDLE with "sorry, connection dropped" |
| OpenAI TTS request 429 | SDK retries; if sustained, speak "sorry, my voice service is throttled" (irony noted) |
| Playback queue underrun (LLM slower than TTS) | Insert brief silence, don't glitch; the output callback tolerates empty queue gracefully |
| Barge-in fires but STT can't open new stream | Log, fall back to IDLE with "sorry, try again" |
| Audio format mismatch (16-bit expected, got 24-bit) | Assertion at router setup; fail loud at startup |

## Test strategy

- **Unit (per stage)**: mock the SDK, feed golden audio fixtures, assert events.
- **Unit**: router dispatches to active consumers only; ring buffer wraps correctly.
- **Unit**: playback flush aborts stream AND drains queue.
- **Integration (with real Deepgram/OpenAI/Porcupine keys, cheap models)**: hey-Carl-say-something E2E under 3 seconds.
- **Integration**: barge-in — start TTS, inject VAD speech_start, assert playback stopped within 250ms of trigger, TTS HTTP connection closed.
- **Integration**: pre-wake ring buffer — utter "hey Carl when was Chicago founded" quickly; verify STT transcript includes "when was Chicago founded" not just "founded".
- **Chaos**: pull the mic USB during RECORDING, verify graceful ERROR → IDLE.
- **Latency benchmark**: p50 and p95 for speech_end → first-audio-byte, tracked in CI-adjacent fixture runs.

## Open questions

- [OPEN] Local STT fallback (whisper.cpp) — add as a second `STT` implementation when Deepgram is unavailable or user is offline? v2. The interface is designed for it.
- [OPEN] Local TTS fallback (Piper) — same. v2.
- [OPEN] Audio device switching mid-session (user plugs in headphones) — sounddevice doesn't handle this transparently. Detect via device-change event and re-open streams. v2.
- [OPEN] Multi-channel input (stereo mic for beamforming) — not with the hardware plan on record ("whatever mic user can borrow"). If we ever get a ReSpeaker 4-mic, revisit.
- [OPEN] Explicit VAD sensitivity per user (some users mumble). Defaults suffice for now; add per-user tuning if we ship multi-user.
- [OPEN] Deepgram-native `UtteranceEnd` event vs. Silero — pick one primary trigger and use the other as backup, or fire on-first? Current design: fire on whichever comes first, with an assertion if they disagree by >2s.
- [OPEN] Deepgram Flux vs. Nova-3 for the voice-agent path — Flux is Deepgram's newer voice-agent-optimized model with integrated turn detection and lower latency. Nova-3 is more general-purpose. Benchmark both after Milestone 2 lands.
