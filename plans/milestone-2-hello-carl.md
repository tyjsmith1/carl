# Milestone 2 — Hello Carl

## Goal

Replace every mock stage from Milestone 1 with the real integration. After this milestone, you say **"hey Carl, when was Chicago founded"** into your mic and Carl responds with a spoken answer in ~1-2 seconds. No tools, no memory yet — just the vertical slice working end-to-end.

## Prerequisites

- Milestone 1 complete (scaffold runs, all interfaces wired, mock pipeline passes).
- API keys in `.env`:
  - `ANTHROPIC_API_KEY`
  - `DEEPGRAM_API_KEY`
  - `OPENAI_API_KEY`
- Working USB mic and speaker connected. Verify with `python -m sounddevice` (should list both).
- **Note**: original plan used Picovoice Porcupine as the wake-word engine. Swapped to OpenWakeWord on 2026-07-26 because Picovoice's signup now requires a corporate email. OpenWakeWord ships a pre-trained `hey_jarvis` model; custom "hey_carl" training via Colab notebook is a backlog item.

## Non-goals

- Tools (Milestone 3).
- Memory retrieval or save (Milestone 4).
- Barge-in / TTS interruption (Milestone 5).
- Nightly consolidation (Milestone 6).

## What changes vs Milestone 1

Config flips from mocks to real impls:

```yaml
providers:
  wake_word: openwakeword  # was: mock
  vad:       silero        # was: mock
  stt:       deepgram      # was: mock
  reasoner:  claude_sonnet # was: mock
  tts:       openai        # was: mock
  search:    mock          # not used yet
```

## Files to create/modify

### New: real stage implementations

Each is a class implementing the Protocol from `carl/stages/{stage}/base.py`. All ~50-150 lines each.

#### `carl/stages/wake_word/openwakeword.py`

```python
import numpy as np
import openwakeword
from openwakeword.model import Model
from carl.stages.wake_word.base import WakeWord

class OpenWakeWord(WakeWord):
    def __init__(self, config):
        # First run downloads pretrained model files (~few MB).
        openwakeword.utils.download_models()
        self._model_key = config.model_name
        model_arg = config.model_path or config.model_name
        self._model = Model(
            wakeword_models=[model_arg],
            inference_framework=config.inference_framework,
        )
        self._threshold = config.threshold
        self.sample_rate = 16000
        self.frame_length = 1280   # 80 ms at 16 kHz — see spec for router chunk math
        self._buffer = bytearray()
        self._on_wake_cb = None

    async def start(self, on_wake):
        self._on_wake_cb = on_wake

    def wants_audio(self) -> bool:
        return True

    def feed(self, pcm_i16: bytes) -> None:
        # Router hands us its native 512-sample frames; buffer to 1280-sample
        # (80 ms) predictions.
        self._buffer.extend(pcm_i16)
        while len(self._buffer) >= self.frame_length * 2:
            chunk = bytes(self._buffer[: self.frame_length * 2])
            del self._buffer[: self.frame_length * 2]
            arr = np.frombuffer(chunk, dtype=np.int16)
            scores = self._model.predict(arr)
            if scores.get(self._model_key, 0.0) > self._threshold:
                # Bridge to async: schedule the coroutine on the pipeline loop
                # (main.py holds a ref to the loop; use loop.call_soon_threadsafe).
                if self._on_wake_cb is not None:
                    self._loop.call_soon_threadsafe(
                        asyncio.create_task,
                        self._on_wake_cb({"keyword": self._model_key,
                                           "score": scores[self._model_key]}),
                    )

    async def stop(self):
        return   # nothing to clean up
```

**Threading**: called from the audio router's callback thread. OpenWakeWord is thread-safe per instance; a single instance handled per router callback thread is fine. The wake event bridges back to the asyncio loop via `loop.call_soon_threadsafe`.

**Model files**: `openwakeword.utils.download_models()` on first run pulls the pretrained models (a few MB) to a cache dir. Subsequent runs use the cache. Custom-trained `.tflite` / `.onnx` models can be loaded by passing `model_path` instead of `model_name`.

#### `carl/stages/vad/silero.py`

Two instances: one for utterance-end (used during RECORDING), one for barge-in (used during SPEAKING, but Milestone 5 wires that). Milestone 2 only needs the utterance-end one.

```python
from silero_vad import load_silero_vad, VADIterator
import numpy as np

class SileroVAD(VAD):
    def __init__(self, config):
        self._model = load_silero_vad(onnx=False)
        self._iter = VADIterator(
            self._model,
            threshold=config.threshold,
            sampling_rate=config.sample_rate,
            min_silence_duration_ms=config.min_silence_ms_utterance_end,
            speech_pad_ms=config.speech_pad_ms,
        )
        self.sample_rate = config.sample_rate
        self.frame_length = 512

    async def process(self, pcm_i16: bytes) -> VADEvent | None:
        pcm = np.frombuffer(pcm_i16, dtype=np.int16).astype(np.float32) / 32768.0
        evt = self._iter(pcm, return_seconds=True)
        if evt is None: return None
        if "start" in evt: return SpeechStart(timestamp=evt["start"])
        if "end"   in evt: return SpeechEnd(timestamp=evt["end"])

    def reset(self):
        self._iter.reset_states()
```

Do a warmup inference at startup:

```python
def warmup(vad):
    silence = b"\x00" * (vad.frame_length * 2)
    asyncio.run(vad.process(silence))
```

#### `carl/stages/stt/deepgram.py`

Uses the current `deepgram-sdk` v7+ API surface: `client.listen.v2.connect()` as a context manager, `EventType.*` handlers from `deepgram.core.events`. This differs from older v3/v4 examples on the web — do not use `listen.asynclive.v("1")`; that path is gone.

```python
from deepgram import DeepgramClient
from deepgram.core.events import EventType

class DeepgramSTT(STT):
    def __init__(self, config):
        self._client = DeepgramClient()   # reads DEEPGRAM_API_KEY from env
        self._config = config

    async def open_stream(self, on_transcript):
        session = DeepgramSession(self._client, self._config, on_transcript)
        await session._open()
        return session

class DeepgramSession(STTSession):
    def __init__(self, client, config, on_transcript):
        self._client = client
        self._config = config
        self._on_transcript = on_transcript
        self._conn = None
        self._final_text = ""

    async def _open(self):
        # Enter the SDK's context manager manually so we can hold the
        # connection across our own async boundary. `__enter__` returns
        # the connection object.
        self._cm = self._client.listen.v2.connect(
            model=self._config.model,               # "nova-3"
            language=self._config.language,         # "en-US"
            encoding="linear16",
            sample_rate=16000,
            interim_results=self._config.interim_results,   # True
            vad_events=True,
            utterance_end_ms=1000,
            smart_format=self._config.smart_format,
            endpointing=self._config.endpointing_ms,        # 300
        )
        self._conn = self._cm.__enter__()

        def _on_message(message):
            # message.type distinguishes Transcript vs UtteranceEnd vs Metadata.
            # (Exact attribute names live in the deepgram.core.messages module —
            # print message.__dict__ once during dev to lock these in.)
            if message.type == "Transcript":
                alt = message.channel.alternatives[0]
                is_final = getattr(message, "is_final", False)
                if is_final:
                    self._final_text = (self._final_text + " " + alt.transcript).strip()
                asyncio.create_task(self._on_transcript(
                    Transcript(text=alt.transcript, is_final=is_final,
                                confidence=getattr(alt, "confidence", None))
                ))
            elif message.type == "UtteranceEnd":
                # Fire our SpeechEnd event through a callback registered on the pipeline
                self._on_utterance_end()

        self._conn.on(EventType.MESSAGE, _on_message)
        self._conn.on(EventType.ERROR,   lambda err: log.error("deepgram_error", error=str(err)))
        self._conn.on(EventType.CLOSE,   lambda _: log.info("deepgram_closed"))
        self._conn.start_listening()

    async def send(self, pcm_i16: bytes) -> None:
        self._conn.send(pcm_i16)

    async def close(self) -> str:
        # Exit the context manager to close cleanly.
        self._cm.__exit__(None, None, None)
        return self._final_text
```

**Important gotchas verified against the current SDK (v7.6.0, released 2026-07-22):**
- `client.listen.v2.connect()` is a **context manager**, not a coroutine. It returns the connection object on `__enter__`. We hold the CM open across async boundaries because our audio stream is async.
- Events are strings on `message.type`: `"Transcript"`, `"UtteranceEnd"`, `"Metadata"`, `"SpeechStarted"`. Print one during dev to lock the exact field names for your version.
- `endpointing` default is **10ms** (very aggressive). We use 300ms.
- `utterance_end_ms` is a *separate* mechanism; requires `interim_results=True` AND `vad_events=True` to fire.
- Config `stt.deepgram.model` should be `nova-3` (or `nova-3-general` alias). Nova-3 supports en-US natively.

**Alternative to consider after Milestone 2**: Deepgram's newer `flux-general-en` model is purpose-built for voice agents (lower latency, integrated turn-detection). Same SDK surface — flip config `stt.deepgram.model: flux-general-en` and benchmark.

#### `carl/stages/reasoner/claude.py`

Per `reasoner.md` — the streaming turn loop with tool-use, but for this milestone there are no tools yet, so it's simpler:

```python
from anthropic import AsyncAnthropic

class ClaudeReasoner(Reasoner):
    def __init__(self, config, memory_store=None, tool_registry=None):
        self.client = AsyncAnthropic()
        self.config = config
        # memory_store and tool_registry are None at milestone 2

    async def turn(self, request, on_tool_call=None, on_filler_needed=None):
        system = self._build_system_prompt([])   # no memories yet
        messages = [{"role": "user", "content": request.user_transcript}]

        async with self.client.messages.stream(
            model=self.config.model,
            system=system,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ) as stream:
            buffer = ""
            async for text in stream.text_stream:
                buffer += text
                for sentence in extract_complete_sentences(buffer):
                    yield SpokenChunk(text=sentence, is_final=False)
                    buffer = buffer[len(sentence):]
            if buffer:
                yield SpokenChunk(text=buffer, is_final=True)
```

The base personality prompt (`config/prompts/carl_base.md`) becomes real content:

```
You are Carl, a personal voice assistant. Direct, warm, dry. You value the user's time.

Your voice is spoken, not written: no bullet lists, no headings, no markdown.
One idea per sentence. Complete sentences. Keep answers short unless asked for detail.
```

#### `carl/stages/tts/openai.py`

Per pipeline.md, using OpenAI's streaming TTS at 24 kHz PCM:

```python
from openai import AsyncOpenAI

class OpenAITTS(TTS):
    def __init__(self, config):
        self.client = AsyncOpenAI()
        self.config = config

    async def speak(self, text, chunk_out, cancel):
        async with self.client.audio.speech.with_streaming_response.create(
            model=self.config.model,          # "gpt-4o-mini-tts"
            voice=self.config.voice,          # "nova"
            input=text,
            response_format="pcm",
            instructions=self._instructions(),
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=4096):
                if cancel.is_set(): break
                await chunk_out(chunk)
```

For this milestone, `cancel` is never set (barge-in is milestone 5) but the plumbing is here.

### New: audio router (real)

`carl/pipeline/router.py` — real sounddevice InputStream + OutputStream.

**Critical**: sounddevice callbacks run on the audio thread, NOT the asyncio loop. Use `janus.Queue` or `loop.call_soon_threadsafe` to bridge. Recommendation: `janus.Queue[bytes]` — the callback does a sync put, an async task drains and dispatches.

```python
import sounddevice as sd
import janus, asyncio, numpy as np

class AudioRouter:
    FRAME = 512
    IN_SR = 16000
    OUT_SR = 24000

    def __init__(self, event_loop):
        self._loop = event_loop
        self._in_queue: janus.Queue[bytes] = janus.Queue()
        self._out_queue: janus.Queue[bytes] = janus.Queue()
        self._ring = collections.deque(maxlen=16)  # ~500ms at 32ms/frame
        self._consumers: dict[str, Callable] = {}
        self._active_set: set[str] = set()

    def start(self):
        self._in_stream = sd.RawInputStream(
            samplerate=self.IN_SR, blocksize=self.FRAME,
            channels=1, dtype='int16',
            callback=self._in_cb,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=self.OUT_SR, blocksize=1024,
            channels=1, dtype='int16',
            callback=self._out_cb,
        )
        self._in_stream.start()
        self._out_stream.start()
        asyncio.create_task(self._dispatch_loop())

    def _in_cb(self, indata, frames, time_info, status):
        # audio thread
        self._in_queue.sync_q.put_nowait(bytes(indata))

    def _out_cb(self, outdata, frames, time_info, status):
        # audio thread; pull from queue, pad with silence if underrun
        try:
            data = self._out_queue.sync_q.get_nowait()
            n = min(len(data), len(outdata))
            outdata[:n] = data[:n]
            if n < len(outdata):
                outdata[n:] = b'\x00' * (len(outdata) - n)
        except janus.SyncQueueEmpty:
            outdata[:] = b'\x00' * len(outdata)

    async def _dispatch_loop(self):
        while True:
            frame = await self._in_queue.async_q.get()
            self._ring.append(frame)
            for name in list(self._active_set):
                await self._consumers[name](frame)

    async def push_output(self, pcm: bytes):
        await self._out_queue.async_q.put(pcm)

    async def abort_output(self):
        # drain queue + flush OS buffer
        while not self._out_queue.async_q.empty():
            try: self._out_queue.async_q.get_nowait()
            except asyncio.QueueEmpty: break
        self._out_stream.abort()
        self._out_stream.start()   # restart for next TTS
```

### Modify: pipeline orchestrator

The FSM transition handlers now do real work:

- `_on_wake_detected`: open Deepgram stream, replay ring buffer into it, activate ["vad", "stt"] consumers.
- `_on_speech_end`: close STT stream, get final transcript, kick off reasoner.
- Reasoner streams `SpokenChunk`s; each is fed to `tts.speak()` → PCM bytes → `router.push_output()`.
- `_on_tts_complete`: transition SPEAKING → RECORDING (follow-up window).

Follow-up window is a `asyncio.create_task(sleep_then_check)` pattern; if `speech_start` fires within the window, stay in RECORDING; otherwise → IDLE.

### Modify: `main.py`

Just call `router.start()` before `pipeline.run()`. Everything else already wired from Milestone 1.

## Pseudocode: the fine-grained flow

```
User: (says "hey Carl")
├─ Porcupine detects on frame N. Emits wake_detected event.
├─ FSM: IDLE → RECORDING
│   ├─ Open Deepgram stream (async)
│   ├─ router.replay_ring(deepgram_session.send)  # ~500ms pre-buffer
│   └─ router.activate("vad", "stt")               # now live frames go to both
User: (says "when was Chicago founded")
├─ Frames flow into VAD (finds speech_start, then continues) and STT
├─ STT emits interim_results, final chunks
User: (stops speaking, 700ms of silence)
├─ VAD emits speech_end
├─ FSM: RECORDING → THINKING
│   ├─ router.deactivate("vad", "stt")
│   ├─ final_transcript = await stt_session.close()
│   └─ Kick off reasoner.turn(request)
├─ Reasoner streams tokens → sentence extractor yields "Chicago was founded in 1833."
├─ FSM: THINKING → SPEAKING (on first SpokenChunk)
│   └─ For each chunk: tts.speak(chunk.text, chunk_out=router.push_output, cancel=cancel_event)
├─ TTS pushes PCM into router; router pushes into OS audio buffer
├─ User hears "Chicago was founded in 1833. By a group of settlers..."
├─ Reasoner completes; tts_complete event
├─ FSM: SPEAKING → RECORDING (follow-up window, 3s)
├─ (nothing said)
└─ FSM: RECORDING → IDLE
```

Total: ~1.5s from stopping speech to Carl starting to speak. Meets the "feels responsive" bar per pipeline.md latency budget.

## Definition of done

- [ ] All 5 real stage impls exist, unit tests pass with SDK mocks.
- [ ] Router's threading contract is documented at the top of `router.py` (audio thread vs asyncio loop).
- [ ] Full pipeline runs. Saying **"hey Jarvis, when was Chicago founded"** (pretrained MVP phrase; custom "hey Carl" is a backlog item) into a mic produces a spoken answer.
- [ ] Event log for one conversation contains: `wake_detected`, `stt_final`, `llm_request`, `llm_response`, `tts_first_byte`, `tts_complete`, and the correct state_transitions.
- [ ] `sqlite3 ~/.carl/carl.db "SELECT latency_ms_to_first_audio FROM turns"` shows values under 2000ms for most turns on a decent internet connection.
- [ ] Recovery: unplug the mic mid-conversation; Carl transitions to ERROR, announces via TTS, returns to IDLE when mic reconnects.
- [ ] `pytest -m integration` (with API keys set) runs a scripted E2E test successfully.

## Acceptance test (manual)

Sit at your laptop. Terminal shows:

```
$ carl
carl 0.1.0 — providers: porcupine, silero, deepgram, claude_sonnet, openai
Warming up VAD... ok.
Loading wake-word model (hey_carl_mac.ppn)... ok.
Listening for "hey jarvis".
```

You say: *"hey Carl, when was Chicago founded"*.

Terminal logs:

```
[FSM] IDLE → RECORDING (conv=01H..., turn=01H...)
[STT] final: "when was chicago founded"
[FSM] RECORDING → THINKING
[REASONER] stream open (model=claude-sonnet-4-6)
[FSM] THINKING → SPEAKING
[TTS] first byte at 340ms
```

You hear (voice: nova): *"Chicago was founded in 1833 by a group of settlers on the site of a Potawatomi village at the mouth of the Chicago River."*

Terminal:

```
[TTS] complete (dur 4200ms, 84 chars)
[FSM] SPEAKING → RECORDING (follow-up 3s)
[FSM] RECORDING → IDLE
Listening for "hey jarvis".
```

## Traps / lessons

- **Deepgram SDK version churn**: the SDK jumped through v3 → v4 → v7 in about a year and the v2 `client.listen.v2.connect()` context-manager surface above is *current* as of v7.6.0 (2026-07-22). Older tutorials and StackOverflow answers show `client.listen.asynclive.v("1")` / `LiveOptions` / `LiveTranscriptionEvents` — that pattern is dead. Pin `deepgram-sdk == 7.6.*` in `pyproject.toml` so tutorials don't drift you back.
- **Sounddevice sample-rate mismatch**: if your default audio device doesn't support 16 kHz input, sounddevice silently resamples (badly). Explicitly set `device=` in config to a device that supports it, or resample in the router. Test by playing back what the router captures.
- **OpenWakeWord first-run model download**: first import triggers `download_models()` which pulls ~few MB from GitHub. Do this once at install time (e.g. in `pip install`'s post-install or a `carl setup` command) so first user turn isn't delayed.
- **OpenWakeWord frame size**: 80 ms (1280 samples at 16 kHz), 2.5× Porcupine's 32 ms. The audio router buffers frames — verify the buffering math when landing the real router.
- **Wake-model license**: pretrained OpenWakeWord models are CC BY-NC-SA 4.0 — non-commercial only. Fine for personal use; if Carl ever gets commercial application, train custom models (Apache-2.0 base is fine, only the pretrained weights are restricted).
- **First-token latency on Claude cold cache**: first turn of a session is ~200ms slower than subsequent (no cache hit). Warm the cache at startup with a throwaway call? Optional; measure first.
- **sounddevice + `abort()`**: after `.abort()`, the OutputStream is in a stopped state. Call `.start()` again before the next `push_output`. If you skip this and TTS goes silent forever, this is why.
- **STT partials arriving after final**: race condition where a partial event fires just after final. Filter: only accept partials until `is_final=True`, then close.
