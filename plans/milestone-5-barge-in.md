# Milestone 5 — Barge-in (Tier 2)

## Goal

Enable Tier 2 barge-in: while Carl is speaking, if the user starts talking, TTS cuts off within ~250ms and Carl transitions to RECORDING with the user's new utterance. See `conversation-state.md` for the state machine detail. No echo cancellation (that's Tier 3, deferred).

## Prerequisites

- Milestone 4 complete (memory works; full pipeline solid).
- Specs read: `conversation-state.md` (barge-in flow section), `pipeline.md` (TTS cancellation + playback flush).

## Non-goals

- Tier 3 (always-on wake word during playback + AEC).
- Music-playback interruption.
- Speaker verification (rejecting TV audio as false trigger).

## What changes vs Milestone 4

1. A second Silero VAD instance (barge-in VAD) is instantiated with `min_silence_duration_ms=150` for fast detection.
2. During `SPEAKING`, the audio router routes frames to this barge-in VAD (in addition to any wake-word listener, which stays paused per Tier 2).
3. On `speech_start` while in `SPEAKING`, pipeline fires the two-level TTS cancellation (HTTP stream + playback flush + audio device abort).
4. Assistant text truncation for history (LLM should see what user actually heard, not full response).
5. Follow-up window ↔ barge-in interaction: barge-in can happen during the follow-up RECORDING window too — treat it just like normal speech start.

## Files to create/modify

### New: barge-in VAD instance

In `carl/stages/vad/silero.py`, allow multiple instances with different configs:

```python
def create_utterance_end_vad(config) -> SileroVAD:
    return SileroVAD(SileroConfig(
        threshold=config.threshold,
        sample_rate=16000,
        min_silence_duration_ms=config.min_silence_ms_utterance_end,  # 700
        speech_pad_ms=config.speech_pad_ms,                            # 100
    ))

def create_bargein_vad(config) -> SileroVAD:
    return SileroVAD(SileroConfig(
        threshold=config.threshold,
        sample_rate=16000,
        min_silence_duration_ms=config.min_silence_ms_bargein,          # 150
        speech_pad_ms=0,                                                # don't pad
    ))
```

Both are lightweight (~2MB model). Load both at startup, keep both warm.

### Modify: `carl/pipeline/router.py`

Register both VADs with distinct names:

```python
router.register("vad_utterance_end", utterance_end_vad.process)
router.register("vad_bargein", bargein_vad.process)
```

Activation per state:
- `RECORDING`: `activate("vad_utterance_end", "stt")`
- `SPEAKING`: `activate("vad_bargein")`

### Modify: `carl/pipeline/orchestrator.py` — the barge-in handler

```python
async def _on_speech_start_during_speaking(self, event):
    """Barge-in detected. Fire two-level cancellation and transition to RECORDING."""
    assert self.state == State.SPEAKING

    played_bytes = self.router.output_bytes_written    # tracked in Playback
    played_ms = played_bytes / (24000 * 2) * 1000

    # 1. Signal TTS to stop delivering.
    self._tts_cancel.set()

    # 2. Flush playback (drains queue + aborts OS audio buffer).
    await self.router.abort_output()

    # 3. Log the interruption with truncation info.
    self.event_log.log("tts", "tts_interrupted",
                       payload={"bytes_played": played_bytes,
                                "played_ms": int(played_ms)},
                       conversation_id=self._conv_id, turn_id=self._turn_id)

    # 4. Estimate what was actually spoken for LLM history.
    # 15 chars/sec approximation
    played_chars = int(played_ms / 1000 * 15)
    truncated_assistant_text = self._current_assistant_text[:played_chars]

    # 5. Update the turn record with actual spoken text.
    await self._update_turn(self._turn_id,
                             assistant_text=truncated_assistant_text,
                             interrupted=True)

    # 6. Add to conversation history what user actually heard.
    self._history[-1].assistant_text = truncated_assistant_text

    # 7. Start a new turn: transition to RECORDING with a fresh STT stream.
    self._turn_id = ulid.new().str
    self.state = State.RECORDING
    self._current_stt = await self.stages.stt.open_stream(self._on_stt_transcript)
    await self.router.replay_ring(self._current_stt.send)   # pre-VAD-onset audio
    self.router.activate("vad_utterance_end", "stt")
    self.event_log.log("fsm", "state_transition",
                       payload={"from": "SPEAKING", "to": "RECORDING", "trigger": "bargein"})
```

### Modify: `Playback` (in router)

Track cumulative bytes written to the OS output buffer so we can estimate played-ms on cancellation:

```python
class Playback:
    def __init__(self, ...):
        self.output_bytes_written = 0
    def _on_output(self, outdata, ...):
        # ... existing pull-from-queue logic
        self.output_bytes_written += bytes_actually_written
```

Reset `output_bytes_written = 0` when starting a new TTS response.

### Modify: `TTS.speak` — accept and honor cancel

Already accepts `cancel: asyncio.Event` per `pipeline.md`. Verify the OpenAI impl actually breaks on cancel (breaking out of the `async for` closes the streaming context manager, which closes the HTTP connection).

### Modify: reasoner — abort mid-stream

Add `abort()` method to the reasoner:

```python
class ClaudeReasoner:
    def __init__(self, ...):
        self._current_stream = None

    async def turn(self, ...):
        # ... at each stream open:
        self._current_stream = stream
        try:
            ...
        finally:
            self._current_stream = None

    async def abort(self):
        """Cancel any in-flight streaming turn."""
        if self._current_stream:
            await self._current_stream.close()
```

Pipeline calls `reasoner.abort()` if barge-in fires while THINKING (rare — usually barge-in is during SPEAKING).

### Modify: config

Barge-in already in `carl.yaml`:

```yaml
conversation:
  bargein_enabled: true
vad:
  silero:
    min_silence_ms_bargein: 150
```

Verify both are honored.

## Pseudocode: the flow

```
Carl (SPEAKING): "Chicago was founded in 1833 by a group of settlers on the site of a—"
User (interrupts): "wait, when?"
├─ Frame N: barge-in VAD sees speech energy, but too short (< 150ms)
├─ Frame N+2: still speech energy, cumulative 96ms — not yet
├─ Frame N+5: 160ms of speech energy → speech_start
├─ Pipeline._on_speech_start_during_speaking fires:
│    ├─ tts_cancel.set()                        # HTTP stream closes on next iteration
│    ├─ router.abort_output()                    # OS audio buffer flushed
│    ├─ played_bytes = 41000 (bytes written to OS)
│    ├─ played_ms = 41000 / 24000 / 2 * 1000 = ~854ms
│    ├─ played_chars ≈ 12                       # "Chicago was "
│    ├─ Log tts_interrupted event
│    ├─ Update turn: assistant_text = "Chicago was ", interrupted = True
│    ├─ Transition SPEAKING → RECORDING (new turn_id)
│    └─ Open new Deepgram stream, replay ring buffer
├─ Deepgram receives ring buffer (contains ~500ms including "wait, when?")
├─ VAD (utterance_end) sees speech, then silence → speech_end
├─ STT final: "wait when"
├─ Reasoner turn:
│    messages history now shows:
│      user: "when was Chicago founded"
│      assistant: "Chicago was"                  # truncated!
│      user: "wait when"                          # the interruption
│    → LLM understands "when" refers to founding of Chicago
│    → Response: "1833."
├─ TTS speaks "1833." (~600ms)
```

The truncated history is the load-bearing detail — without it, the LLM sees its previous full response and thinks "wait, when" refers to something else.

## Definition of done

- [ ] Two Silero VAD instances instantiated with different tunings.
- [ ] Router state map: `SPEAKING → active {vad_bargein}`; `RECORDING → active {vad_utterance_end, stt}`.
- [ ] Barge-in trigger cancels TTS HTTP stream (verify via network trace / event log's `tts_interrupted` event within 300ms of `speech_start`).
- [ ] Playback stops within 250ms of `speech_start` (measured wall-clock in acceptance test).
- [ ] `turns.assistant_text` reflects truncated content when `interrupted=True`.
- [ ] LLM history uses truncated text — verified by integration test where the second turn references only what was actually spoken.
- [ ] Follow-up-window RECORDING: barge-in during that state is handled identically (no code branching needed).
- [ ] Robustness: five rapid interrupt attempts in a row don't corrupt state (each properly resets).
- [ ] `pytest tests/test_bargein.py` passes.

## Acceptance test (manual)

1. *"hey Carl, tell me about the history of Chicago."*
2. Carl starts a long spoken response ("Chicago is a major city in the Midwest with a rich history dating back...").
3. About 2 seconds in, interrupt: *"just the founding year, please."*
4. Carl should stop within a beat and respond: *"1833."*

Verify in event log:
- `tts_interrupted` event with `played_ms` roughly matching how long you heard.
- `turns` table: first turn has `interrupted=True` and truncated `assistant_text`.
- Second turn's `user_transcript` = "just the founding year please".
- Third response cleanly answers the follow-up.

## Traps / lessons

- **VAD hearing itself**: without hardware AEC, the barge-in VAD may hear the speaker output through the mic and trigger a false interrupt. Mitigations from `pipeline.md`:
  - Physically separate mic and speaker (>30cm).
  - Bump `min_silence_ms_bargein` to 200ms if false triggers happen.
  - Tune Silero threshold up (0.6-0.7) at cost of some real-user misses.
  - If still bad, this is when the user should buy the Jabra Speak or ReSpeaker.
- **Audio buffer overflow on rapid interrupts**: if user keeps interrupting, ensure `output_bytes_written` resets between TTS turns. Otherwise played_ms accumulates and truncation gets weird.
- **Reasoner still running when barge-in fires**: if barge-in comes during THINKING (before any TTS started), `_on_speech_start` handler needs the THINKING variant — abort reasoner, transition to RECORDING. Two handlers (one for SPEAKING, one for THINKING), same event trigger.
- **`.abort()` + `.start()` glitch**: on some audio backends, `sd.OutputStream.abort()` followed immediately by `.start()` produces a click. Add a tiny silence prime (zero-fill 40ms) before pushing real TTS after abort.
- **Character-count → seconds mapping is rough**: ~15 chars/sec is average English speech. Different voices differ (nova is faster, sage is slower). Accept ~20% error; the LLM doesn't need character-perfect truncation, just directional.
