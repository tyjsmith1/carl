# Conversation State Spec

## Purpose

The finite-state machine that governs Carl's overall behavior — when to listen for a wake word, when to record user speech, when to think, when to speak, and how to move between those states cleanly (including on interruption). Every audio and processing decision hangs off this FSM.

## States

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
                    │    (wake word detected)                     │
                    ▼                                             │
       ┌──────────────────┐                                       │
       │      IDLE        │◄──────────────────────────────────────┤
       │  (wake listener  │   (silence timeout /                  │
       │    active)       │    error recovery /                   │
       └──────────────────┘    turn complete + no follow-up)      │
                │                                                 │
       wake_detected                                              │
                │                                                 │
                ▼                                                 │
       ┌──────────────────┐                                       │
       │    RECORDING     │                                       │
       │  (VAD monitoring │───► utterance timeout ───► IDLE       │
       │  streams to STT) │                                       │
       └──────────────────┘                                       │
                │                                                 │
      speech_end (VAD)                                            │
                │                                                 │
                ▼                                                 │
       ┌──────────────────┐                                       │
       │    THINKING      │───► llm timeout ───► SPEAKING("...")  │
       │ (LLM + tool loop)│                                       │
       └──────────────────┘                                       │
                │                                                 │
      final text ready                                            │
                │                                                 │
                ▼                                                 │
       ┌──────────────────┐                                       │
       │    SPEAKING      │                                       │
       │ (TTS streaming   │──── done ─────────────────────────────┤
       │  to speaker;     │                                       │
       │  VAD monitors    │──── VAD triggered barge-in ──►        │
       │  mic for user)   │                RECORDING              │
       └──────────────────┘                                       │
                                                                  │
                        (follow-up window: after SPEAKING,        │
                         stay in RECORDING for `follow_up_ms`     │
                         without requiring wake word;             │
                         then decay to IDLE)                      │
                                                                  │
       ┌──────────────────┐                                       │
       │      ERROR       │───────────────────────────────────────┘
       │ (recoverable)    │
       └──────────────────┘
```

**Five states.** Each has one owner (the pipeline coroutine) and one legal set of transitions. The FSM is the *only* thing allowed to transition — stages request transitions by putting events on the FSM's queue.

### State definitions

| State | What's running | Wake-word active? | Mic recording? | TTS active? |
|---|---|---|---|---|
| `IDLE` | Wake-word engine on rolling audio buffer | Yes | Yes (to buffer, not STT) | No |
| `RECORDING` | Silero VAD + STT streaming to Deepgram | No | Yes (to STT) | No |
| `THINKING` | LLM turn + tool loop | No | No | No |
| `SPEAKING` | TTS streaming + audio playback + VAD on mic | No (Tier 2) | Yes (to VAD only, for barge-in) | Yes |
| `ERROR` | Announcing failure ("sorry, I didn't catch that") | No | No | Briefly, if a failure message is being spoken |

**Tier 2 barge-in specifically:** during `SPEAKING`, the wake word is *paused* (that's Tier 2 vs. Tier 3). The mic is open, but only VAD watches it. A `speech_start` event from VAD while in `SPEAKING` = barge-in.

## Transition table

| From | Event | To | Side effects |
|---|---|---|---|
| IDLE | `wake_detected` | RECORDING | Prepend ~500ms of pre-wake audio to STT stream (captures the wake word phrase and start of utterance) |
| IDLE | (never terminates) | — | Wake-word listener runs forever unless system shutdown |
| RECORDING | `speech_end` (VAD) | THINKING | Close STT stream, get final transcript, start LLM turn |
| RECORDING | `utterance_timeout` (max 30s) | THINKING | Same, but log a warning |
| RECORDING | `stt_error` | ERROR → IDLE | Announce "sorry, something went wrong" |
| RECORDING | `no_speech_timeout` (5s of silence with no `speech_start`) | IDLE | User woke Carl by accident |
| THINKING | `llm_final_text` | SPEAKING | Start TTS stream |
| THINKING | `llm_timeout` (20s) | SPEAKING | Speak a fallback ("that's taking too long, try again") |
| THINKING | `llm_error` | ERROR → IDLE | Announce error |
| THINKING | `tool_error_nonrecoverable` | SPEAKING | Speak the error to the user ("I couldn't reach the web") |
| SPEAKING | `tts_complete` | RECORDING (follow-up) | Keep mic open + VAD active for `follow_up_ms` (default 3000ms) so user can respond without saying "hey Carl" |
| SPEAKING | `speech_start` (VAD, barge-in) | RECORDING | Flush TTS stream, flush playback buffer, resume from user's new utterance |
| SPEAKING | `tts_error` | ERROR → IDLE | |
| RECORDING (follow-up) | `no_speech_timeout` (`follow_up_ms`) | IDLE | Conversation naturally ends |
| RECORDING (follow-up) | `speech_start` | RECORDING (normal) | Just continue |
| ERROR | (finished announcing) | IDLE | Ready for next wake word |

**Follow-up window** is the "you can keep talking without waking me again" behavior. Alexa doesn't do this well. It's Carl's differentiator: after Carl speaks, the mic stays open for 3 seconds. Enables natural back-and-forth ("hey Carl, when was Chicago founded" → "1833" → *within 3 sec* "who was the founder?" → answered without re-waking).

## Barge-in flow (the load-bearing detail)

When VAD emits `speech_start` while in `SPEAKING`:

```
1. FSM transitions SPEAKING → RECORDING immediately (single atomic switch).
2. TTS cancellation signal fires:
     a. asyncio.Event set to break out of the TTS response iterator.
     b. Playback queue (asyncio.Queue of PCM chunks) is drained.
     c. Audio output device is aborted (sounddevice: stream.abort()) to
        flush the OS-level audio buffer (~100-200ms of already-queued audio).
     d. The partial `tts_interrupted` event is logged with bytes_played.
3. STT stream opens fresh with the user's new utterance.
4. The previous turn's assistant_text is truncated to what was actually
   played (bytes_played / bytes_per_ms → chars via TTS timing estimate).
   This is what gets recorded in the `turns` table as `assistant_text`.
   Matters because the LLM's history must reflect what the user actually
   heard, not what the LLM produced.
```

The **truncation-of-actually-heard-text** is subtle and important. If Carl was in the middle of "Chicago was founded in 1833 by a group of settlers who—" when the user interrupted, and the user's next turn is "wait, when?", the LLM needs to see "Chicago was founded in 1833 by a group of settlers who" (not the full response) as the prior assistant turn, or its context is wrong.

Estimate played text via: `played_bytes / (sample_rate * bytes_per_sample) → played_seconds → played_chars` using an approximate 15 chars/sec speaking rate. Not exact, but close enough for the LLM to reason about.

## Timeouts (all in config)

| Timeout | Default | Rationale |
|---|---|---|
| `utterance_max_seconds` | 30 | Cap on a single user utterance. Longer = user is monologuing or forgot the mic was open. |
| `no_speech_timeout` (post-wake) | 5000ms | Wake word triggered but user never spoke (false accept / distraction) |
| `follow_up_ms` | 3000ms | Post-TTS window without needing wake word |
| `llm_timeout_seconds` | 20 | LLM turn (including tool loop) must complete in this many seconds |
| `tool_timeout_seconds` | 15 | Individual tool call cap |
| `tts_timeout_seconds` | 10 | TTS synthesis must produce first byte within this window |
| `wake_reset_seconds` | 300 | If IDLE for 5 minutes with no interaction, force-reset the wake-word engine (defensive) |

## Pipeline coroutine skeleton

```python
class Pipeline:
    """The FSM lives here. Single owner of state transitions."""

    def __init__(self, config, stages):
        self.state: State = State.IDLE
        self.stages = stages
        self.event_queue: asyncio.Queue[PipelineEvent] = asyncio.Queue()
        self.current_conversation_id: str | None = None
        self.current_turn_id: str | None = None
        self._tts_cancel: asyncio.Event = asyncio.Event()

    async def run(self):
        # Kick off the wake-word listener; it puts wake_detected on queue.
        asyncio.create_task(self.stages.wake_word.listen(self.event_queue))
        while True:
            event = await self.event_queue.get()
            await self._handle(event)

    async def _handle(self, event: PipelineEvent):
        # Big transition table dispatch. Each handler:
        #   1. Validates the transition is legal from current state.
        #   2. Fires side effects (start/stop coroutines, flush buffers).
        #   3. Updates self.state.
        #   4. Logs a state_transition event.
        handler = self._TRANSITIONS.get((self.state, event.type))
        if handler is None:
            log.warning("illegal transition", state=self.state, event=event.type)
            return
        await handler(self, event)
```

The FSM must never be modified except through this queue. Any stage that wants to trigger a transition (e.g. VAD detects speech end) puts an event on the queue; it does NOT call `self.state = X` directly. This is the invariant that keeps concurrency safe.

## Interaction with the event log

Every state transition emits an event to the event log:

```json
{
  "event_type": "state_transition",
  "payload": {"from": "SPEAKING", "to": "RECORDING", "trigger": "bargein"}
}
```

Post-hoc debugging: `SELECT * FROM events WHERE event_type = 'state_transition' AND conversation_id = ?` gives a full trace.

## Failure modes

| Failure | Recovery |
|---|---|
| Illegal transition requested | Log warning, ignore. FSM invariant: only legal transitions succeed silently. |
| Stage coroutine crashes mid-state | Uncaught exception → ERROR state → announce → return to IDLE. Never leave FSM stuck. |
| Audio device disconnected | Detected in sounddevice callback → ERROR state → attempt reconnect (5 tries with backoff) → announce if permanent. |
| Wake-word engine crashes | Restart engine, stay in IDLE. Log `provider_error`. |
| LLM streaming stalls (bytes stop arriving) | `llm_timeout` fires → transition to SPEAKING with fallback text. |
| TTS starts but produces no audio | `tts_timeout` (10s to first byte) → ERROR → announce. |
| Barge-in fires but STT can't open a new stream | Log error, fall back to IDLE with "sorry, try again." |

**Golden rule: no state should be able to hang forever.** Every state has a max-duration timeout that forces a transition somewhere safe.

## Test strategy

- **Unit**: transition table is exhaustive — every (state, event) pair either has a handler or is explicitly rejected.
- **Unit**: illegal transitions are logged but don't mutate state.
- **Unit**: barge-in flush drains playback queue AND aborts audio device (both, not just one).
- **Property**: any sequence of random events keeps FSM in a valid state.
- **Property**: for any conversation, `SUM(state_durations) == wall_clock_time` (no time unaccounted for).
- **Integration**: simulated conversation runs through IDLE → RECORDING → THINKING → SPEAKING → IDLE with correct event log entries.
- **Integration**: barge-in test — during SPEAKING, inject VAD `speech_start`, verify: TTS stopped within 300ms of trigger, audio device flushed, new STT stream open, `assistant_text` truncated.
- **Integration**: chaos — kill each stage mid-conversation, verify recovery.

## Open questions

- [OPEN] Wake-word during SPEAKING (Tier 3 barge-in): what triggers the upgrade path? Probably "user asked for it in memory: prefers to interrupt during music playback." v2.
- [OPEN] Should `follow_up_ms` auto-extend if the user starts speaking near the end of the window? Yes probably — grace period of another ~500ms. Add when it feels wrong.
- [OPEN] Multi-turn "hold" (user says "hold on" — Carl mutes but doesn't return to IDLE). Cute but requires intent detection. v2+.
- [OPEN] Explicit "cancel" phrase during THINKING ("never mind"). Also needs to work during a tool call. Design touches barge-in FSM; defer.
