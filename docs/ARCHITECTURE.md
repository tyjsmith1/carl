# Architecture

The whole system on one page. Deep detail per subsystem lives in `specs/`; this doc is the map that connects them.

## System diagram

```
                                                    ┌───────────────────────────┐
                                                    │      config/carl.yaml     │
                                                    │  + config/prompts/*.md    │
                                                    │  + .env (secrets)         │
                                                    └────────────┬──────────────┘
                                                                 │ loads at start
                                                                 ▼
     ┌───────────────────────────────────────────────────────────────────────────────┐
     │                              Pipeline (FSM owner)                             │
     │   IDLE → RECORDING → THINKING → SPEAKING → (loop)                             │
     │   Barge-in: SPEAKING → RECORDING via VAD                                      │
     └───┬─────────────┬──────────────┬─────────────┬──────────────┬──────┬─────────┘
         │             │              │             │              │      │
         ▼             ▼              ▼             ▼              ▼      ▼
    ┌─────────┐  ┌──────────┐   ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
    │ Audio   │  │ Wake Word│   │  VAD     │  │  STT     │  │Reasoner │  │  TTS     │
    │ Router  │  │(OpenWakeWord)│(Silero x2)  │(Deepgram)│  │(Claude  │  │(OpenAI)  │
    │ sd I/O  │  │          │   │          │  │          │  │ Sonnet) │  │          │
    └────┬────┘  └──────────┘   └──────────┘  └──────────┘  └────┬────┘  └─────┬────┘
         │                                                        │             │
         │                                                        ▼             │
         │                                             ┌────────────────────┐   │
         │                                             │  Tool Registry     │   │
         │                                             │  - web_search      │   │
         │                                             │  - save_memory     │   │
         │                                             │  - correct_memory  │   │
         │                                             │  - list_memories   │   │
         │                                             │  - forget          │   │
         │                                             │  (MCP later)       │   │
         │                                             └─────────┬──────────┘   │
         │                                                       │              │
         │                                                       ▼              │
         │                                             ┌────────────────────┐   │
         │                                             │  MemoryStore       │◄──┘
         │                                             │  (SQLite)          │
         │                                             │  - retrieve()      │
         │                                             │  - save()          │
         │                                             │  - supersede()     │
         │                                             │  - tombstone()     │
         │                                             └─────────┬──────────┘
         │                                                       │
         ▼                                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                          Event Log (SQLite)                          │
    │  every wake, VAD event, transcript, LLM call, tool call, TTS chunk,  │
    │  memory op → append-only, batched writes, non-blocking               │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │ reads yesterday's events
                                       ▼
                            ┌──────────────────────┐
                            │  Nightly Consolidator │
                            │  (Claude Opus 4.7)   │
                            │  cron 0 3 * * *       │
                            │  → new memories      │
                            │  → supersede stale   │
                            │  → tombstone deleted │
                            └──────────────────────┘
```

## The pipeline in one paragraph

`sounddevice` runs one continuous full-duplex audio stream. The `AudioRouter` dispatches PCM frames to different consumers based on the FSM state: in `IDLE`, frames go to Porcupine + a rolling ring buffer. When Porcupine detects "hey Carl," the FSM transitions to `RECORDING`; the router opens a Deepgram stream, replays the ~500ms ring buffer into it (so the wake phrase and utterance start aren't cut off), and starts feeding live frames to both Deepgram and Silero VAD. When Silero (or Deepgram's own endpointing) fires `speech_end`, the FSM transitions to `THINKING`, closes the STT stream, and hands the final transcript to the Reasoner. The Reasoner loads recent conversation history and retrieved memories, calls Claude with the tool registry, and streams tokens back. Sentence-boundary text goes to the TTS stage, which streams PCM chunks back through the router to the speaker. If tools are requested, the pipeline executes them (in parallel where safe), plays a rotating filler phrase ("let me check"), feeds tool results back to Claude, and continues. During `SPEAKING`, a *second* Silero VAD instance watches the mic for barge-in — if the user starts talking, TTS is cancelled at both the HTTP-stream level and the OS audio-buffer level, and the FSM transitions straight back to `RECORDING` with the user's new utterance. Everything at every stage is logged to the SQLite event log for later consolidation.

## Subsystem map

| Subsystem | Spec | Role |
|---|---|---|
| **Config** | `specs/config.md` | YAML + env vars. Chooses implementations per stage. Hot-reload for prompts only. |
| **Pipeline / FSM** | `specs/conversation-state.md` | Owns state (IDLE/RECORDING/THINKING/SPEAKING/ERROR), transition table, timeouts, barge-in coordination. |
| **Audio + Wake + VAD + STT + TTS** | `specs/pipeline.md` | The audio path. Router, provider interfaces (`WakeWord`, `VAD`, `STT`, `TTS`), and concrete implementations. |
| **Reasoner** | `specs/reasoner.md` | Claude integration. Streaming turn loop with tool-use, prompt caching, sentence-boundary chunking, filler-speech coordination. |
| **Tools** | `specs/tools.md` | `ToolProvider` + `ToolRegistry`. Native tools + safety tiers + confirmation flow. MCP-ready seam. |
| **Memory** | `specs/memory.md` | SQLite-backed store. Typed memories (fact/preference/correction/routine). Retrieval strategies (dump-all now, embed-topk later). Correction via supersession. |
| **Event Log** | `specs/event-log.md` | SQLite append-only record of everything. Non-blocking writes. Consumed by consolidation and debug. |
| **Nightly Consolidation** | `specs/nightly-consolidation.md` | Opus 4.7 walks yesterday's events, proposes memory changes, dedupes, applies. Runs on cron. Dry-run capable. |

## Data flow: one turn from wake to reply

```
1. Porcupine on frame N: keyword_index >= 0 → wake_detected event on FSM queue.
2. FSM: IDLE → RECORDING. Open Deepgram stream. Replay ring buffer. Activate {vad_utterance_end, stt} consumers.
3. Router streams frames → both VAD and Deepgram.
4. VAD (silence > 700ms) → speech_end event.
5. FSM: RECORDING → THINKING. Close STT. Retrieve top memories.
6. Reasoner turn:
     a. Build system prompt: personality + memories block + rules.
     b. client.messages.stream(model=sonnet-4.6, messages, tools).
     c. Stream text_deltas → extract complete sentences → yield SpokenChunk.
        │ If content_block_start(tool_use): fire filler phrase.
        │ On end_turn: exit loop.
        │ On stop_reason=tool_use: execute tools in parallel, feed back, iterate.
7. First SpokenChunk arrives → FSM: THINKING → SPEAKING. TTS starts on that chunk.
8. TTS streams PCM chunks → router pushes to speaker.
9. Router simultaneously feeds mic to vad_bargein. If speech_start fires:
     a. Set tts_cancel Event → closes HTTP stream to OpenAI.
     b. Drain playback queue + abort OutputStream → flush OS audio buffer.
     c. FSM: SPEAKING → RECORDING (new turn_id).
     d. Truncate assistant_text for LLM history to what was actually played.
10. tts_complete → FSM: SPEAKING → RECORDING (follow-up window, 3s).
11. Silence for 3s → FSM: RECORDING → IDLE.
```

## Latency budget

```
User stops speaking
  │
  ├─ VAD detects silence-end          ~700ms  (tuned; balance for natural pauses)
  ├─ STT closes, returns final        ~200ms  (Deepgram flush)
  ├─ Reasoner first token             ~400ms  (Sonnet 4.6 typical)
  ├─ First complete sentence          +200-500ms (10-15 tokens)
  ├─ TTS first byte                   ~340ms  (OpenAI PCM streaming)
  ├─ First audio to speaker           ~30ms   (sounddevice + OS buffer prime)
  └─ TOTAL to first spoken syllable:  ~1.0-1.5s
```

Cache warm hits (after turn 2 in a conversation) shave ~150ms off the Reasoner first-token.

## Extensibility seams

Design commitments that let v2/v3 features slot in without rewrites:

| Seam | What it enables |
|---|---|
| `ToolProvider` interface | Add MCP support as a new provider class. No refactor. |
| Event log capturing everything from day 1 | Enables the nightly consolidation, future behavioral learning, and observability tools. |
| `MemoryStore` retrieval strategies | Swap `dump_all` → `embed_topk` when memory count grows, no pipeline code changes. |
| `WakeWord`/`STT`/`TTS`/`VAD` protocol per stage | Add local implementations (whisper.cpp, Piper) as alternate providers when we want offline mode. |
| `user_id` column in memory + event tables | Multi-user via voice-ID becomes a runtime addition, not a schema migration. |
| Scheduler seam in pipeline (planned) | System-initiated turns (timers firing, reminders) — pipeline can accept events from cron/timer sources, not just wake-word. |
| Prompt hot-reload | Personality tuning without restart. |

## What lives in one process vs. crossing processes

**One Python process (`carl` daemon):**
- Everything on the audio path: router, wake, VAD, STT, reasoner, TTS.
- Tool registry (native tools in-process; MCP tools via subprocess/HTTP later).
- Memory store (SQLite in same file).
- Event log writer.

**Separate process (`carl consolidate` cron job):**
- Nightly consolidation. Reads the same SQLite; writes memory changes; runs during Carl's idle time.

**Rationale**: keep the interactive path in one process for latency and simplicity. Batch work (consolidation) doesn't need to share memory with the daemon; SQLite handles the coordination.

## Concurrency model

- **Asyncio single event loop** for everything on the audio path.
- **`sounddevice` audio callbacks** run on a separate thread (native audio callback); bridged to asyncio via `janus.Queue`.
- **SQLite writes** use `aiosqlite` (which pools a thread executor). WAL mode allows concurrent reads.
- **Event log writes** are batched by a single writer task pulling from an asyncio.Queue. Fire-and-forget from stages; never block on the log.

## Failure philosophy

- **Fail loud and early at startup.** Bad config, missing keys, unreachable providers → exit 1 with a clear message. Never limp through startup.
- **Fail soft at runtime.** Any single-turn failure → transition to ERROR state, announce briefly ("sorry, something went wrong"), return to IDLE. Carl never becomes unresponsive.
- **Every state has a timeout.** Nothing can hang forever.
- **The event log is the flight recorder.** If Carl misbehaves, `SELECT * FROM events WHERE conversation_id = ...` shows what happened.

See failure-modes tables in each spec for the specifics.

## Where to look for what

- "Why did we pick X?" → `docs/DECISIONS.md`
- "What are we deferring / what needs doing before prod?" → `docs/BACKLOG.md`
- "How do I set up my dev env?" → `plans/milestone-1-scaffold.md`
- "How does barge-in actually work?" → `docs/specs/conversation-state.md` (barge-in section) + `docs/specs/pipeline.md` (TTS cancellation + playback flush)
- "How does Carl decide what to remember?" → `docs/specs/nightly-consolidation.md`
- "How do tools get registered?" → `docs/specs/tools.md` (registry section)
- "What's in the event log?" → `docs/specs/event-log.md` (event-type catalog)
- "Where do I put an API key?" → `docs/specs/config.md` (Secrets section)
