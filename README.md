# Carl

A personal voice assistant you own end-to-end.

Carl listens for its wake word, understands what you say, decides what to do (with tools when it helps), and replies in a natural spoken voice. Unlike off-the-shelf assistants, Carl builds up *durable memory* of who you are and *learns from your corrections over time* — it should feel like it's learning *you*, not just accumulating trivia.

## What Carl is

- **Runs on your laptop** (Raspberry Pi later). No cloud dependency for the appliance itself; cloud APIs for the smart bits.
- **Voice-first, streaming end-to-end.** Speech in and out. First spoken syllable within ~1.5s of you finishing a sentence.
- **Interruptible.** Talk over Carl and it stops within a beat, listens to your new question.
- **Actually remembers.** Facts, preferences, corrections, and routines persist across conversations. Every night, Carl reviews the day and consolidates what to keep.
- **Extensible via tools.** Web search on day one. Timers, weather, smart home, MCP integrations as follow-ons.

## What Carl is not (yet)

- A general-purpose smart-home hub. Smart home is a follow-on via MCP.
- A phone or SMS interface. Voice only.
- Multi-user. Schema is ready but voice-ID isn't implemented.
- Local-model-only. Cloud APIs power the smart parts. Local models are a planned alternate implementation, not a v1 requirement.

## Doc layout

Start with `docs/ARCHITECTURE.md` for the big picture. Deep-dive per subsystem is in `docs/specs/`. Sequenced build instructions are in `plans/`. Load-bearing architectural choices are logged in `docs/DECISIONS.md`.

```
README.md                          ← you are here
docs/
  ARCHITECTURE.md                  ← whole system at a glance
  DECISIONS.md                     ← ADR log of key choices
  BACKLOG.md                       ← deferred fixes, milestone callouts, pre-prod checklist
  specs/                           ← deep spec per subsystem
    pipeline.md
    reasoner.md
    memory.md
    tools.md
    conversation-state.md
    event-log.md
    nightly-consolidation.md
    config.md
plans/                             ← sequenced build milestones
  milestone-1-scaffold.md
  milestone-2-hello-carl.md
  milestone-3-web-search.md
  milestone-4-memory.md
  milestone-5-barge-in.md
  milestone-6-consolidation.md
carl/                              ← Python source (created in milestone 1)
config/                            ← YAML config + prompt files (created in milestone 1)
tests/                             ← pytest suite (created in milestone 1)
```

## Reading order

**If you're the person building this:**
1. `docs/ARCHITECTURE.md` — get the shape.
2. `docs/DECISIONS.md` — understand why choices are what they are.
3. `docs/specs/config.md` and `docs/specs/pipeline.md` — the two most cross-referenced specs.
4. `plans/milestone-1-scaffold.md` — start building.

**If you're a fresh Claude session picking up this codebase:**
1. Read `docs/ARCHITECTURE.md`.
2. Read the spec for the subsystem you're touching.
3. Read the milestone plan for what you're building (or fixing).
4. Then read the code.

**If you're the user (Tyler):**
- `docs/ARCHITECTURE.md` gives you the mental model.
- Skip milestone plans unless you want to see how sausage gets made.

## Stack (at a glance)

| Layer | Choice |
|---|---|
| Runtime | Python 3.11+ / asyncio |
| Wake word | OpenWakeWord (`hey_jarvis` for MVP; custom `hey_carl` on backlog) |
| Voice activity | Silero VAD |
| Speech-to-text | Deepgram Nova-3 (streaming) |
| Reasoning | Claude Sonnet 4.6 |
| Nightly consolidation | Claude Opus 4.7 |
| Text-to-speech | OpenAI TTS (`gpt-4o-mini-tts`, streaming PCM) |
| Web search | Tavily |
| Storage | SQLite (WAL mode, one file) |
| Audio I/O | sounddevice (full-duplex) |

Full rationale in `docs/DECISIONS.md`.

## Costs (rough, single-user)

- **Interactive**: ~$0.005-0.02 per conversation turn (Claude Sonnet + Deepgram + OpenAI TTS + optional Tavily).
- **Nightly consolidation**: ~$0.30/day (~$110/year) for Opus 4.7 batch runs.
- **Wake word + VAD**: free (on-device).
- **Ballpark total**: $10-30/month depending on usage.

## Status

Design phase complete. Implementation not started. See `plans/milestone-1-scaffold.md` to begin.

## Running Carl

*Not yet — this section fills in as milestones ship.*

Milestone 1 (target: after scaffold):
```
uv sync
carl
```

Milestone 2 (target: after real integrations):
```
# .env with API keys populated
carl
# → say "hey Carl, when was Chicago founded"
```
