# Decision Log

Architectural decisions that shape the whole system. Format is loosely ADR-style: each entry states the decision, the alternatives considered, why we picked what we picked, and any known consequences to watch. Newest at bottom.

Decisions that were locked in during the initial design brainstorm (2026-07-25). New decisions get appended over time.

---

## ADR-001 — Cloud-first APIs for MVP, modular for local

**Decision**: Every stage uses cloud APIs (Deepgram/Claude/OpenAI/Tavily) for the MVP. Every stage sits behind an interface so local implementations (whisper.cpp, Piper, local LLMs) can swap in via config later.

**Alternatives considered**:
- Fully local from day 1 — small models on a Pi give bad answers to trivia questions; the "fully local Alexa" is a common MVP graveyard.
- Cloud-only with no seam for local — locks us in and violates the modular commitment.

**Why**: Fastest path to a working demo with good UX. Cost is trivial for single-user personal use. Seams preserve future optionality.

**Consequences**: Requires internet for the assistant to function. API key management is a real thing. Costs scale with usage (mitigated by prompt caching + the fact that voice UX bounds turns per day).

---

## ADR-002 — Python + asyncio

**Decision**: Python 3.11+ with asyncio. Not TypeScript.

**Alternatives**: TypeScript on Node/Deno.

**Why**: Voice/audio ecosystem is Python-first. `sounddevice`, `silero-vad`, `pvporcupine`, and `piper-tts` all have native Python integrations; TS/Node has to shell out or use less-maintained ONNX ports for the same. User's stated preference was TS but agreed Python was smoother once tradeoffs were spelled out.

**Consequences**: If the user wants to develop in TS specifically, we'd need to reconsider. Nothing else materially changes.

---

## ADR-003 — Streaming end-to-end, not batch request/response

**Decision**: Every stage that can stream, streams. STT emits partials. LLM streams tokens. TTS streams PCM chunks. Sentence-boundary detection kicks TTS off before LLM finishes. Playback runs on a queue that can be flushed mid-stream.

**Alternatives**: Naive batch (record utterance → transcribe → send to LLM → get full response → TTS → play). Simpler but adds 4-6s of dead air per turn.

**Why**: Voice UX target of ~1-1.5s time-to-first-syllable is unachievable without streaming.

**Consequences**: More concurrency to manage. Cancellation and abort semantics have to be right (see ADR-005 on barge-in). Retrofitting streaming later would be a rewrite.

---

## ADR-004 — LLM-native tool orchestration, no router layer

**Decision**: Claude's native tool-use API drives orchestration. No pre-LLM classifier, no LangChain-style planner. Tools are registered with schema + description; the LLM picks when to call them.

**Alternatives**: Rule-based router that classifies intent before invoking LLM (2023-era pattern). LangChain agents.

**Why**: Modern tool-calling models are strictly better at tool selection than any router we'd build. Building a router above the LLM is reinventing worse tool-calling.

**Consequences**: Tool `description` fields become the prompt-engineering surface. Iteration on descriptions is how we fix over/under-triggering. LLM cost per turn is slightly higher (tools always in context) but prompt caching mitigates.

---

## ADR-005 — Tier 2 barge-in from day 1, Tier 3 deferred

**Decision**: v1 supports Tier 2 barge-in — while Carl is speaking, VAD monitors the mic and cuts TTS on any human speech energy. Wake-word runs only in IDLE (paused during SPEAKING). Tier 3 (always-on wake word during playback + acoustic echo cancellation) is deferred.

**Alternatives**:
- Tier 1 (no barge-in) — feels clunky, Alexa-2015.
- Tier 3 in v1 — AEC is a real project; hardware AEC in the mic simplifies but user's starting mic has none.

**Why**: Tier 2 gets ~80% of the feel for ~20% of the work. Tier 3 upgrades cleanly later (the audio pipeline is already full-duplex).

**Consequences**: TTS output MUST be interruptible mid-stream (uses OpenAI's `with_streaming_response` + `response_format=pcm`). Without hardware AEC, VAD may false-trigger from speaker bleed if mic is too close to speaker; mitigation is physical separation and threshold tuning. When the user upgrades hardware, upgrade to Tier 3.

---

## ADR-006 — Memory and behavioral learning designed in from day 1

**Decision**: Even though the memory system is fully implemented over Milestones 4 + 6, three architectural seams exist from Milestone 1:
1. Event log captures everything.
2. Retrieval hook is in the LLM turn (returns `[]` initially).
3. `save_memory` is registered as a tool (no-op at first).

**Alternatives**: Add memory later as a bolt-on when the interactive path is stable.

**Why**: The user explicitly wants memory + learning as core to the product, not an add-on. Bolt-on memory has predictable pain points (event log doesn't have what consolidation needs; retrieval hooks are shoehorned into a place they don't fit). Design-in avoids all of this at ~10 extra lines of code in v1.

**Consequences**: The event log spec ships in Milestone 1 with real schema, even though nothing reads it yet. Same for memory tools. Small overhead, big future payoff.

---

## ADR-007 — SQLite for everything (memories, events, conversations)

**Decision**: One SQLite file (`~/.carl/carl.db`) for memories, event log, conversations, turns. WAL mode. Migrations via numbered `.sql` files.

**Alternatives**:
- Postgres — overkill for single-user local.
- JSONL for events + separate SQLite for memories — makes join queries painful for consolidation.
- DuckDB — nice analytical queries but heavier and less battle-tested for our write pattern.

**Why**: Zero-ops. One file to back up. Queryable. `sqlite3` CLI for debugging. Migrates cleanly. Enough for tens of millions of events on a single machine.

**Consequences**: Single writer to the DB (WAL mode + a batched writer task); reads are concurrent. Full-text search on transcripts would need `sqlite-fts` extension if we want it later.

---

## ADR-008 — Typed memory (fact/preference/correction/routine) with supersession, not overwrite

**Decision**: Memories are typed. Corrections don't overwrite prior memories; they create a new memory that points to the old one via `superseded_by`. Deletion is tombstoning, not DELETE.

**Alternatives**:
- Flat blob memory (a single "notes about the user" list).
- In-place overwrite on correction.

**Why**: Types drive retrieval weighting (preferences and corrections always injected; facts and routines only when relevant) and consolidation heuristics. Supersession preserves audit trail — "what did Carl think about my dog before I corrected it, and when did I correct it?" is a well-formed query. Tombstoning enables undo.

**Consequences**: Slightly more schema complexity. Retrieval queries filter `WHERE superseded_by IS NULL AND tombstoned = FALSE`. Old memory rows accumulate; occasional GC job may be needed at scale (not v1).

---

## ADR-009 — Nightly consolidation runs Opus 4.7, not Sonnet

**Decision**: The nightly job uses `claude-opus-4-7`. The interactive reasoner uses `claude-sonnet-4-6`.

**Alternatives**: One model everywhere.

**Why**: Consolidation is latency-insensitive but reasoning-quality-sensitive. Opus is the right tool. Sonnet is the right tool for interactive turns where first-token latency matters. Cost tradeoff favors specialization: Opus at 3am on 10K tokens is ~$0.30/day; Opus on every interactive turn would be many multiples of that and slower.

**Consequences**: Two models to manage. Config already has both blocks. Consolidation cost budget: ~$110/year. Watch it if prompt bloats.

---

## ADR-010 — Silent memory recall (don't narrate what you remembered)

**Decision**: Carl does not say "I remember Ruby is your dog" or similar. It just uses the memory naturally. The system prompt has an explicit rule forbidding narration.

**Alternatives**: Narrate every retrieved memory for transparency ("I remember...").

**Why**: The user picked silent when the tradeoff was presented. Silent feels magical; narrated feels performative and slightly creepy over time. Explicit user tools (`list_memories`, "hey Carl, what do you know about me?") give transparency on demand.

**Consequences**: The system prompt rule needs periodic tuning — LLMs tend to want to narrate. May need a post-processor if the rule fails.

---

## ADR-011 — Tavily as first tool (web_search), not Brave/Serper/Google

**Decision**: Tavily is the search provider. Interface allows swapping.

**Alternatives**:
- Brave Search API — cheaper but requires more result-massaging for LLM use.
- Serper (Google) — cheap, good, but general-purpose.
- Google Programmable Search — free tier is stingy (100/day).

**Why**: Tavily is designed specifically for LLM tool-use. Returns pre-synthesized answers alongside cited results. Free tier is generous. Response format is compact and tool-result-friendly.

**Consequences**: One more vendor dependency. Free tier rate limits are low (~single-digit RPS); mitigated by per-turn serialization.

---

## ADR-012 — Docs-first, then scaffold, then vertical slice

**Decision**: Write the full doc set (specs + milestone plans + top-level docs) before writing any code. Then Milestone 1 (scaffold + mock impls). Then Milestone 2 (vertical slice).

**Alternatives**: Skeleton-first (write code stubs, iterate).

**Why**: User explicitly stated "little faith you'll build the skeleton properly on one go" — the design has enough surface area that stubs without a spec would produce gaps we can't see in advance. Docs first is the senior-engineer move; forces us to answer hard questions before they become 400-line PRs to revisit.

**Consequences**: Larger up-front time investment before running code. Milestones can be handed off to fresh sessions (or new engineers) with just the plan + relevant specs — that's the payoff.

---

## ADR-013 — MCP as a v2 seam, not a v1 dependency

**Decision**: The tool system has a `ToolProvider` interface with `NativeToolProvider` as the v1 implementation. `MCPToolProvider` is defined in the spec but not built until v2.

**Alternatives**: Ship MCP integration in v1.

**Why**: MCP is the right *destination* for tool extensibility (especially for smart-home integration, which the user is interested in but needs to inventory hardware first). Building it in v1 adds complexity before we have a concrete need. The interface seam is free; the implementation waits.

**Consequences**: When the user has smart-home hardware to plug in, adding MCP is a new provider class, not a refactor.

---

## ADR-014 — Follow-up window: keep mic open for 3s after Carl speaks

**Decision**: After TTS completes, FSM stays in RECORDING (not IDLE) for `follow_up_ms` (default 3000ms). If user speaks in that window, no wake word needed.

**Alternatives**: Require wake word for every turn (Alexa behavior).

**Why**: Enables natural back-and-forth ("hey Carl, when was Chicago founded" → "1833" → *without re-waking* "who founded it?"). Small feature, large UX win, differentiates Carl from Alexa.

**Consequences**: Barge-in and follow-up-window are the same state (RECORDING) — simplifies the FSM. Failure mode: if user is talking to someone else in the room right after Carl finishes, Carl might try to interpret it. Mitigation: 3s is short enough that this rarely triggers on random chatter; VAD threshold tunable.

---

## ADR-015 — Iterative personality via hot-reloaded prompt files

**Decision**: The base system prompt lives in `config/prompts/carl_base.md` and is watched by `watchfiles`. Edits take effect on the next turn without restart.

**Alternatives**: Prompt hardcoded in Python (restart to change).

**Why**: User explicitly wants to iterate on personality easily. Prompt engineering is the highest-leverage tuning knob; making it 0-friction to change matters. Also enables `config/prompts/carl_sassy.md` alt-personality experiments.

**Consequences**: File-watcher adds a dependency (`watchfiles`) and a small startup task. Prompt errors mid-run keep the last-known-good until next successful reload. Only prompts hot-reload; other config still requires restart.
