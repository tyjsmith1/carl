# Backlog

Central log of deferred work. Three sections:
1. **Deferred fixes** — known bugs / cleanup we chose not to fix now.
2. **Milestone callouts** — things a specific future milestone must address, cross-linked from `plans/`.
3. **Pre-production checklist** — things that don't matter for solo local dev on the current machine but must exist before Carl is shipped, shared, or expected to survive a laptop reinstall.

**Discipline**: when a "later" item surfaces during work, add it here in the same session. Don't rely on memory or on comments in code — those rot silently. Entries can be short; a one-line pointer to context beats nothing.

Format: `- [severity] short description — why deferred / when to address. (source)`
Severity: `low` (nice-to-have), `med` (real bug, low blast radius), `high` (real bug, could bite users).

---

## Deferred fixes (from code review, Milestone-1.5)

- [med] **Migration script non-atomic** — `apply_migrations` uses `sqlite3.executescript` which implicit-commits per statement. If migration 002 partially fails, tables are created but `PRAGMA user_version` isn't bumped → re-run hits `table already exists` and user is stuck. **When**: before migration 002 lands. Fix: wrap each migration in `BEGIN EXCLUSIVE; ... COMMIT;`. (`carl/storage/db.py:80`, code-review finding)
- [med] **`deep_merge` accepts `null` yaml values** — if `carl.local.yaml` has `providers: ~`, deep_merge replaces base's `providers` with `None`, then Pydantic validation fails cryptically. **When**: whenever we see a user hit it. Fix: skip merge when override is `None`, or fail loud with a clear message. (`carl/config/schema.py:219`)
- [med] **`AudioRouter.activate()` replaces the active set** — asymmetric with `deactivate()` which is incremental. Milestone-1 stub doesn't exercise this; Milestone-2 might trip on it. **When**: Milestone 2, when real audio dispatch happens. Fix: decide on semantics (probably `activate` should also be incremental, and add `set_active` for full replacement).  (`carl/pipeline/router.py:50`)
- [med] **`NativeToolProvider` caches tool instances at `__init__`** — tools registered *after* the provider is built (via a late `discover()` in a test or plugin flow) show up in `list_tools()` but return "Unknown tool" from `execute()`. **When**: if we ever add hot-reload for tools, or dynamic tool registration. Fix: instantiate lazily on first `execute()`. (`carl/tools/native_provider.py:82`)
- [med] **Reasoner → TTS is sequential per chunk** — `await tts.speak(chunk.text, ...)` blocks the next chunk. Real streaming TTS is meant to be fed chunks concurrently. Mock's instant `speak` hides this. **When**: Milestone 2, when real OpenAI TTS lands. Fix: pipe reasoner chunks into a `chunk_queue`, have one long-lived `tts.speak_stream(chunk_queue)` per turn. (`carl/pipeline/orchestrator.py:218`)
- [low] **5 remaining mypy warnings** — async-generator return-type edge cases + `inspect.signature(cls.__init__)` on class object. Not real bugs, all type-narrowing complaints. **When**: any time — 15 minutes of typing fixes. Fix: add `# type: ignore[attr-defined]` where warranted or narrow the types. (`mypy carl/`)
- [low] **`_sanitize` runs per-event on the sync `log()` path** — regex sub on every string in every payload, even benign ones. Fine for MVP; move to the batched `_flush` thread when we see log-heavy workloads. (`carl/events/log.py:45`)
- [low] **Manual signal handling in `main.py` largely duplicates `asyncio.run`'s KeyboardInterrupt behavior** — could simplify by using `asyncio.Runner` or letting SIGINT propagate. Deferred because the current code works. (`carl/main.py:105`)

---

## Milestone callouts

Cross-linked from the specific milestone plan. When starting a milestone, read its section here first.

### Milestone 2 (Hello Carl — real integrations)

- **[VERIFY] tags in spec** — Porcupine, Silero, OpenAI TTS, and Tavily sections in `docs/specs/pipeline.md` have `[VERIFY]` markers on version numbers and free-tier policies. Confirm each against live docs (Deepgram already re-verified during Milestone-1.5). (`docs/specs/pipeline.md`)
- **Refactor `_on_wake_detected` to fully event-queue-driven** — Milestone-1 does the whole flow inline. Real VAD needs event-driven transitions. Docstring already flags this. (`carl/pipeline/orchestrator.py:1`)
- **Refactor STT interface** — `open_stream()` + `close() -> str` doesn't map to real streaming STT where finals arrive via callback while audio still flows. Consider: `close()` returns `None`; finals go through `on_transcript(is_final=True)` only. (`docs/specs/pipeline.md`, code-review I2)
- **Remove `send(b"")` mock hack from orchestrator** — pipeline currently sends one empty frame to trigger the mock STT partial. Delete once mock is replaced. (`carl/pipeline/orchestrator.py:165`)
- **Benchmark Deepgram Nova-3 vs Flux** — Flux is Deepgram's newer voice-agent-optimized model. Same SDK; one-line config change to test. Decide after basic pipeline works. (`docs/specs/pipeline.md` open questions)
- **Pin `deepgram-sdk == 7.6.*`** — SDK has had three major rewrites; older tutorials show dead API paths. Pin narrowly to avoid drift. (`plans/milestone-2-hello-carl.md` traps)
- **Real `AudioRouter` implementation** — Milestone-1 has a stub; real sounddevice + janus queue bridge arrives here. Reshape `activate`/`deactivate` semantics per deferred fix above.
- **`.ppn` file expiration** — Picovoice free-tier custom wake-word files expire after ~30 days. Add a startup check that surfaces expiration early with instructions to regenerate. (`docs/specs/pipeline.md`)

### Milestone 3 (Web search + tool orchestration)

- **Tool description iteration** — the `description` field is the LLM's prompt for tool selection. Expect a week of usage where we watch for over/under-triggering and refine wording. Log every tool call decision so we can spot patterns. (`docs/specs/tools.md`)
- **Filler-phrase cadence tuning** — 250ms delay-before-filler suggested in milestone-3; iterate based on real feel.
- **Tool-result truncation** — cap at ~2KB before feeding back to Claude. Currently unenforced. (`docs/specs/tools.md` open questions)

### Milestone 4 (Memory)

- **Silent-recall LLM discipline** — Claude tends to want to narrate ("I remember..."). Iterate the system-prompt rule; consider a post-processor that flags "I remember" patterns. (`docs/specs/reasoner.md`)
- **Short-ID collision handling** — 6-char ULID suffix collides at ~1 in a million; currently `find_by_short_id` returns None on multi-match. Consider raising `AmbiguousShortIDError` if this ever bites. (`carl/memory/sqlite_store.py:203`)
- **Wire `memory.retrieval_strategy` to real embed_topk implementation** — Milestone-1.5 wired config-passing; the actual `embed_topk` code path still needs building.

### Milestone 5 (Barge-in)

- **`_tts_cancel` producer** — Milestone-1 threads the Event through but never sets it. Barge-in's VAD `speech_start` handler is what sets it. (`carl/pipeline/orchestrator.py:196`)
- **Tier 3 barge-in** — Tier 2 is committed for v1; Tier 3 (always-on wake word during playback + hardware/software AEC) is the eventual upgrade when user gets a mic with hardware AEC. (`docs/DECISIONS.md` ADR-005)

### Milestone 6 (Nightly consolidation)

- **Consolidation prompt iteration** — expect 2 weeks of dry-run mode to tune it. Watch for over-triggering (saves trivia) and under-triggering (saves nothing). (`docs/specs/nightly-consolidation.md`)
- **Timezone handling** — "yesterday" is local calendar day; DST transitions can cause off-by-one. Use `zoneinfo` with explicit user tz. (`plans/milestone-6-consolidation.md` traps)

### Post-Milestone-6 / v2+

- **MCP tool provider** — `ToolProvider` interface seam exists; `MCPToolProvider` implementation deferred until we want to wire in Home Assistant or similar. (`docs/specs/tools.md`)
- **Multi-user via voice ID** — schema has `user_id` column throughout; voice-classification stage between wake-word and STT is v3-ish.
- **Behavioral learning beyond consolidation** — pattern-based routine promotion, preference decay, cross-conversation summarization.
- **Local model fallbacks** — whisper.cpp (STT), Piper (TTS), local LLM. Interfaces support it; impls deferred.
- **Follow-up window auto-extension** — if user starts speaking near end of the 3s window, grace-extend by ~500ms. Cheap UX win. (`docs/specs/conversation-state.md`)

---

## Pre-production checklist

These are things that don't matter while Carl runs only on Tyler's laptop but become critical the moment it's shared, deployed, or expected to survive a laptop rebuild.

### Version control + collaboration

- [ ] `git init` + first commit. Currently not a git repo — no history, no backup, no undo. Every code change is one `rm -rf` away from oblivion. **Do this before Milestone 2.**
- [ ] Push to a remote (private GitHub repo). Backup + history off-machine.
- [ ] `LICENSE` file. Personal project — MIT or "All rights reserved" both fine, but pick one.

### CI + quality gates

- [ ] GitHub Actions running `pytest` + `ruff check` + `mypy` on every push. Currently only run manually. Even solo, this catches regressions during rapid iteration.
- [ ] Wheel-build smoke test in CI — `python -m build && pip install dist/*.whl && carl --version` — catches the class of bugs that hit us with the missing package-data (`docs/DECISIONS.md` — we already tripped this once).

### Secrets + credentials

- [ ] API-key rotation plan. Currently: keys in `.env`, no rotation, no expiry monitoring. At minimum, calendar reminder every 90 days for Anthropic + OpenAI + Deepgram.
- [ ] Never commit `.env` (already gitignored, but re-verify after `git init`).
- [ ] Picovoice `.ppn` file expiration monitoring — free-tier custom wake words expire after ~30 days; add a startup check + a re-gen cron, or upgrade to paid tier. (see Milestone 2 callout)

### Data + backups

- [ ] `~/.carl/carl.db` backup strategy. Right now: one laptop-drive failure and Carl forgets everything. Options: Time Machine (macOS built-in, easy), or a `carl backup` subcommand that dumps to S3/iCloud/etc.
- [ ] `carl memory export` / `import` for moving Carl to a new machine. Already noted in `docs/specs/memory.md` as v2.
- [ ] Retention job for the `events` table (currently just documented as "30 days" in the spec; not implemented). Prevents unbounded DB growth.

### Runtime / observability

- [ ] Structured error reporting. `structlog` writes to stderr; no aggregation. Options: Sentry, or just tail-and-alert on `ERROR` lines from the log.
- [ ] Cost tracking. Consolidation logs `cost_usd` per run; the interactive path doesn't yet track token usage per turn. Add before real usage starts to avoid surprise bills.
- [ ] Latency SLO tracking. `turns.latency_ms_to_first_audio` is captured but nothing surfaces the p50/p95 over time. Grafana would be overkill; a `carl stats` subcommand that queries the DB is enough.
- [ ] Health-check on startup — verifies DB write, LLM API auth, mic + speaker present. Currently we fail loud on each in turn; a single preflight would be cleaner.

### Distribution / update

- [ ] Auto-update mechanism (if this becomes a long-running personal service). For personal use, `git pull && pip install -e .` is fine — document it.
- [ ] Systemd service file (Linux) or `launchd` plist (macOS) — turns Carl into a real always-on service instead of a foreground process in a terminal.
- [ ] `carl uninstall` — clean removal. Especially: what happens to `~/.carl/carl.db`?
- [ ] Package data verified in every release (`SOURCES.txt` includes `.sql`). We already hit this bug once; CI check above prevents recurrence.

### Security posture

- [ ] Filesystem permissions on `~/.carl/carl.db` (chmod 600 — it contains every transcript). Currently default umask.
- [ ] If wire up smart-home tools: sensitive-tier confirmation flow is designed but not exercised. Real integration testing needed before `unlock_door` becomes a real tool. (`docs/specs/tools.md` safety tiers)
- [ ] PII sweep in `_sanitize` is regex-based — catches API keys but not addresses, emails, medical info. Acceptable for single-user local; revisit if this ever leaves the machine.

### Multi-user (if it ever happens)

- [ ] Voice-ID stage between wake-word and STT to populate `user_id` non-trivially. Schema is ready; runtime code isn't.
- [ ] Per-user encryption at rest. Currently: filesystem permissions only.
- [ ] Consolidation cross-user leak protection — every WHERE clause needs a `user_id` filter, and the consolidation job must never pull user-A transcripts into user-B memory. Add tests before shipping this.
