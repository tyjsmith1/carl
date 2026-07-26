# Milestone 6 — Nightly Consolidation

## Goal

Implement the nightly "sleep" job. Every night at 3am, Claude Opus walks yesterday's event log, proposes new memories to save, memories to supersede, and memories to tombstone. Runs on a cron trigger, single-instance, dry-run capable. See `nightly-consolidation.md` for the spec.

## Prerequisites

- Milestone 5 complete (full pipeline stable in real use for at least a few days so there's real data to consolidate).
- `nightly-consolidation.md` spec read.
- `config/prompts/consolidation.md` prompt template ready (created in this milestone).

## Non-goals

- Weekly review job.
- User-facing "here's what I plan to remember" review UI.
- Behavioral learning beyond memory extraction (e.g., pattern-based routine promotion).

## What changes

Adds a new subcommand `carl consolidate` and a cron-triggered background daemon. Both invoke the same `Consolidator` class.

## Files to create/modify

### New: `carl/consolidation/consolidator.py`

The core class. Runs the full flow described in `nightly-consolidation.md`.

```python
from datetime import date, datetime, timedelta
import fcntl, ulid, json
from anthropic import AsyncAnthropic

class Consolidator:
    def __init__(self, config, memory_store, event_log, db_path):
        self.config = config
        self.memory_store = memory_store
        self.event_log = event_log
        self.db_path = db_path
        self.client = AsyncAnthropic()
        self.lock_path = os.path.expanduser("~/.carl/consolidation.lock")

    async def run_for_date(self, target_date: date, *, dry_run: bool = False):
        # 1. Acquire lock or exit.
        lock_fd = self._acquire_lock()
        if lock_fd is None:
            log.warn("consolidation already running, exiting")
            return

        try:
            # 2. Query yesterday's data.
            turns = await self._load_turns(target_date)
            if not turns:
                self._log_run(target_date, {"turns_scanned": 0})
                return

            tool_calls = await self._load_tool_calls(target_date)
            current_memories = await self.memory_store.list()

            # 3. Build prompt.
            prompt = self._build_prompt(turns, tool_calls, current_memories)

            # 4. Run Opus with tool-use for proposals.
            proposals = await self._get_proposals(prompt)

            # 5. Validate + apply (or dump to file if dry_run).
            summary = await self._apply(proposals, current_memories, dry_run=dry_run,
                                          target_date=target_date)

            # 6. Log run event.
            self._log_run(target_date, summary)
        finally:
            self._release_lock(lock_fd)

    async def _get_proposals(self, prompt: str) -> list[Proposal]:
        proposals = []
        messages = [{"role": "user", "content": prompt}]
        for iteration in range(10):
            response = await self.client.messages.create(
                model=self.config.consolidation.model,        # claude-opus-4-7
                max_tokens=self.config.consolidation.max_tokens,
                temperature=0.3,
                tools=CONSOLIDATION_TOOLS,                     # per nightly-consolidation.md
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            new_props = []
            finalize_called = False
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "finalize":
                        finalize_called = True
                        continue
                    new_props.append(Proposal(kind=block.name, input=block.input))

            proposals.extend(new_props)

            # Provide neutral tool_results so LLM continues.
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": "acknowledged"}
                    for b in response.content if b.type == "tool_use"
                ],
            })

            if finalize_called or response.stop_reason == "end_turn":
                break

        return proposals
```

### New: `carl/consolidation/validators.py`

Per `nightly-consolidation.md`, apply checks before write:

```python
class ProposalValidator:
    MAX_CONTENT_CHARS = 500
    MAX_PROPOSALS_PER_RUN = 20
    PII_PATTERNS = [
        re.compile(r"\d{3}-\d{2}-\d{4}"),          # SSN
        re.compile(r"\b\d{13,19}\b"),               # card number
    ]

    def validate(self, proposals: list[Proposal], current_memories: list[Memory]) -> ValidationResult:
        rejected = []
        accepted = []
        for p in proposals:
            reason = self._check(p, current_memories)
            if reason: rejected.append((p, reason))
            else: accepted.append(p)
        # Cap at 20; keep highest confidence ones
        if len(accepted) > self.MAX_PROPOSALS_PER_RUN:
            accepted = sorted(accepted, key=lambda p: -p.confidence)[:self.MAX_PROPOSALS_PER_RUN]
        return ValidationResult(accepted=accepted, rejected=rejected)
```

### New: `config/prompts/consolidation.md`

Full prompt template per `nightly-consolidation.md`. Uses `{{placeholders}}` for substitution.

### New: `carl/consolidation/scheduler.py`

Cron-triggered runner. Two options:
1. **External cron** (simpler): `crontab` calls `carl consolidate --for-date=yesterday`. Recommend this for MVP.
2. **In-process scheduler** (self-contained): `apscheduler` running inside a Carl daemon. Nicer UX but Carl main loop must stay up.

Ship option 1 first. Option 2 later.

`carl consolidate` subcommand:

```python
@click.command()
@click.option("--for-date", default="yesterday", help="Date to consolidate (YYYY-MM-DD or 'yesterday')")
@click.option("--dry-run/--no-dry-run", default=None,
              help="Override config.consolidation.dry_run")
def consolidate(for_date, dry_run):
    config = Config.load()
    target = _parse_date(for_date)
    dry_run = dry_run if dry_run is not None else config.consolidation.dry_run
    memory_store = SQLiteMemoryStore(config.storage.sqlite_path)
    event_log = EventLog(config.storage.sqlite_path)
    consolidator = Consolidator(config, memory_store, event_log, config.storage.sqlite_path)
    asyncio.run(consolidator.run_for_date(target, dry_run=dry_run))
```

Suggested crontab:

```
0 3 * * * cd /path/to/assistant && /path/to/venv/bin/carl consolidate --for-date=yesterday >> ~/.carl/consolidation.log 2>&1
```

### Modify: `carl/pipeline/orchestrator.py`

No change needed for MVP if using external cron. If in-process:

```python
# In main.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler

def main():
    # ... existing setup
    if config.consolidation.enabled and config.consolidation.in_process_scheduler:
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            lambda: consolidator.run_for_date(date.today() - timedelta(days=1)),
            CronTrigger.from_crontab(config.consolidation.cron),
        )
        scheduler.start()
    # ... run pipeline
```

## Pseudocode: the flow

```
3:00am cron fires.
├─ carl consolidate --for-date=yesterday
├─ Acquire ~/.carl/consolidation.lock (fcntl.LOCK_EX). If busy: exit.
├─ Query: turns for yesterday (34 turns across 8 conversations).
├─ Query: tool calls for yesterday.
├─ Query: current active memories (47 total).
├─ Build prompt from consolidation.md template with substitutions:
│    {{current_memories_block}}      → formatted list
│    {{transcript_block}}            → conv-grouped turns
│    {{tool_calls_summary}}          → count per tool
├─ Call Opus with CONSOLIDATION_TOOLS:
│    Iteration 1: Opus emits 3x propose_new_memory + 1x propose_supersede.
│    Runner: acknowledge each tool_use, feed back neutral results, continue.
│    Iteration 2: Opus emits 1x propose_new_memory + finalize.
│    Runner: exits loop.
├─ Proposals collected: 5 total.
├─ Validator: rejects 1 (duplicate of existing memory), accepts 4.
├─ dry_run=false → apply in single SQL transaction:
│    - 3 memories saved with origin="consolidation"
│    - 1 supersede applied
├─ Emit consolidation_run event with summary.
├─ Release lock. Exit 0.
```

## Definition of done

- [ ] `carl consolidate --for-date=YYYY-MM-DD` works with real data.
- [ ] `--dry-run` writes proposals to `~/.carl/consolidation-proposals-{date}.json` and changes zero DB rows.
- [ ] Lock file prevents overlapping runs (second run exits cleanly with a warning).
- [ ] Validators enforce: max 20 proposals, 500-char content cap, PII pattern rejection, duplicate detection converts to supersede.
- [ ] `consolidation_run` event emitted with full summary (turns_scanned, memories_saved, cost_usd, etc.).
- [ ] Cost tracking: each run's `cost_usd` reflects input+output token pricing for Opus 4.7.
- [ ] Golden test: fixture event log for a fake day produces expected proposals (deterministic given temperature=0 or seeded prompt).
- [ ] Crontab entry documented in README under "Setting up nightly consolidation".
- [ ] `pytest tests/test_consolidation.py tests/test_validators.py` passes.

## Acceptance tests

**Test 1 — Preference emergence:**
1. Over a few days, deliberately ask Carl for weather in Fahrenheit three times.
2. Run `carl consolidate --for-date=today` after the third day (or wait for cron).
3. Expected: new memory of type `preference`, content something like "User prefers weather in Fahrenheit."
4. Next weather question → Carl uses Fahrenheit without asking.

**Test 2 — Correction captured:**
1. Session 1: *"Ruby is a golden retriever."*
2. Session 2: *"Actually, Ruby's a labrador — I was wrong before."*
3. Run consolidation.
4. Expected: the original memory is superseded; new memory is "User's dog Ruby is a labrador."

**Test 3 — Dry-run safety:**
1. `carl consolidate --for-date=today --dry-run`
2. Inspect `~/.carl/consolidation-proposals-{date}.json` — see all proposals.
3. Verify `memories` table unchanged (`SELECT COUNT(*) FROM memories WHERE created_at > <time-of-run>` = 0).

**Test 4 — Lock behavior:**
1. Start `carl consolidate --for-date=yesterday` in one terminal (add a sleep to make it run long enough to race).
2. Immediately start another in another terminal.
3. Second run exits with "consolidation already running" message; DB touched only once.

**Test 5 — Cost sanity:**
After a full week of use, sum of `cost_usd` from `consolidation_run` events should be under $3.

## Traps / lessons

- **Prompt fragility**: consolidation prompts are prone to over-triggering ("save this trivial fact") or under-triggering ("save nothing today"). Iterate the prompt based on the proposals it generates. Use dry-run mode for the first 2 weeks in production.
- **`propose_*` tool schema drift**: if Opus starts generating malformed proposals, the validator catches it — but iterate the schema descriptions to prevent it happening.
- **Timezone bugs**: "yesterday" is local calendar day. On DST transitions, this gets weird. Use `zoneinfo` with an explicit user timezone, not `datetime.now()`.
- **Retention interaction**: consolidation reads events for yesterday. If retention job archives before consolidation runs, consolidation sees empty. Run order: consolidation FIRST (say 3am), retention SECOND (say 4am). Cron order matters.
- **Cost surprise**: if Opus decides to spend 5 iterations debating a proposal, cost per run can double. Cap iterations at 10 in the code (per pseudocode above). If a run hits the cap, log a warning — probably a prompt issue.
- **PII sweep is not compliance**: the regex-based PII check catches SSN and card patterns but not addresses, emails, medical info. For a personal single-user assistant on your own machine, filesystem permissions are enough. Don't confuse this with a real DLP program.
- **First run on a fresh install**: with zero prior memories, Opus can go wild proposing everything. That's fine — dry-run for the first week to sanity check.
