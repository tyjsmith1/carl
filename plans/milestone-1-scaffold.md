# Milestone 1 — Scaffold

## Goal

Stand up the whole project structure with every interface, config loader, event log, and pipeline FSM in place — but every stage is a mock. Nothing hits a real API. The result is a program you can run: it prints `wake_detected` when you press enter, feeds a canned transcript through a mock reasoner, and "speaks" by printing text.

**Why do this before real integrations?** Because the wiring is the risky part. Interfaces, FSM, event log, config, tool registry, memory schema — all of these are load-bearing and interact with each other. Getting them right against mocks means the real integrations in later milestones are drop-in replacements. It also means the whole codebase compiles and runs end-to-end from day 1, which catches architectural mistakes early.

## Prerequisites

- Python 3.11+ installed.
- `uv` or `pip` package manager.
- Empty repo (`/Users/tylersmith/Desktop/dev/assistant/`).
- Specs read: `config.md`, `event-log.md`, `conversation-state.md`, `memory.md`, `reasoner.md`, `pipeline.md`, `tools.md`.

## Non-goals

- No real audio I/O yet.
- No real Anthropic/Deepgram/OpenAI/Porcupine/Tavily calls.
- No memory retrieval or consolidation.
- No barge-in.

## Repo layout to create

```
assistant/
├── pyproject.toml                    # uv/pip config, deps, ruff, pytest
├── .env.example                      # committed template
├── .gitignore
├── config/
│   ├── carl.yaml                     # canonical, with `providers: *: mock`
│   └── prompts/
│       ├── carl_base.md              # starter personality
│       └── filler_phrases.yaml
├── carl/
│   ├── __init__.py
│   ├── main.py                       # entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── schema.py                 # Pydantic models
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── fsm.py                    # State + PipelineEvent + transitions
│   │   ├── router.py                 # AudioRouter (mock-friendly)
│   │   └── orchestrator.py           # Pipeline class, the run loop
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── wake_word/
│   │   │   ├── base.py               # WakeWord Protocol
│   │   │   └── mock.py               # StdinTriggeredWakeWord
│   │   ├── vad/
│   │   │   ├── base.py
│   │   │   └── mock.py               # ScriptedVAD
│   │   ├── stt/
│   │   │   ├── base.py
│   │   │   └── mock.py               # CannedTranscriptSTT
│   │   ├── reasoner/
│   │   │   ├── base.py
│   │   │   └── mock.py               # EchoReasoner (says "you said: X")
│   │   └── tts/
│   │       ├── base.py
│   │       └── mock.py               # PrintTTS (prints instead of speaking)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py                  # MemoryStore Protocol
│   │   ├── sqlite_store.py           # SQLiteMemoryStore
│   │   └── models.py                 # Memory dataclass
│   ├── tools/
│   │   ├── __init__.py               # decorator + registry
│   │   ├── base.py                   # ToolProvider, ToolDefinition
│   │   ├── native_provider.py        # NativeToolProvider
│   │   └── examples/
│   │       └── noop.py               # one dummy tool for wiring test
│   ├── events/
│   │   ├── __init__.py
│   │   ├── log.py                    # EventLog
│   │   └── schema.py                 # SQL migrations
│   └── storage/
│       ├── __init__.py
│       ├── db.py                     # SQLite connection factory (WAL mode)
│       └── migrations/
│           ├── __init__.py
│           └── 001_initial.sql
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_fsm.py
    ├── test_event_log.py
    ├── test_memory_store.py
    ├── test_tool_registry.py
    └── test_pipeline_e2e_mock.py     # the "does the whole thing wire up" test
```

## Files: what goes in each

### `pyproject.toml`

```toml
[project]
name = "carl"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40",                # reasoner + consolidation
    "pydantic>=2.5",                  # config schema
    "pyyaml>=6.0",                    # config loading
    "python-dotenv>=1.0",             # .env
    "ulid-py>=1.1",                   # IDs
    "structlog>=24",                  # structured logging
    # Real integrations (mocked at this milestone but installed):
    "sounddevice>=0.4",
    "numpy>=1.26",
    "silero-vad>=5.1",
    "pvporcupine>=3.0",
    "deepgram-sdk>=7.6,<8",           # v2 API surface (client.listen.v2.connect); v3/v4 patterns are dead
    "openai>=1.40",
    "tavily-python>=0.5",
    "watchfiles>=0.21",               # prompt hot-reload
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.6", "mypy>=1.10"]

[project.scripts]
carl = "carl.main:main"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### `config/carl.yaml`

Per `config.md` schema, with all `providers.*: mock` for this milestone:

```yaml
version: 1
providers:
  wake_word: mock
  vad:       mock
  stt:       mock
  reasoner:  mock
  tts:       mock
  search:    mock
# ... rest of file per config.md, all real-provider blocks kept but unused
```

### `carl/config/schema.py`

- Pydantic models per section of `carl.yaml`.
- `Literal[...]` types for provider enums with a `"mock"` option added.
- `Config.load(path)` classmethod that merges base + local override, validates.
- Fail-loud errors on missing paths, unknown providers.

### `carl/pipeline/fsm.py`

- `class State(Enum): IDLE, RECORDING, THINKING, SPEAKING, ERROR`
- `PipelineEvent` dataclass with `type: str` and event-specific payload.
- Transition table as `{(State, event_type): handler}`.
- No side effects in the FSM itself — handlers call out to injected stages.

### `carl/pipeline/orchestrator.py`

The main run loop. Per `conversation-state.md`:

```python
class Pipeline:
    def __init__(self, config, stages, memory_store, event_log, registry):
        self.state = State.IDLE
        # ... hold refs
        self.event_queue = asyncio.Queue()

    async def run(self):
        # 1. Start wake-word listener as a background task.
        # 2. Loop: await event, dispatch to handler, log transition.
        # Exit only on SIGTERM (main.py wires signal handling).
```

Each state transition handler is a small method:

```python
async def _on_wake_detected(self, event):
    assert self.state == State.IDLE
    self.state = State.RECORDING
    self._current_conversation_id = ulid.new().str
    self._current_turn_id = ulid.new().str
    self.event_log.log(stage="fsm", event_type="state_transition",
                       payload={"from": "IDLE", "to": "RECORDING"},
                       conversation_id=self._current_conversation_id)
    # Kick off STT stream from the router's ring buffer + live audio.
    self._current_stt = await self.stages.stt.open_stream(self._on_stt_transcript)
    # (mock STT immediately returns a canned transcript, triggering speech_end)
```

### Mock stage implementations

**`StdinTriggeredWakeWord`** — a coroutine that reads from stdin. When user hits Enter, emit `wake_detected`. When user types `exit`, gracefully shutdown.

**`ScriptedVAD`** — for each `open_stream` call, fires `speech_start` immediately, then `speech_end` after 500ms.

**`CannedTranscriptSTT`** — returns a hard-coded transcript from a rotating list (`["when was chicago founded", "hello", "what time is it", ...]`) or reads from `MOCK_STT_TRANSCRIPTS` env var.

**`EchoReasoner`** — takes a transcript, yields `SpokenChunk("You said: {transcript}.", is_final=True)`. Later replaced by real Claude.

**`PrintTTS`** — instead of playing audio, prints `[TTS] {text}` to stdout with a colored prefix.

### Memory store + event log

Real SQLite implementation for both (per `memory.md` and `event-log.md`). Not mocked — the schema and query patterns are the same in mock mode as production, and testing memory-touching code needs a real DB.

- Migrations run at startup; use `PRAGMA user_version` to track.
- WAL mode enabled.
- `~/.carl/carl.db` created if missing.

### Tool registry

- `@tool` decorator (per `tools.md`).
- `NativeToolProvider` that discovers `@tool`-decorated classes under `carl/tools/`.
- One example tool (`carl/tools/examples/noop.py`) that just returns `"ok"` — smoke test the wiring, LLM isn't hooked up yet so nothing will actually invoke it.

### `carl/main.py`

```python
def main():
    load_dotenv()
    config = Config.load()
    logging_setup(config.logging)

    event_log = EventLog(config.storage.sqlite_path)
    memory_store = SQLiteMemoryStore(config.storage.sqlite_path)
    apply_migrations(config.storage.sqlite_path)

    stages = build_stages(config)               # picks impls per config.providers
    registry = build_tool_registry(config, memory_store)

    pipeline = Pipeline(config, stages, memory_store, event_log, registry)

    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(pipeline.shutdown()))
    loop.run_until_complete(pipeline.run())
```

## Pseudocode for the tricky bits

### Stage factory

```python
def build_stages(config: Config) -> Stages:
    wake_word_cls = WAKE_WORD_IMPLS[config.providers.wake_word]   # "mock" → StdinTriggeredWakeWord
    wake_word = wake_word_cls(config.wake_word.dict()[config.providers.wake_word])
    # ... same for each stage
    return Stages(wake_word=wake_word, vad=vad, stt=stt, reasoner=reasoner, tts=tts)
```

Each stage's config subsection is passed by matching key. Only the selected impl's config block is read.

### FSM transition dispatch

```python
class Pipeline:
    _TRANSITIONS = {
        (State.IDLE,      "wake_detected"):    "_on_wake_detected",
        (State.RECORDING, "speech_end"):       "_on_speech_end",
        # ...
    }

    async def _handle(self, event):
        method_name = self._TRANSITIONS.get((self.state, event.type))
        if method_name is None:
            log.warn("illegal_transition", state=self.state.name, event=event.type)
            return
        handler = getattr(self, method_name)
        await handler(event)
```

Cleaner than a giant if/elif. Method-per-transition keeps each handler small and testable.

### Ring buffer prepend on wake

```python
async def _on_wake_detected(self, event):
    # ... transition to RECORDING
    self._current_stt = await self.stages.stt.open_stream(...)
    # Prepend the router's ~500ms ring buffer to STT stream
    await self._router.replay_ring(self._current_stt.send)
    # Router will now also stream live frames to STT
    self._router.activate("vad", "stt")
```

At mock milestone, `CannedTranscriptSTT.send()` is a no-op — it just returns its canned transcript regardless of audio.

## Definition of done

- [ ] Repo initializes with `uv sync` (or `pip install -e .[dev]`).
- [ ] `carl --version` works.
- [ ] `carl` runs, prints startup banner, sits waiting on stdin.
- [ ] Hitting Enter prints (in order):
  ```
  [FSM] IDLE → RECORDING
  [STT] transcript: when was chicago founded
  [FSM] RECORDING → THINKING
  [FSM] THINKING → SPEAKING
  [TTS] You said: when was chicago founded.
  [FSM] SPEAKING → IDLE (after follow-up window)
  ```
- [ ] `~/.carl/carl.db` created with tables per migration 001.
- [ ] Every FSM transition and every stage event has a row in the `events` table.
- [ ] `pytest` passes (integration test `test_pipeline_e2e_mock.py` runs the above sequence and asserts event log contents).
- [ ] `ruff check` and `mypy carl/` clean.

## Acceptance test (manual)

```
$ carl
carl 0.1.0 — mock providers loaded
IDLE. press enter to trigger wake word, type 'exit' to quit.
> [enter]
[FSM] IDLE → RECORDING (conv=01H..., turn=01H...)
[STT] partial: "when"
[STT] partial: "when was"
[STT] final: "when was chicago founded"
[FSM] RECORDING → THINKING
[REASONER] responding...
[FSM] THINKING → SPEAKING
[TTS] "You said: when was chicago founded."
[FSM] SPEAKING → RECORDING (follow-up window)
[FSM] RECORDING → IDLE (timeout)
IDLE. press enter to trigger wake word, type 'exit' to quit.
> exit
[FSM] shutdown clean
$ sqlite3 ~/.carl/carl.db "SELECT stage, event_type FROM events ORDER BY id;"
fsm|state_transition
stt|stt_final
fsm|state_transition
fsm|state_transition
tts|tts_complete
fsm|state_transition
fsm|state_transition
```

## Estimated size

~1500 lines of Python across the modules listed. Roughly:
- Config + schema: ~200
- FSM + orchestrator: ~350
- Router: ~150
- Mock stages (5): ~250 total
- SQLite memory store + migrations: ~300
- Event log: ~150
- Tool registry: ~150
- Tests: ~300

## Traps / lessons

- **`asyncio` + `sounddevice` callback thread**: sounddevice callbacks run on a separate audio thread, NOT the asyncio loop. Any dispatch from the callback into async code must use `loop.call_soon_threadsafe()` or a `janus.Queue`. The mock milestone dodges this (StdinTriggeredWakeWord runs pure async), but the pattern needs to be right in the router for milestone 2.
- **SQLite + WAL + asyncio**: use `aiosqlite` OR run sync SQLite in a thread executor. Do NOT block the loop on writes. The event log's batched writer task uses one dedicated thread.
- **Pydantic v2 `Literal` unions**: adding `"mock"` alongside real provider names means those literals appear in the config type. Fine — mock will get stripped out of real config once we cut a release, or gated behind `--dev-mode`.
- **Test isolation**: every test that touches the DB uses `tmp_path / "test.db"`. Never share DBs between tests.
