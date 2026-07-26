# Config Spec

## Purpose

Single source of truth for which implementation runs at each pipeline stage, credentials/keys, and tuning parameters. Config is the seam that makes the modular architecture actually modular — swapping Deepgram for local whisper.cpp should be a one-line YAML change, no code touched.

## File layout

```
assistant/
├── config/
│   ├── carl.yaml          # canonical config, checked in
│   ├── carl.local.yaml    # optional override, gitignored
│   └── prompts/
│       ├── carl_base.md            # base system prompt (hot-reloaded)
│       ├── carl_sassy.md           # alt personality
│       ├── filler_phrases.yaml     # rotating "let me check" phrases
│       └── consolidation.md        # prompt for nightly job
├── .env                    # secrets only, gitignored
└── .env.example            # committed template
```

Resolution order: `carl.local.yaml` overrides `carl.yaml`; env vars override both for secret fields.

## YAML schema (annotated example)

```yaml
# config/carl.yaml
version: 1

# Which implementation runs at each stage. Values must map to registered
# provider names in code. Adding a new impl = registering it under a new key.
providers:
  wake_word:  porcupine        # porcupine | openwakeword | always_on (dev)
  vad:        silero           # silero | webrtc
  stt:        deepgram         # deepgram | whisper_api | whisper_local
  reasoner:   claude_sonnet    # claude_sonnet | claude_opus | claude_haiku
  tts:        openai           # openai | elevenlabs | piper
  search:     tavily           # tavily | brave | serper

# Per-provider settings. Only the keys for the *selected* provider are read;
# other blocks are ignored (useful for keeping alt configs alongside).
wake_word:
  porcupine:
    keyword_path: config/wake_words/hey_carl_mac.ppn
    sensitivity: 0.5           # 0.0 (few false-accepts) → 1.0 (few false-rejects)

vad:
  silero:
    threshold: 0.5
    min_silence_ms_utterance_end: 700   # how long silence = user done talking
    min_silence_ms_bargein: 150         # how quickly interruption cuts TTS
    speech_pad_ms: 100
    sample_rate: 16000

stt:
  deepgram:
    model: nova-3
    language: en-US
    interim_results: true
    smart_format: true
    endpointing_ms: 300                 # Deepgram-native end-of-turn (belt+suspenders w/ Silero)

reasoner:
  claude_sonnet:
    model: claude-sonnet-4-6
    max_tokens: 1024
    system_prompt_path: config/prompts/carl_base.md
    streaming: true
    tool_choice: auto
    temperature: 0.7

  claude_opus:                          # used by nightly consolidation job
    model: claude-opus-4-7
    max_tokens: 4096
    temperature: 0.3

tts:
  openai:
    model: gpt-4o-mini-tts              # supports voice instructions + streaming
    voice: nova
    response_format: pcm                # required for mid-stream cancellation
    sample_rate: 24000
    instructions_path: config/prompts/tts_style.txt   # optional

search:
  tavily:
    search_depth: basic                 # basic | advanced
    max_results: 5
    include_answer: true                # let Tavily pre-synthesize

# Audio I/O
audio:
  input_device: default                 # `sounddevice`-visible device name or index
  output_device: default
  input_sample_rate: 16000              # capture rate (Silero+Porcupine both want 16k)
  output_sample_rate: 24000             # TTS native rate (avoid resample)
  channels: 1

# Storage
storage:
  sqlite_path: ~/.carl/carl.db
  event_log_retention_days: 30          # events older than this get archived, not deleted

# Memory system
memory:
  retrieval_strategy: dump_all          # dump_all | embed_topk
  top_k: 8                              # only used if embed_topk
  embedding_model: text-embedding-3-small
  auto_inject: true                     # add retrieved memories to every prompt
  narrate_recall: false                 # silent per user preference

# Tool orchestration
tools:
  require_confirmation:                 # tool names in the "sensitive" safety tier
    - send_message
    - delete_all_memories
  filler_speech: true                   # speak filler phrase during tool execution
  parallel_execution: true              # execute tool_use blocks concurrently
  max_iterations: 5                     # safety cap on tool-use loop per turn

# Conversation state
conversation:
  bargein_enabled: true                 # Tier 2 VAD-triggered interrupt
  utterance_max_seconds: 30             # hard cap on a single user utterance
  llm_timeout_seconds: 20
  tts_timeout_seconds: 10
  history_turns: 10                     # how many prior turns kept in LLM context

# Nightly consolidation
consolidation:
  enabled: true
  cron: "0 3 * * *"                     # 3am local time
  dry_run: false                        # if true, propose but don't write

# Observability
logging:
  level: INFO                           # DEBUG | INFO | WARN | ERROR
  event_log_stdout: false               # also stream event log to stdout
```

## Secrets (`.env`)

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
PICOVOICE_ACCESS_KEY=...
TAVILY_API_KEY=tvly-...
```

Loaded via `python-dotenv`. **Never** put keys in YAML. `.env.example` gets committed with placeholder values so contributors know what's needed.

## Config loading

Pydantic v2 model per section:

```python
# carl/config/schema.py
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

class ProvidersConfig(BaseModel):
    wake_word: Literal["porcupine", "openwakeword", "always_on"]
    vad:       Literal["silero", "webrtc"]
    stt:       Literal["deepgram", "whisper_api", "whisper_local"]
    reasoner:  Literal["claude_sonnet", "claude_opus", "claude_haiku"]
    tts:       Literal["openai", "elevenlabs", "piper"]
    search:    Literal["tavily", "brave", "serper"]

class Config(BaseModel):
    version: int
    providers: ProvidersConfig
    # ... one submodel per top-level section
    
    @classmethod
    def load(cls, path: Path = Path("config/carl.yaml")) -> "Config":
        base = yaml.safe_load(path.read_text())
        local = Path("config/carl.local.yaml")
        if local.exists():
            deep_merge(base, yaml.safe_load(local.read_text()))
        return cls(**base)
```

Load once at startup; freeze the object; pass sub-configs into stage constructors.

## Hot-reload behavior

Only prompts hot-reload. Everything else requires restart.

- `config/prompts/*.md` — watched with `watchfiles`; on change, the reasoner's system prompt is swapped in for the *next* turn (mid-turn is untouched to avoid mid-generation weirdness).
- Rationale: prompts are the iteration surface for personality tuning. Everything else is stable enough that restart is fine.

## Failure modes

| Failure | Detection | Behavior |
|---|---|---|
| Missing YAML file | Startup | Fatal — print path expected, exit 1 |
| Invalid schema (wrong types, unknown provider) | Pydantic validation | Fatal — print error with field path |
| Missing env var referenced by chosen provider | Provider construction | Fatal — print which key is missing |
| Referenced prompt file missing | Reasoner init | Fatal — print path |
| Wake-word `.ppn` file missing | Wake-word init | Fatal — print path, link to Console |
| Local override references a provider not in schema | Load | Fatal with clear "unknown provider `foo`" |
| Prompt file changed on disk | Watchfiles callback | Log INFO, swap for next turn |
| Prompt file becomes invalid mid-run | Reasoner call | Log ERROR, keep last-known-good |

Fail loud and early. Bad config should never make it to first audio.

## Test strategy

- **Unit**: Pydantic model round-trips a golden YAML fixture. Rejects malformed configs.
- **Unit**: env-var interpolation. Missing keys raise cleanly.
- **Unit**: local-override merge respects deep-merge semantics (not top-level replace).
- **Integration**: startup with the canonical `carl.yaml` and stub `.env` succeeds and instantiates every declared provider.
- **Integration**: prompt hot-reload — touch a prompt file, verify next turn uses the new content.

## Open questions

- [OPEN] Do we want a `carl.dev.yaml` profile as well (e.g. `always_on` wake word, mock STT, verbose logging)? Punt until we feel the pain.
- [OPEN] Version-bump strategy for `version:` field when schema evolves. Probably fail-with-helpful-migration-message.
