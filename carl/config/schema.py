"""Pydantic schema for Carl's config. See docs/specs/config.md for the canonical spec."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ─── Provider selection ──────────────────────────────────────────────────────

WakeWordProvider = Literal["mock", "openwakeword"]
VADProvider = Literal["mock", "silero", "webrtc"]
STTProvider = Literal["mock", "deepgram", "whisper_api", "whisper_local"]
ReasonerProvider = Literal["mock", "claude_sonnet", "claude_opus", "claude_haiku"]
TTSProvider = Literal["mock", "openai", "elevenlabs", "piper"]
SearchProvider = Literal["mock", "tavily", "brave", "serper"]


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wake_word: WakeWordProvider = "mock"
    vad: VADProvider = "mock"
    stt: STTProvider = "mock"
    reasoner: ReasonerProvider = "mock"
    tts: TTSProvider = "mock"
    search: SearchProvider = "mock"


# ─── Per-provider configs ────────────────────────────────────────────────────


class OpenWakeWordConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # For MVP: use a pre-trained model name from openwakeword's model list
    # (e.g. "hey_jarvis", "alexa", "hey_mycroft"). To use a custom-trained
    # model, set model_path to the .tflite/.onnx file path instead.
    model_name: str = "hey_jarvis"
    model_path: str | None = None            # overrides model_name if set
    threshold: float = 0.5                    # 0.0-1.0; higher = fewer false accepts
    # onnx is the default because tflite-runtime is a separate install
    # (`pip install tflite-runtime`) that isn't always available on macOS ARM.
    # onnxruntime is a hard dep of openwakeword, so it always works.
    inference_framework: Literal["tflite", "onnx"] = "onnx"


class WakeWordConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    openwakeword: OpenWakeWordConfig | None = None


class SileroConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = 0.5
    min_silence_ms_utterance_end: int = 700
    min_silence_ms_bargein: int = 150
    speech_pad_ms: int = 100
    sample_rate: int = 16000


class VADConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    silero: SileroConfig | None = None


class DeepgramConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "nova-3"
    language: str = "en-US"
    interim_results: bool = True
    smart_format: bool = True
    endpointing_ms: int = 300
    utterance_end_ms: int = 1000


class STTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deepgram: DeepgramConfig | None = None


class ClaudeReasonerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    max_tokens: int = 1024
    system_prompt_path: str | None = None
    streaming: bool = True
    tool_choice: Literal["auto", "any", "none"] = "auto"
    temperature: float = 0.7


class ReasonerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claude_sonnet: ClaudeReasonerConfig | None = None
    claude_opus: ClaudeReasonerConfig | None = None
    claude_haiku: ClaudeReasonerConfig | None = None


class OpenAITTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "gpt-4o-mini-tts"
    voice: str = "nova"
    response_format: Literal["pcm", "mp3", "opus", "aac", "flac", "wav"] = "pcm"
    sample_rate: int = 24000
    # OpenAI clamps to 0.25–4.0. 1.0 = natural pace; 1.15–1.25 feels "brisk"
    # without pitch-shifting artifacts; above ~1.5 starts sounding chipmunky.
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions_path: str | None = None


class ElevenLabsTTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # voice_id is the UUID-ish identifier from ElevenLabs — grab it from
    # the voice's URL in the ElevenLabs UI or via the voices API.
    voice_id: str
    # eleven_flash_v2_5 is the low-latency model (~200-300ms TTFB), best for
    # conversation. eleven_multilingual_v2 sounds better but adds 500+ms.
    model_id: str = "eleven_flash_v2_5"
    # Must match router.OUT_SAMPLE_RATE (24000) so the router can play the
    # audio without a resampling stage. pcm_24000 = 16-bit LE mono @ 24kHz.
    output_format: str = "pcm_24000"
    # Voice-settings knobs. Defaults chosen for natural-sounding conversation:
    # moderate stability (varied prosody), high similarity to the source
    # voice, no style exaggeration. Tune per voice.
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    style: float = Field(default=0.0, ge=0.0, le=1.0)
    use_speaker_boost: bool = True
    # ElevenLabs speed range is much tighter than OpenAI's (0.7–1.2).
    speed: float = Field(default=1.0, ge=0.7, le=1.2)


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    openai: OpenAITTSConfig | None = None
    elevenlabs: ElevenLabsTTSConfig | None = None


class TavilyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_depth: Literal["basic", "advanced"] = "basic"
    max_results: int = 5
    include_answer: bool | Literal["basic", "advanced"] = True


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tavily: TavilyConfig | None = None


# ─── Other sections ──────────────────────────────────────────────────────────


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_device: str | int = "default"
    output_device: str | int = "default"
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    channels: int = 1


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sqlite_path: str = "~/.carl/carl.db"
    event_log_retention_days: int = 30

    def resolved_path(self) -> Path:
        return Path(self.sqlite_path).expanduser()


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrieval_strategy: Literal["dump_all", "embed_topk"] = "dump_all"
    top_k: int = 8
    embedding_model: str = "text-embedding-3-small"
    auto_inject: bool = True
    narrate_recall: bool = False


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    require_confirmation: list[str] = Field(default_factory=list)
    filler_speech: bool = True
    parallel_execution: bool = True
    max_iterations: int = 5
    tool_timeout_seconds: int = 15


class ConversationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bargein_enabled: bool = True
    utterance_max_seconds: int = 30
    llm_timeout_seconds: int = 20
    tts_timeout_seconds: int = 10
    history_turns: int = 10
    # Follow-up window: after Carl finishes speaking, keep the mic open and
    # listen for the user to keep talking without needing another wake word.
    # Enables natural back-and-forth ("what's the weather" → answer → "and
    # tomorrow?" all in one exchange). Bumped from 3s to 7s to give the user
    # time to think between Carl's answer and their next thought.
    follow_up_ms: int = 7000
    # Guard delay at the start of the follow-up window before VAD activates.
    # Prevents Carl's own TTS tail-echo (still bleeding out of desktop
    # speakers on setups without acoustic echo cancellation) from
    # false-triggering VAD as if it were the user speaking.
    followup_guard_ms: int = 600
    # Cap on consecutive follow-up turns per wake-word cycle. Prevents a
    # runaway feedback loop when the mic picks up Carl's own speech,
    # background TV, or other ambient chatter and keeps triggering new turns.
    # After the cap, requires a fresh "hey Jarvis" wake to continue.
    max_followup_turns: int = 3
    no_speech_timeout_ms: int = 5000


class ConsolidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    cron: str = "0 3 * * *"
    dry_run: bool = False
    model: str = "claude-opus-4-7"
    max_tokens: int = 4096


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    event_log_stdout: bool = False


# ─── Root ────────────────────────────────────────────────────────────────────


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    reasoner: ReasonerConfig = Field(default_factory=ReasonerConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: Path | str = "config/carl.yaml") -> Config:
        base_path = Path(path)
        if not base_path.exists():
            raise FileNotFoundError(f"config not found at {base_path.resolve()}")
        base_data: dict = yaml.safe_load(base_path.read_text()) or {}

        local_path = base_path.with_name(base_path.stem + ".local" + base_path.suffix)
        if local_path.exists():
            local_data: dict = yaml.safe_load(local_path.read_text()) or {}
            base_data = deep_merge(base_data, local_data)

        return cls.model_validate(base_data)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; override wins on scalar collisions. Public so tests can hit it."""
    result = dict(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
