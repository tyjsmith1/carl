"""Per-turn metrics + rough cost estimation.

We track usage (tokens, characters, audio bytes) per turn and emit a
compact `turn_complete` log line so scanning "what happened this
session" is a single grep instead of a jigsaw of state transitions.

Cost estimates use rough pricing constants — they'll drift as providers
change pricing. Treat the dollar figures as ballpark, not billing.
Update PRICING below when pricing shifts.

Skipped by design:
- Precise cost accounting (use provider dashboards)
- Per-request tracing (event_log already captures this)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# ─── Pricing (USD per 1M units; roughly Q3 2026) ────────────────────
PRICING = {
    # Anthropic Sonnet 4.6
    "anthropic_input_per_mtok": 3.00,
    "anthropic_output_per_mtok": 15.00,
    # Deepgram Nova-3 streaming ($/minute of audio)
    "deepgram_per_minute": 0.0043,
    # OpenAI gpt-4o-mini-tts ($/1M input text tokens; roughly $15/1M chars in practice)
    "openai_tts_per_mchar": 15.00,
    # ElevenLabs Flash v2.5 on Creator plan; heavily plan-dependent
    "elevenlabs_per_mchar": 150.00,
}


@dataclass
class TurnMetrics:
    """Accumulated during one turn, emitted at turn_complete."""
    conversation_id: str
    turn_id: str
    started_at: float = field(default_factory=time.monotonic)
    # LLM
    input_tokens: int = 0
    output_tokens: int = 0
    llm_first_token_ms: int | None = None
    # STT (Deepgram)
    stt_audio_bytes: int = 0     # 16-bit @ 16kHz = 32000 bytes/sec
    # TTS
    tts_chars: int = 0
    tts_provider: str = "unknown"   # "openai" or "elevenlabs" — different pricing
    tts_first_byte_ms: int | None = None
    # Tools
    tools_called: list[str] = field(default_factory=list)
    # Turn outcome
    interrupted: bool = False

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)

    def stt_seconds(self) -> float:
        return self.stt_audio_bytes / (16000 * 2)

    def estimate_cost_usd(self) -> float:
        c = 0.0
        c += self.input_tokens / 1_000_000 * PRICING["anthropic_input_per_mtok"]
        c += self.output_tokens / 1_000_000 * PRICING["anthropic_output_per_mtok"]
        c += self.stt_seconds() / 60 * PRICING["deepgram_per_minute"]
        # Only bill known TTS providers. Mock/piper/unknown fall through
        # to $0 rather than inflating the openai bucket by default.
        tts_key = {
            "elevenlabs": "elevenlabs_per_mchar",
            "openai": "openai_tts_per_mchar",
        }.get(self.tts_provider)
        if tts_key:
            c += self.tts_chars / 1_000_000 * PRICING[tts_key]
        return c


@dataclass
class SessionMetrics:
    """Rolled up across a session — emitted at shutdown.

    `started_at` is set on the FIRST turn (not at pipeline construction)
    so elapsed_seconds reflects the actual conversation duration, not
    idle-plus-conversation. Process uptime is a different metric.
    """
    started_at: float | None = None
    turns_completed: int = 0
    turns_interrupted: int = 0
    turns_empty: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stt_audio_bytes: int = 0
    # Bucketed per provider so cost is right when mixed. `other` catches
    # mock/piper/future providers — DON'T lump them into openai (silently
    # inflates the openai cost line during dev runs).
    tts_chars_openai: int = 0
    tts_chars_elevenlabs: int = 0
    tts_chars_other: int = 0
    total_tools_called: int = 0
    latencies_ms: list[int] = field(default_factory=list)  # turn total latency

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

    def add_turn(self, m: TurnMetrics) -> None:
        # First real turn kicks off the session clock so 'elapsed_seconds'
        # measures conversation duration, not process idle time.
        if self.started_at is None:
            self.started_at = m.started_at
        if m.interrupted:
            self.turns_interrupted += 1
        else:
            self.turns_completed += 1
        self.input_tokens += m.input_tokens
        self.output_tokens += m.output_tokens
        self.stt_audio_bytes += m.stt_audio_bytes
        if m.tts_provider == "elevenlabs":
            self.tts_chars_elevenlabs += m.tts_chars
        elif m.tts_provider == "openai":
            self.tts_chars_openai += m.tts_chars
        else:
            self.tts_chars_other += m.tts_chars
        self.total_tools_called += len(m.tools_called)
        self.latencies_ms.append(m.elapsed_ms())

    def estimate_cost_usd(self) -> float:
        c = 0.0
        c += self.input_tokens / 1_000_000 * PRICING["anthropic_input_per_mtok"]
        c += self.output_tokens / 1_000_000 * PRICING["anthropic_output_per_mtok"]
        c += (self.stt_audio_bytes / (16000 * 2)) / 60 * PRICING["deepgram_per_minute"]
        c += self.tts_chars_openai / 1_000_000 * PRICING["openai_tts_per_mchar"]
        c += self.tts_chars_elevenlabs / 1_000_000 * PRICING["elevenlabs_per_mchar"]
        return c

    def avg_latency_ms(self) -> int:
        return int(sum(self.latencies_ms) / len(self.latencies_ms)) if self.latencies_ms else 0
