"""Carl CLI entry point.

Subcommands:
    carl                 Run the interactive voice loop (default).
    carl run             Same as `carl`.
    carl migrate         Apply pending SQL migrations, print version.
    carl tools list      List registered tools.
    carl consolidate     (Milestone 6) run the nightly consolidation job.

Milestone 1 ships `run`, `migrate`, `tools list`. Others land as their
milestones do.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path

import certifi
import click
import structlog
from dotenv import load_dotenv

# macOS Python 3.12 (Python.org installer) ships without a wired-up CA bundle,
# so requests / websockets against Deepgram / Anthropic / OpenAI fail with
# CERTIFICATE_VERIFY_FAILED. Point every stdlib SSL user at certifi's bundle.
# Skipped if the user has already set SSL_CERT_FILE explicitly.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from carl.config import Config
from carl.events import EventLog
from carl.memory import SQLiteMemoryStore
from carl.pipeline import AudioRouter, Pipeline
from carl.stages import build_stages
from carl.storage.db import apply_migrations
from carl.tools import NativeToolProvider, ToolRegistry

VERSION = "0.1.0"

log = structlog.get_logger()


def _setup_logging(level_name: str, log_dir: Path | None = None) -> None:
    level = getattr(logging, level_name, logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_dir is not None:
        # Daily-rotated file log so past sessions are inspectable without
        # scrolling terminal buffers. Path is ~/.carl/logs/carl-YYYY-MM-DD.log
        # by default; rotates at midnight; keeps 30 days.
        from logging.handlers import TimedRotatingFileHandler
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            log_dir / "carl.log", when="midnight", backupCount=30, encoding="utf-8",
        )
        # Include the ISO date suffix on rotated files.
        file_handler.suffix = "%Y-%m-%d"
        handlers.append(file_handler)
    logging.basicConfig(format="%(message)s", level=level, handlers=handlers, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),  # colors would render as ANSI escapes in the file
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


@click.group(invoke_without_command=True)
@click.version_option(VERSION, prog_name="carl")
@click.option("--config", "config_path", default="config/carl.yaml", type=click.Path(),
              help="Path to config file")
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """Personal voice assistant."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    if ctx.invoked_subcommand is None:
        # Default: run.
        ctx.invoke(run)


@cli.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Run the interactive voice loop."""
    load_dotenv()
    config = Config.load(ctx.obj["config_path"])

    # Apply migrations before anything else touches the DB.
    db_path = config.storage.resolved_path()
    # Log dir sits next to the db (typically ~/.carl/logs/). Daily-rotated
    # file logs so past sessions are reviewable without terminal scrollback.
    log_dir = db_path.parent / "logs"
    _setup_logging(config.logging.level, log_dir=log_dir)

    schema_version = apply_migrations(db_path)
    log.info("db_ready", path=str(db_path), schema_version=schema_version)
    log.info("file_logging", log_dir=str(log_dir))

    memory_store = SQLiteMemoryStore(db_path, config=config.memory)
    stages = build_stages(config)

    # Discover native tools and build the registry.
    # Walk carl.tools (not just carl.tools.examples) so real tools like
    # web_search.py, memory tools (M4), etc. are picked up alongside the
    # noop example tool.
    NativeToolProvider.discover("carl.tools")

    # `providers` here gates tools via @tool(provider_group=...). Any tool
    # whose group corresponds to a "mock" provider is skipped at construction.
    # `injected` fills the tool's constructor kwargs by name.
    native_provider = NativeToolProvider(
        injected={
            "memory_store": memory_store,
            "tavily_config": config.search.tavily,
        },
        providers=config.providers.model_dump(),
    )
    tool_registry = ToolRegistry(providers=[native_provider])

    event_log = EventLog(db_path, stdout_mirror=config.logging.event_log_stdout)
    router = AudioRouter(
        input_device=config.audio.input_device,
        output_device=config.audio.output_device,
    )
    pipeline = Pipeline(
        config=config,
        stages=stages,
        memory_store=memory_store,
        event_log=event_log,
        tool_registry=tool_registry,
        router=router,
    )

    # Strong refs to shutdown tasks — Python 3.11+ only weakly references
    # active tasks, so a bare `asyncio.create_task(...)` from a signal
    # handler can be GC'd before it reaches its first await.
    shutdown_tasks: set[asyncio.Task] = set()

    async def _main() -> None:
        loop = asyncio.get_running_loop()

        def _sig_handler() -> None:
            log.info("signal_received_shutting_down")
            task = asyncio.create_task(pipeline.shutdown())
            shutdown_tasks.add(task)
            task.add_done_callback(shutdown_tasks.discard)

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _sig_handler)

        await tool_registry.initialize()
        print(f"carl {VERSION} — providers: "
              f"{config.providers.wake_word}, {config.providers.vad}, "
              f"{config.providers.stt}, {config.providers.reasoner}, "
              f"{config.providers.tts}", flush=True)
        await pipeline.run()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_main())
    print("[FSM] shutdown clean", flush=True)


@cli.command()
@click.pass_context
def migrate(ctx: click.Context) -> None:
    """Apply pending SQL migrations."""
    load_dotenv()
    config = Config.load(ctx.obj["config_path"])
    _setup_logging(config.logging.level)
    db_path = config.storage.resolved_path()
    version = apply_migrations(db_path)
    click.echo(f"schema at version {version} ({db_path})")


@cli.group()
def tools() -> None:
    """Tool-registry inspection."""


@tools.command("list")
@click.pass_context
def tools_list(ctx: click.Context) -> None:
    """List registered tools."""
    load_dotenv()
    config = Config.load(ctx.obj["config_path"])
    _setup_logging(config.logging.level)
    db_path = config.storage.resolved_path()
    apply_migrations(db_path)
    memory_store = SQLiteMemoryStore(db_path, config=config.memory)

    # Walk carl.tools (not just carl.tools.examples) so real tools like
    # web_search.py, memory tools (M4), etc. are picked up alongside the
    # noop example tool.
    NativeToolProvider.discover("carl.tools")
    provider = NativeToolProvider(
        injected={
            "memory_store": memory_store,
            "tavily_config": config.search.tavily,
        },
        providers=config.providers.model_dump(),
    )
    registry = ToolRegistry(providers=[provider])
    asyncio.run(registry.initialize())
    for defn in registry.definitions():
        internal_tag = " (internal)" if defn.internal else ""
        click.echo(f"  {defn.name} [{defn.safety_tier}]{internal_tag} — {defn.description[:80]}")


@cli.group()
def audio() -> None:
    """Audio device inspection + configuration helpers."""


@audio.command("list")
def audio_list() -> None:
    """List sounddevice's visible audio devices in a form suitable for pasting into carl.local.yaml.

    Devices marked (default in) / (default out) are what "input_device: default"
    and "output_device: default" resolve to in the current shell. To pin a
    specific device across shell / user changes, copy its name or index into
    config/carl.local.yaml under the `audio:` block.
    """
    try:
        import sounddevice as sd  # heavy import; lazy so `carl --version` stays fast
    except OSError as e:
        click.echo(f"sounddevice failed to load PortAudio: {e}", err=True)
        click.echo("On macOS: `brew install portaudio` and reinstall sounddevice.", err=True)
        raise SystemExit(1) from e

    try:
        default_in, default_out = sd.default.device
    except (TypeError, ValueError):
        default_in, default_out = None, None

    devices = sd.query_devices()
    click.echo("index  name                                             in  out  tags")
    click.echo("─" * 78)
    for i, dev in enumerate(devices):
        tags = []
        if i == default_in:
            tags.append("default in")
        if i == default_out:
            tags.append("default out")
        tag_str = ", ".join(tags) or ""
        click.echo(
            f"{i:>4}   {dev['name'][:48]:<48} {dev['max_input_channels']:>2}  "
            f"{dev['max_output_channels']:>2}   {tag_str}"
        )

    click.echo()
    click.echo("To pin a device, copy its name (or index) into config/carl.local.yaml:")
    click.echo("  audio:")
    click.echo("    input_device: \"MacBook Pro Microphone\"    # or an integer index")
    click.echo("    output_device: \"MacBook Pro Speakers\"")


if __name__ == "__main__":
    cli()
