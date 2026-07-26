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
import signal
import sys

import click
import structlog
from dotenv import load_dotenv

from carl.config import Config
from carl.events import EventLog
from carl.memory import SQLiteMemoryStore
from carl.pipeline import AudioRouter, Pipeline
from carl.stages import build_stages
from carl.storage.db import apply_migrations
from carl.tools import NativeToolProvider, ToolRegistry

VERSION = "0.1.0"

log = structlog.get_logger()


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
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
    _setup_logging(config.logging.level)

    # Apply migrations before anything else touches the DB.
    db_path = config.storage.resolved_path()
    schema_version = apply_migrations(db_path)
    log.info("db_ready", path=str(db_path), schema_version=schema_version)

    memory_store = SQLiteMemoryStore(db_path, config=config.memory)
    stages = build_stages(config)

    # Discover native tools and build the registry.
    NativeToolProvider.discover("carl.tools.examples")
    native_provider = NativeToolProvider(
        injected={"memory_store": memory_store}
    )
    tool_registry = ToolRegistry(providers=[native_provider])

    event_log = EventLog(db_path, stdout_mirror=config.logging.event_log_stdout)
    router = AudioRouter()
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

    NativeToolProvider.discover("carl.tools.examples")
    provider = NativeToolProvider(injected={"memory_store": memory_store})
    registry = ToolRegistry(providers=[provider])
    asyncio.run(registry.initialize())
    for defn in registry.definitions():
        click.echo(f"  {defn.name} [{defn.safety_tier}] — {defn.description[:80]}")


if __name__ == "__main__":
    cli()
