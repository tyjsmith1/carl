"""SQLite connection helpers + migration runner.

Design:
- WAL mode enabled for concurrent reads while a writer holds the write lock.
- Migrations live as numbered .sql files in carl/storage/migrations/.
- PRAGMA user_version tracks the applied migration count.
- Both sync (`connect`) and async (`aconnect`) helpers provided — the event log's
  batched writer uses sync (in a dedicated thread); readers use aiosqlite.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

import aiosqlite


def ensure_parent(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path) -> sqlite3.Connection:
    """Sync connection. Caller is responsible for closing.
    If a PRAGMA fails after the raw connect succeeds, we close the connection
    before re-raising so the caller never receives a partially-initialized handle."""
    ensure_parent(db_path)
    conn = sqlite3.connect(str(db_path), isolation_level=None)   # autocommit; we use BEGIN explicitly
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn
    except Exception:
        conn.close()
        raise


async def aconnect(db_path: Path) -> aiosqlite.Connection:
    """Async connection via aiosqlite. Caller is responsible for closing.
    Closes the connection (and its background thread) on PRAGMA failure so
    aborted startups don't leak threads."""
    ensure_parent(db_path)
    conn = await aiosqlite.connect(str(db_path))
    try:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL;")
        await conn.execute("PRAGMA foreign_keys = ON;")
        await conn.execute("PRAGMA synchronous = NORMAL;")
        return conn
    except Exception:
        await conn.close()
        raise


# ─── Migrations ──────────────────────────────────────────────────────────────


def _iter_migrations() -> Iterator[tuple[int, str]]:
    """Yield (version, sql_text) in ascending order."""
    migration_files = sorted(
        resources.files("carl.storage.migrations").iterdir(),
        key=lambda p: p.name,
    )
    for f in migration_files:
        if not f.name.endswith(".sql"):
            continue
        try:
            version = int(f.name.split("_", 1)[0])
        except ValueError:
            continue
        yield version, f.read_text()


def apply_migrations(db_path: Path) -> int:
    """Apply any pending migrations. Returns the final schema version.

    Migrations run in ascending numeric order. Each migration is expected to end
    with `PRAGMA user_version = N;` to advance the version. If a migration file
    fails partway, SQLite is left at the last successful version.
    """
    ensure_parent(db_path)
    conn = connect(db_path)
    try:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        for version, sql in _iter_migrations():
            if version <= current:
                continue
            conn.executescript(sql)
            new = conn.execute("PRAGMA user_version").fetchone()[0]
            if new != version:
                raise RuntimeError(
                    f"migration {version} did not set PRAGMA user_version={version} "
                    f"(saw {new}); refusing to continue"
                )
            current = new
        return current
    finally:
        conn.close()
