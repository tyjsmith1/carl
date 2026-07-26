"""Shared fixtures for the Carl test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from carl.config import Config
from carl.memory import SQLiteMemoryStore
from carl.storage.db import apply_migrations


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Fresh SQLite database with migrations applied, isolated to this test."""
    db = tmp_path / "carl.db"
    apply_migrations(db)
    return db


@pytest.fixture
def default_config(tmp_db: Path) -> Config:
    """Config with mock providers everywhere and a tmp storage path."""
    cfg = Config()
    cfg.storage.sqlite_path = str(tmp_db)
    return cfg


@pytest_asyncio.fixture
async def memory_store(tmp_db: Path) -> AsyncIterator[SQLiteMemoryStore]:
    """SQLite memory store that auto-drains fire-and-forget tasks at teardown."""
    store = SQLiteMemoryStore(tmp_db)
    yield store
    await store.close()
