"""MemoryStore tests. See docs/specs/memory.md."""

from __future__ import annotations

import pytest

from carl.memory import SQLiteMemoryStore


@pytest.mark.asyncio
async def test_save_and_get(memory_store: SQLiteMemoryStore) -> None:
    m = await memory_store.save(content="user has a dog named Ruby", type="fact", subject="dog")
    assert m.id
    assert m.type == "fact"
    assert m.short_id == m.id[-6:]

    fetched = await memory_store.get(m.id)
    assert fetched is not None
    assert fetched.content == m.content


@pytest.mark.asyncio
async def test_retrieve_orders_by_type_priority(memory_store: SQLiteMemoryStore) -> None:
    fact = await memory_store.save(content="F", type="fact")
    pref = await memory_store.save(content="P", type="preference")
    routine = await memory_store.save(content="R", type="routine")
    correction = await memory_store.save(content="C", type="correction")

    results = await memory_store.retrieve(query="anything", k=10)
    types = [m.type for m in results]
    # correction < preference < routine < fact
    assert types == ["correction", "preference", "routine", "fact"]

    # Sanity that all 4 came back
    ids = {m.id for m in results}
    assert ids == {fact.id, pref.id, routine.id, correction.id}


@pytest.mark.asyncio
async def test_supersede_hides_old(memory_store: SQLiteMemoryStore) -> None:
    old = await memory_store.save(content="Ruby is a labrador", type="fact", subject="dog")
    new = await memory_store.supersede(old.id, "Ruby is a golden retriever")

    active = await memory_store.list()
    assert new.id in {m.id for m in active}
    assert old.id not in {m.id for m in active}

    # Old still exists if we ask for superseded
    with_superseded = await memory_store.list(include_superseded=True)
    assert old.id in {m.id for m in with_superseded}


@pytest.mark.asyncio
async def test_tombstone_hides(memory_store: SQLiteMemoryStore) -> None:
    m = await memory_store.save(content="temp fact", type="fact")
    await memory_store.tombstone(m.id)
    active = await memory_store.list()
    assert m.id not in {x.id for x in active}


@pytest.mark.asyncio
async def test_find_by_short_id_unique(memory_store: SQLiteMemoryStore) -> None:
    m = await memory_store.save(content="fact", type="fact")
    found = await memory_store.find_by_short_id(m.short_id)
    assert found is not None and found.id == m.id


@pytest.mark.asyncio
async def test_retrieve_filters_by_type(memory_store: SQLiteMemoryStore) -> None:
    await memory_store.save(content="F", type="fact")
    await memory_store.save(content="P", type="preference")
    prefs_only = await memory_store.retrieve(query="", types=["preference"], k=10)
    assert len(prefs_only) == 1
    assert prefs_only[0].type == "preference"


@pytest.mark.asyncio
async def test_supersede_missing_id_raises(memory_store: SQLiteMemoryStore) -> None:
    with pytest.raises(ValueError):
        await memory_store.supersede("nonexistent_id", "new content")


@pytest.mark.asyncio
async def test_retrieve_k_zero_returns_empty(memory_store: SQLiteMemoryStore) -> None:
    """Regression test for the falsy-zero bug: k=0 must return no memories."""
    await memory_store.save(content="fact 1", type="fact")
    await memory_store.save(content="fact 2", type="fact")
    results = await memory_store.retrieve(query="", k=0)
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_k_none_returns_all(memory_store: SQLiteMemoryStore) -> None:
    """k=None means 'no explicit limit' — falls back to config default (8 with no config)."""
    for i in range(3):
        await memory_store.save(content=f"fact {i}", type="fact")
    results = await memory_store.retrieve(query="", k=None)
    assert len(results) == 3
