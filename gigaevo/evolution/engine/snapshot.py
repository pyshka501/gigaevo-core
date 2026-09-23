"""Versioned snapshot of generic engine state, persisted to Redis.

Written by the base :class:`EvolutionEngine` via ``_write_snapshot`` and read
by any stage or external consumer that needs engine-aware behavior.

Last-writer-wins semantics: the engine is single-process async; exactly one
coroutine writes the snapshot. No CAS, no retries.

Sync + async access
-------------------
Some readers are sync (notably ``Stage.compute_hash``, a classmethod used for
cache-key computation). To avoid forcing them async, this module maintains a
process-wide ``_CURRENT_SNAPSHOT`` variable that is updated by every
``_write_snapshot`` call alongside the Redis write. Sync readers call
``get_current_snapshot()``; async readers (or out-of-process consumers)
call ``load_engine_snapshot(storage)``.
"""

from __future__ import annotations

import json
from typing import Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict

ENGINE_SNAPSHOT_KEY = "engine:snapshot"


class EngineSnapshot(BaseModel):
    total_mutants: int = 0
    next_iteration: int = 0
    programs_processed: int = 0
    completion_reason: str | None = None
    version: int = 0

    model_config = ConfigDict(frozen=True, extra="forbid")


class SnapshotUnavailableError(ValueError):
    """A saved engine snapshot cannot be used as a pause checkpoint input."""


class _SnapshotStorage(Protocol):
    async def load_run_state_str(self, field: str) -> str | None: ...
    async def save_run_state(self, field: str, value: int | str) -> None: ...


_CURRENT_SNAPSHOT: EngineSnapshot = EngineSnapshot()


def get_current_snapshot() -> EngineSnapshot:
    """Return the in-process snapshot mirror. Sync-safe."""
    return _CURRENT_SNAPSHOT


def set_current_snapshot(snap: EngineSnapshot) -> None:
    """Overwrite the in-process mirror. Called by ``EvolutionEngine._write_snapshot``
    and ``_load_snapshot_on_resume`` only — do not call from application code.
    """
    global _CURRENT_SNAPSHOT
    _CURRENT_SNAPSHOT = snap


def _reset_current_snapshot_for_tests() -> None:
    """Reset the module-level mirror to defaults. Use in test fixtures only."""
    global _CURRENT_SNAPSHOT
    _CURRENT_SNAPSHOT = EngineSnapshot()


async def load_engine_snapshot(storage: _SnapshotStorage) -> EngineSnapshot:
    """Load the snapshot from Redis, returning defaults if absent or corrupt."""
    raw = await storage.load_run_state_str(ENGINE_SNAPSHOT_KEY)
    if raw is None:
        return EngineSnapshot()
    try:
        return EngineSnapshot.model_validate_json(raw)
    except Exception as exc:
        logger.warning(
            "[EngineSnapshot] snapshot JSON corrupt ({}); returning defaults", exc
        )
        return EngineSnapshot()


async def load_required_engine_snapshot(storage: _SnapshotStorage) -> EngineSnapshot:
    """Load a complete, versioned engine snapshot without fallback defaults.

    This is one prerequisite for a future quiescent-pause checkpoint. It does
    not establish that work has drained, that strategy state is durable, or
    that Redis itself has been fsynced. Ordinary crash resume intentionally
    continues to use ``load_engine_snapshot`` and its legacy fallback.
    """
    raw = await storage.load_run_state_str(ENGINE_SNAPSHOT_KEY)
    if raw is None:
        raise SnapshotUnavailableError("engine snapshot is missing")

    def reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, value in pairs:
            if key in values:
                raise ValueError(f"duplicate field: {key}")
            values[key] = value
        return values

    try:
        data = json.loads(raw, object_pairs_hook=reject_duplicate_fields)
        if not isinstance(data, dict):
            raise ValueError("snapshot is not an object")
        if set(data) != set(EngineSnapshot.model_fields):
            raise ValueError("snapshot fields are incomplete or unexpected")
        snapshot = EngineSnapshot.model_validate(data, strict=True)
        if snapshot.version < 1:
            raise ValueError("snapshot has no persisted version")
        if (
            min(
                snapshot.total_mutants,
                snapshot.next_iteration,
                snapshot.programs_processed,
            )
            < 0
        ):
            raise ValueError("snapshot counters must be non-negative")
    except (TypeError, ValueError) as exc:
        raise SnapshotUnavailableError("engine snapshot is invalid") from exc
    return snapshot
