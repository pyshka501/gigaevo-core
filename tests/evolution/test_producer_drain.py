"""Local producer drain is not a durable pause/checkpoint protocol."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.programs.program_state import ProgramState


def _make_engine(*, drain_timeout_s: float = 1.0) -> SteadyStateEvolutionEngine:
    storage = AsyncMock()
    storage.count_by_status.return_value = 0
    storage.snapshot = MagicMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    return SteadyStateEvolutionEngine(
        storage=storage,
        strategy=AsyncMock(),
        mutation_operator=AsyncMock(),
        config=SteadyStateEngineConfig(
            max_in_flight=10,
            loop_interval=0.001,
            post_cap_drain_grace_s=0.01,
            terminal_drain_timeout_s=drain_timeout_s,
        ),
        writer=writer,
        metrics_tracker=MagicMock(),
    )


@pytest.mark.asyncio
async def test_request_drains_ten_producers_then_registered_and_storage_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_engine()
    producers_started = asyncio.Event()
    release_producers = asyncio.Event()
    release_children = asyncio.Event()
    release_storage = asyncio.Event()
    started: list[int] = []
    cancelled: list[int] = []

    async def fake_mutant(active_engine: SteadyStateEvolutionEngine, task_id: int):
        try:
            started.append(task_id)
            if len(started) == 10:
                producers_started.set()
            await release_producers.wait()
            async with active_engine._in_flight_lock:
                active_engine._in_flight.add(f"child-{task_id}")
            return f"child-{task_id}"
        except asyncio.CancelledError:
            cancelled.append(task_id)
            raise
        finally:
            active_engine._producer_sema.release()

    async def fake_ingestor(active_engine: SteadyStateEvolutionEngine) -> None:
        await release_children.wait()
        async with active_engine._in_flight_lock:
            active_engine._in_flight.clear()
        while active_engine._running:
            await asyncio.sleep(0)

    async def fake_sampler(_active_engine: SteadyStateEvolutionEngine) -> None:
        await asyncio.Event().wait()

    async def count_by_status(status: str) -> int:
        return int(
            status == ProgramState.RUNNING.value and not release_storage.is_set()
        )

    engine.storage.count_by_status.side_effect = count_by_status
    monkeypatch.setattr(
        "gigaevo.evolution.engine.dispatcher.run_one_mutant", fake_mutant
    )
    monkeypatch.setattr(
        "gigaevo.evolution.engine.steady_state.ingestor_loop", fake_ingestor
    )
    monkeypatch.setattr(
        "gigaevo.evolution.engine.steady_state.backpressure_sampler_loop",
        fake_sampler,
    )
    engine._await_idle = AsyncMock()
    engine._ingest_completed_programs = AsyncMock()
    engine._reconcile_memory_attempts = AsyncMock()
    engine._write_snapshot = AsyncMock()
    engine._final_ingestion_sweep = AsyncMock()

    engine._task = asyncio.create_task(engine.run())
    try:
        await asyncio.wait_for(producers_started.wait(), timeout=1.0)
        engine.request_producer_drain()
        waiter = asyncio.create_task(engine.wait_for_producer_drain(timeout_s=2.0))
        assert len(started) == 10
        assert not waiter.done()

        release_producers.set()
        assert engine._dispatcher_task is not None
        await asyncio.wait_for(engine._dispatcher_task, timeout=1.0)
        assert len(started) == 10  # no eleventh dispatch after request
        assert not cancelled
        assert not waiter.done()  # children are still in flight

        release_children.set()
        await asyncio.sleep(0)
        assert not waiter.done()  # RUNNING storage record is still present
        assert not engine._producer_drain_completed

        release_storage.set()
        await asyncio.wait_for(waiter, timeout=2.0)
        assert engine._producer_drain_completed
        assert len(started) == 10
    finally:
        release_producers.set()
        release_children.set()
        release_storage.set()
        if engine._task is not None and not engine._task.done():
            engine._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await engine._task


@pytest.mark.asyncio
async def test_requested_drain_ignores_short_cap_grace_and_times_out() -> None:
    engine = _make_engine(drain_timeout_s=0.05)
    engine._running = True
    engine.request_producer_drain()
    engine._in_flight.add("blocked-child")

    with pytest.raises(TimeoutError, match="terminal drain timed out"):
        await engine._await_terminal_drain(strict=True)

    assert not engine._producer_drain_completed
    assert engine._in_flight == {"blocked-child"}


@pytest.mark.asyncio
async def test_unregistered_running_storage_record_fails_closed() -> None:
    engine = _make_engine(drain_timeout_s=0.05)
    engine._running = True
    engine.request_producer_drain()
    engine.storage.count_by_status.side_effect = lambda state: int(
        state == ProgramState.RUNNING.value
    )

    with pytest.raises(TimeoutError, match="queued=0, running=1"):
        await engine._await_terminal_drain(strict=True)

    assert not engine._producer_drain_completed


@pytest.mark.asyncio
async def test_waiter_timeout_and_cancel_do_not_cancel_drain() -> None:
    engine = _make_engine()
    engine._running = True
    finish = asyncio.Event()
    engine._task = asyncio.create_task(finish.wait())
    engine.request_producer_drain()
    try:
        with pytest.raises(TimeoutError):
            await engine.wait_for_producer_drain(timeout_s=0.01)
        assert not engine._task.done()
        assert not engine._producer_drain_completed

        waiter = asyncio.create_task(engine.wait_for_producer_drain(timeout_s=1.0))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not engine._task.done()
        assert engine._producer_drain_requested
    finally:
        finish.set()
        await engine._task

    with pytest.raises(RuntimeError, match="construct a new engine"):
        engine.start()
