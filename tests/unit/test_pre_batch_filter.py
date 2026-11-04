# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import contextlib
import datetime as dt
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from firecube.core.controlplane import ChunkManager
from firecube.core.index_spec import IndexSpec, ItemInfo, RegularTimeAxis
from firecube.ingestor.api import (
    DirectZarrIngestor,
    EngineConfig,
    PipelineBatch,
    PluginContext,
    RuntimeIngestContext,
)
from firecube.ingestor.errors import ConfigurationError
from firecube.ingestor.runtime import engine as engine_module
from firecube.ingestor.runtime.engine import (
    PipelineExecutor,
    _create_batches_with_parallel_filter,
)
from firecube.ingestor.types.context import OutputPaths, PipelineResult, PipelineRunState
from tests.helpers.resume_spans import record_run_with_span
from tests.helpers.storage import begin_write_session, make_test_binding

pytestmark = pytest.mark.unit


class _RecordingBatchPlanner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.return_value: list[PipelineBatch] = []

    def create_batches(
        self,
        host: Any,
        ctx: Any,
        batch_size: int,
        *,
        items: Any = None,
    ) -> Any:
        recorded_items = list(items) if items is not None else list(host.discover_source_files(ctx))
        self.calls.append(
            {"host": host, "ctx": ctx, "batch_size": batch_size, "items": recorded_items}
        )
        return iter(self.return_value)


def _make_runtime_ctx() -> RuntimeIngestContext:
    return RuntimeIngestContext(source="/tmp/source", target="/tmp/target")


class _MockCapableIngestor(DirectZarrIngestor):
    PRODUCT_NAME: ClassVar[str] = "mock_cap"

    def __init__(self, *, items: list[int], inspect_impl: Any = None) -> None:
        super().__init__(name="mock_cap")
        self._items = items
        self._inspect_impl = inspect_impl
        self.discover_calls = 0
        self.inspect_calls: list[Any] = []
        self._create_batches_calls = 0

    def index_spec(self, ctx: PluginContext) -> IndexSpec:
        return IndexSpec(
            name="pre_batch_filter_v1",
            groups={
                "data": RegularTimeAxis(
                    coordinate="timestamp",
                    epoch="2026-01-01T00:00:00Z",
                    cadence_s=1,
                    mode="exact",
                    slot_count=100,
                )
            },
        )

    def inspect_item(self, item: Any, ctx: PluginContext) -> ItemInfo | None:
        self.inspect_calls.append(item)
        if self._inspect_impl is not None:
            return self._inspect_impl(item, ctx)
        return ItemInfo(
            coordinate=dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=int(item))
        )

    def discover_source_files(self, ctx: PluginContext) -> Any:
        self.discover_calls += 1
        return list(self._items)

    def zarr_schema(self, ctx: PluginContext) -> list[Any]:
        return []

    def build_write_intents(self, batch: PipelineBatch, ctx: PluginContext) -> list[Any]:
        return []

    def ingest(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _create_batches(self, ctx: Any, batch_size: int) -> Any:
        self._create_batches_calls += 1
        return iter([])


class _MockNonCapableIngestor(DirectZarrIngestor):
    PRODUCT_NAME: ClassVar[str] = "mock_nc"

    def __init__(self) -> None:
        super().__init__(name="mock_nc")
        self.inspect_calls = 0
        self._create_batches_calls = 0

    def discover_source_files(self, ctx: PluginContext) -> Any:
        return list(range(50))

    def zarr_schema(self, ctx: PluginContext) -> list[Any]:
        return []

    def build_write_intents(self, batch: PipelineBatch, ctx: PluginContext) -> list[Any]:
        return []

    def ingest(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _create_batches(self, ctx: Any, batch_size: int) -> Any:
        self._create_batches_calls += 1
        return iter([])


def _log() -> logging.Logger:
    return logging.getLogger("firecube.test.pre_batch_filter")


def test_no_slot_flags_uses_standard_path() -> None:
    host = _MockCapableIngestor(items=list(range(50)))
    host._batch_planner = _RecordingBatchPlanner()  # type: ignore[assignment]
    ctx = _make_runtime_ctx()
    cfg = EngineConfig()

    result = _create_batches_with_parallel_filter(
        host=host, ctx=ctx, batch_size=10, engine_config=cfg, log=_log()
    )

    assert result == []
    assert host._create_batches_calls == 1
    assert host.inspect_calls == []
    assert host.discover_calls == 0
    assert host._batch_planner.calls == []  # type: ignore[attr-defined]


def test_parallel_mode_filters_items() -> None:
    host = _MockCapableIngestor(items=list(range(200)))
    planner = _RecordingBatchPlanner()
    host._batch_planner = planner  # type: ignore[assignment]
    ctx = _make_runtime_ctx()
    cfg = EngineConfig(slot_start=0, slot_end=100)
    host._bind_index_at_startup(PluginContext(ctx))
    begin_write_session(host, PluginContext(ctx), engine_config=cfg)

    result = _create_batches_with_parallel_filter(
        host=host, ctx=ctx, batch_size=10, engine_config=cfg, log=_log()
    )

    assert host._create_batches_calls == 0
    assert host.discover_calls == 1
    assert host.inspect_calls == list(range(200))

    assert len(planner.calls) == 1
    call = planner.calls[0]
    assert call["batch_size"] == 10
    assert call["items"] == list(range(100))
    assert result == []


def test_filter_returns_empty_gives_empty_batches() -> None:
    host = _MockCapableIngestor(
        items=list(range(200)),
        inspect_impl=lambda item, ctx: None,
    )
    planner = _RecordingBatchPlanner()
    host._batch_planner = planner  # type: ignore[assignment]
    ctx = _make_runtime_ctx()
    cfg = EngineConfig(slot_start=0, slot_end=100)
    host._bind_index_at_startup(PluginContext(ctx))

    result = _create_batches_with_parallel_filter(
        host=host, ctx=ctx, batch_size=10, engine_config=cfg, log=_log()
    )

    assert result == []
    assert planner.calls == []
    assert host._create_batches_calls == 0


def test_filter_raises_wrapped_as_config_error() -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("kaboom")

    host = _MockCapableIngestor(items=list(range(10)))
    host._batch_planner = _RecordingBatchPlanner()  # type: ignore[assignment]
    ctx = _make_runtime_ctx()
    cfg = EngineConfig(slot_start=0, slot_end=5)
    host._bind_index_at_startup(PluginContext(ctx))

    original = engine_module.filter_items_by_index
    engine_module.filter_items_by_index = _boom  # type: ignore[method-assign]
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            _create_batches_with_parallel_filter(
                host=host, ctx=ctx, batch_size=10, engine_config=cfg, log=_log()
            )
    finally:
        engine_module.filter_items_by_index = original  # type: ignore[method-assign]

    message = str(exc_info.value)
    assert "filter_items_by_index raised an error" in message
    assert "kaboom" in message
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_filter_returns_wrong_type_raises_typeerror() -> None:
    host = _MockCapableIngestor(items=list(range(10)))
    host._batch_planner = _RecordingBatchPlanner()  # type: ignore[assignment]
    ctx = _make_runtime_ctx()
    cfg = EngineConfig(slot_start=0, slot_end=5)
    host._bind_index_at_startup(PluginContext(ctx))

    original = engine_module.filter_items_by_index
    engine_module.filter_items_by_index = lambda *args, **kwargs: 42  # type: ignore[method-assign]

    try:
        with pytest.raises(TypeError, match="must return a Sequence"):
            _create_batches_with_parallel_filter(
                host=host, ctx=ctx, batch_size=10, engine_config=cfg, log=_log()
            )
    finally:
        engine_module.filter_items_by_index = original  # type: ignore[method-assign]


def test_non_capable_plugin_uses_standard_path() -> None:
    host = _MockNonCapableIngestor()
    host._batch_planner = _RecordingBatchPlanner()  # type: ignore[assignment]
    ctx = _make_runtime_ctx()
    cfg = EngineConfig()

    result = _create_batches_with_parallel_filter(
        host=host, ctx=ctx, batch_size=10, engine_config=cfg, log=_log()
    )

    assert result == []
    assert host._create_batches_calls == 1
    assert host.inspect_calls == 0
    assert host._batch_planner.calls == []  # type: ignore[attr-defined]


def test_non_direct_zarr_host_uses_standard_path() -> None:
    """A non-DirectZarrIngestor host should always take the standard path,
    even with slot flags set (defensive guard; capability gate enforces upstream)."""
    host = MagicMock()
    host._create_batches = MagicMock(return_value=iter([]))
    host.discover_source_files = MagicMock()
    ctx = _make_runtime_ctx()
    cfg = EngineConfig(slot_start=0, slot_end=100)

    result = _create_batches_with_parallel_filter(
        host=host, ctx=ctx, batch_size=10, engine_config=cfg, log=_log()
    )

    assert result == []
    host._create_batches.assert_called_once_with(ctx, 10)
    host.discover_source_files.assert_not_called()


# --- resume filter: guard conditions and serial / slot-range insertion -----------------

_RESUME_PRODUCT = "mock_cap"


@contextlib.contextmanager
def _written_manager(root: Path) -> Iterator[ChunkManager]:
    """A real control plane in ``root`` where a complete run wrote slots 0..9 of group ``data``.

    The host name is also the product and plugin name, which is what the
    engine falls back to without an output session. Item ``n`` of
    ``_MockCapableIngestor`` sits on slot ``n``.
    """
    root.mkdir(parents=True, exist_ok=True)
    manager = ChunkManager(binding=make_test_binding(root), workspace=root)
    try:
        record_run_with_span(
            manager,
            product=_RESUME_PRODUCT,
            plugin=_RESUME_PRODUCT,
            run_id="run-complete",
            ranges=[[0, 9]],
            terminal="complete",
        )
        yield manager
    finally:
        manager.close()


@pytest.fixture
def written_spans(tmp_path: Path) -> Iterator[ChunkManager]:
    with _written_manager(tmp_path) as manager:
        yield manager


def _resume_config(**overrides: Any) -> EngineConfig:
    options: dict[str, Any] = {"resume_existing": True, "write_mode": "direct"}
    options.update(overrides)
    return EngineConfig(**options)


def _resume_host(
    manager: ChunkManager,
    *,
    items: list[int],
    inspect_impl: Any = None,
    bind: bool = True,
) -> _MockCapableIngestor:
    host = _MockCapableIngestor(items=items, inspect_impl=inspect_impl)
    host._chunk_manager = manager
    if bind:
        host._bind_index_at_startup(PluginContext(_make_runtime_ctx()))
    return host


class _FilterItemIngestor(_MockCapableIngestor):
    """Refuses every item whose number is 3 mod 4 at ``filter_item``."""

    def filter_item(self, item: Any, ctx: PluginContext) -> bool:
        return int(item) % 4 != 3


def test_filter_guard_matrix(tmp_path: Path, subtests: pytest.Subtests) -> None:
    configs: dict[str, Any] = {
        "resume_existing_off": _resume_config(resume_existing=False),
        "force_reingest_on": _resume_config(force_reingest=True),
        "staged_write_mode": _resume_config(write_mode="staged"),
        "opted_out": _resume_config(resume_skip_written=False),
        "no_index_binding": _resume_config(),
    }
    for name, cfg in configs.items():
        with subtests.test(msg=name), _written_manager(tmp_path / name) as manager:
            host = _resume_host(manager, items=list(range(20)), bind=name != "no_index_binding")
            planner = _RecordingBatchPlanner()
            host._batch_planner = planner  # type: ignore[assignment]

            result = _create_batches_with_parallel_filter(
                host=host, ctx=_make_runtime_ctx(), batch_size=10, engine_config=cfg, log=_log()
            )

            assert result == []
            assert host.inspect_calls == []
            assert host._create_batches_calls == 1
            assert planner.calls == []
            assert host._timestamps_skipped_before_batching == 0

    with subtests.test(msg="non_direct_zarr_host"):
        mock_host = MagicMock()
        mock_host._create_batches = MagicMock(return_value=iter([]))
        mock_host.discover_source_files = MagicMock()
        mock_host.inspect_item = MagicMock()
        ctx = _make_runtime_ctx()

        result = _create_batches_with_parallel_filter(
            host=mock_host,
            ctx=ctx,
            batch_size=10,
            engine_config=_resume_config(),
            log=_log(),
        )

        assert result == []
        mock_host._create_batches.assert_called_once_with(ctx, 10)
        mock_host.inspect_item.assert_not_called()
        mock_host.discover_source_files.assert_not_called()


def test_serial_resume_filters_items_before_batching(written_spans: ChunkManager) -> None:
    host = _FilterItemIngestor(items=list(range(20)))
    host._chunk_manager = written_spans
    host._bind_index_at_startup(PluginContext(_make_runtime_ctx()))

    batches = _create_batches_with_parallel_filter(
        host=host,
        ctx=_make_runtime_ctx(),
        batch_size=3,
        engine_config=_resume_config(),
        log=_log(),
    )

    # Slots 0..9 are written: items 0..9 never reach a batch, inspected before any decode.
    # Of 10..19, filter_item then removes 11, 15 and 19.
    assert [item for batch in batches for item in batch.items] == [10, 12, 13, 14, 16, 17, 18]
    assert max(len(batch.items) for batch in batches) == 3
    assert host.inspect_calls == list(range(20))
    assert host._create_batches_calls == 0
    assert host._timestamps_skipped_before_batching == 10


def test_serial_resume_keeps_items_that_cannot_be_placed(written_spans: ChunkManager) -> None:
    def placeable_only_below_ten(item: Any, ctx: Any) -> ItemInfo | None:
        if item % 2:
            return None
        return ItemInfo(
            coordinate=dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=int(item))
        )

    host = _resume_host(written_spans, items=list(range(12)), inspect_impl=placeable_only_below_ten)
    planner = _RecordingBatchPlanner()
    host._batch_planner = planner  # type: ignore[assignment]

    _create_batches_with_parallel_filter(
        host=host,
        ctx=_make_runtime_ctx(),
        batch_size=10,
        engine_config=_resume_config(),
        log=_log(),
    )

    # Even items 0..8 sit on written slots and drop; odd items have no coordinate and stay, as
    # does the unwritten slot 10.
    assert planner.calls[0]["items"] == [1, 3, 5, 7, 9, 10, 11]
    assert host._timestamps_skipped_before_batching == 5


def test_serial_filter_disabled_on_inspect_not_implemented(written_spans: ChunkManager) -> None:
    def not_implemented(item: Any, ctx: Any) -> ItemInfo | None:
        raise NotImplementedError

    host = _resume_host(written_spans, items=list(range(20)), inspect_impl=not_implemented)
    planner = _RecordingBatchPlanner()
    host._batch_planner = planner  # type: ignore[assignment]

    result = _create_batches_with_parallel_filter(
        host=host,
        ctx=_make_runtime_ctx(),
        batch_size=10,
        engine_config=_resume_config(),
        log=_log(),
    )

    assert result == []
    assert host._create_batches_calls == 1
    assert planner.calls == []
    assert host.inspect_calls == [0]
    assert host._timestamps_skipped_before_batching == 0


def test_serial_resume_without_written_slots_never_inspects(tmp_path: Path) -> None:
    manager = ChunkManager(binding=make_test_binding(tmp_path), workspace=tmp_path)
    try:
        host = _resume_host(manager, items=list(range(20)))
        planner = _RecordingBatchPlanner()
        host._batch_planner = planner  # type: ignore[assignment]

        _create_batches_with_parallel_filter(
            host=host,
            ctx=_make_runtime_ctx(),
            batch_size=10,
            engine_config=_resume_config(),
            log=_log(),
        )

        # A fresh product has no coverage to compare against, so no item is positioned.
        assert host.inspect_calls == []
        assert planner.calls[0]["items"] == list(range(20))
        assert host._timestamps_skipped_before_batching == 0
    finally:
        manager.close()


def test_slot_range_filter_reuses_inspected_dict(written_spans: ChunkManager) -> None:
    host = _resume_host(written_spans, items=list(range(200)))
    planner = _RecordingBatchPlanner()
    host._batch_planner = planner  # type: ignore[assignment]
    ctx = _make_runtime_ctx()
    cfg = _resume_config(slot_start=0, slot_end=100)
    begin_write_session(host, PluginContext(ctx), engine_config=cfg)

    _create_batches_with_parallel_filter(
        host=host, ctx=ctx, batch_size=10, engine_config=cfg, log=_log()
    )

    # One inspection per discovered item, shared by the window filter and the resume filter.
    assert host.inspect_calls == list(range(200))
    assert planner.calls[0]["items"] == list(range(10, 100))
    assert host._timestamps_skipped_before_batching == 10


def test_dropped_count_surfaces_in_pipeline_summary(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    for stash, per_batch, expected in ((4, 3, 7), (0, 3, 3), (4, 0, 4)):
        with subtests.test(msg=f"stash_{stash}_per_batch_{per_batch}"):
            batch = PipelineBatch(
                batch_id="b1", data_path=Path("."), items=[], size_bytes=0, files_count=0
            )
            result = PipelineResult(
                batch=batch,
                outputs=OutputPaths(primary="s3://bucket/product"),
                success=True,
                metrics={"pipeline": {"timestamps_skipped": per_batch}},
            )
            state = PipelineRunState(
                product="dummy",
                pipeline_workers=1,
                batch_size=1,
                batches=(batch,),
                ingestion_start_time=0.0,
                batch_creation_duration=0.0,
                processing_start_time=0.0,
                results=(result,),
            )
            host = MagicMock(spec=DirectZarrIngestor)
            host._aggregate_metrics.return_value = {}
            host.name = "dummy"
            host._chunk_manager = None
            host._timestamps_skipped_before_batching = stash
            ctx = RuntimeIngestContext(source=str(tmp_path), target=str(tmp_path / "t.zarr"))

            finalized = PipelineExecutor().finalize(ctx, state, host)

            assert finalized.metrics.pipeline is not None
            assert finalized.metrics.pipeline.timestamps_skipped == expected


def _run_pipeline_over(
    manager: ChunkManager, items: list[int]
) -> tuple[Any, _RecordingBatchPlanner]:
    host = _resume_host(manager, items=items)
    host.engine_config = _resume_config(pipeline_workers=2)
    planner = _RecordingBatchPlanner()
    host._batch_planner = planner  # type: ignore[assignment]
    return PipelineExecutor().run_pipeline(_make_runtime_ctx(), host), planner


def test_all_items_skipped_yields_zero_batches_with_skip_count(
    written_spans: ChunkManager,
) -> None:
    # Items 0..9 are all written: no batch is planned and the run reports the skips.
    result, planner = _run_pipeline_over(written_spans, list(range(10)))

    assert planner.calls == []
    assert result.metrics.pipeline is not None
    assert result.metrics.pipeline.timestamps_skipped == 10


def test_nothing_skipped_leaves_empty_result_without_skip_metrics(
    written_spans: ChunkManager,
) -> None:
    # Items 20..29 are unwritten, so the (recording) planner is asked for them; it plans
    # none, which leaves an empty result that reports no skip.
    result, planner = _run_pipeline_over(written_spans, list(range(20, 30)))

    assert [call["items"] for call in planner.calls] == [list(range(20, 30))]
    assert result.metrics.pipeline is None
