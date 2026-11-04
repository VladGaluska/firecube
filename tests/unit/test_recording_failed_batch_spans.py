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

"""``SpanRecorder.record_batch_failure`` writes one truthful span per group.

A failed append batch reports committed, failed and not-attempted groups in
its result metrics; the recorder turns them into an ``active`` span, a
``failed`` span carrying the touched ranges and the repair outcome, and a
``failed`` span with a ``not_attempted`` reason, under distinct keys. A plain
failure (no append outcome) keeps the old shape: one ``failed`` span per
batch group.

The ``written_to`` stamp of ``active`` spans (the durability marker the resume
filter reads) is covered at the end of the module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from firecube.core.controlplane import ChunkManager
from firecube.ingestor.runtime.recording import SpanRecorder
from firecube.ingestor.types.context import (
    IngestContext,
    IngestResult,
    OutputPaths,
    PipelineBatch,
    PipelineResult,
    PluginContext,
    RuntimeIngestContext,
)
from tests.helpers.storage import make_test_binding

pytestmark = pytest.mark.unit

_PRODUCT = "product.zarr"
_RUN_ID = "run-failed-batch"
_BATCH_ID = "batch_0001"


def _plugin_ctx() -> PluginContext:
    runtime_ctx = RuntimeIngestContext.from_ingest_context(
        IngestContext(source="."),
        run_id=_RUN_ID,
        temp_root=Path("."),
        materializer=lambda p: Path(p),
    )
    return PluginContext(runtime_ctx)


def _manager(tmp_path: Path) -> ChunkManager:
    manager = ChunkManager(
        binding=make_test_binding(tmp_path, product=_PRODUCT), workspace=tmp_path / "work"
    )
    manager.record_run_started(
        product=_PRODUCT,
        run_id=_RUN_ID,
        output_path=str(tmp_path / _PRODUCT),
        output_format="zarr",
        size=0,
        meta={"plugin": "test"},
    )
    return manager


def _finish_failed(manager: ChunkManager, tmp_path: Path) -> None:
    manager.record_run_terminal(
        product=_PRODUCT,
        run_id=_RUN_ID,
        output_path=str(tmp_path / _PRODUCT),
        output_format="zarr",
        size=0,
        meta={"plugin": "test"},
        status="failed",
        error="boom",
    )


def _spans_by_group(manager: ChunkManager) -> dict[str, dict]:
    """Return the projected span records of the run, keyed by group."""
    chunks = manager.list_chunks(product=_PRODUCT, chunk_type="span", include_replaced=True)
    spans: dict[str, dict] = {}
    for chunk in chunks:
        record = dict(chunk.record or {})
        record["key"] = chunk.key
        record["status"] = chunk.status
        record["meta"] = dict(chunk.meta or {})
        spans[record["meta"]["group"]] = record
    assert len(spans) == len(chunks)
    return spans


def _entry(group: str, ranges: list[list[int]], **extra: object) -> dict:
    entry: dict = {
        "group": group,
        "arrays": [f"{group}/value"],
        "time_index_ranges": ranges,
        "aligned": True,
        "state_array": f"{group}/firecube_timestamp_state",
        "state_deleted_value": 2,
        "time_min": "2024-01-01T00:00:00",
        "time_max": "2024-01-02T00:00:00",
        "time_dim_name": "timestamp",
    }
    entry.update(extra)
    return entry


def _batch(tmp_path: Path, groups: list[str]) -> PipelineBatch:
    return PipelineBatch(batch_id=_BATCH_ID, data_path=tmp_path, items=["x.nc"], groups=groups)


def test_failed_append_batch_records_committed_failed_and_not_attempted_spans(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    batch = _batch(tmp_path, ["A", "B", "C"])
    repair = {
        "state_marked_ranges": [[0, 1]],
        "truncated_to": None,
        "group_removed": False,
        "error": None,
    }
    result = PipelineResult(
        batch=batch,
        success=False,
        error="injected tail append failure",
        metrics={
            "zarr": {"batch_processing": {"batches_written": 1}},
            "coverage": [_entry("A", [[5, 6]])],
            "failed_coverage": [_entry("B", [[0, 1]], write_strategy="append_failed")],
            "failed_group": "B",
            "not_attempted_groups": ["C"],
            "repair": repair,
        },
    )

    SpanRecorder(manager).record_batch_failure(
        ctx=_plugin_ctx(),
        batch=batch,
        result=result,
        slice_meta={"plugin": "test"},
        run_id=_RUN_ID,
        product=_PRODUCT,
    )
    _finish_failed(manager, tmp_path)

    spans = _spans_by_group(manager)
    assert set(spans) == {"A", "B", "C"}
    assert {record["key"] for record in spans.values()} == {
        f"span_{_RUN_ID}_{_BATCH_ID}_A",
        f"span_{_RUN_ID}_{_BATCH_ID}_B",
        f"span_{_RUN_ID}_{_BATCH_ID}_C",
    }

    committed = spans["A"]
    assert committed["status"] == "active"
    assert committed["span"]["time_index_ranges"] == [[5, 6]]
    assert committed["span"]["timestamps_written"] == 2
    assert "repair" not in committed["meta"]

    failed = spans["B"]
    assert failed["status"] == "failed"
    assert failed["span"]["time_index_ranges"] == [[0, 1]]
    assert failed["span"]["write_strategy"] == "append_failed"
    assert failed["span"]["reason"] == "injected tail append failure"
    assert failed["meta"]["repair"] == repair
    assert failed["meta"]["time_min"] == "2024-01-01T00:00:00"

    not_attempted = spans["C"]
    assert not_attempted["status"] == "failed"
    assert not_attempted["span"]["time_index_ranges"] == []
    assert not_attempted["span"]["reason"] == (
        "not_attempted; group B failed: injected tail append failure"
    )
    assert "repair" not in not_attempted["meta"]

    active = manager.list_chunks(product=_PRODUCT, chunk_type="span", status="active")
    assert [chunk.meta["group"] for chunk in active if chunk.meta] == ["A"]


def test_failed_group_without_touched_slots_records_failed_span_without_ranges(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    batch = _batch(tmp_path, ["A"])
    result = PipelineResult(
        batch=batch,
        success=False,
        error="cannot decode input",
        metrics={
            "zarr": {},
            "coverage": [],
            "failed_coverage": [],
            "failed_group": "A",
            "not_attempted_groups": [],
            "repair": {
                "state_marked_ranges": [],
                "truncated_to": None,
                "group_removed": False,
                "error": None,
            },
        },
    )

    SpanRecorder(manager).record_batch_failure(
        ctx=_plugin_ctx(),
        batch=batch,
        result=result,
        slice_meta={"plugin": "test"},
        run_id=_RUN_ID,
        product=_PRODUCT,
    )
    _finish_failed(manager, tmp_path)

    spans = _spans_by_group(manager)
    assert set(spans) == {"A"}
    assert spans["A"]["status"] == "failed"
    assert spans["A"]["span"]["time_index_ranges"] == []
    assert spans["A"]["span"]["reason"] == "cannot decode input"
    assert spans["A"]["meta"]["repair"]["truncated_to"] is None
    assert manager.list_chunks(product=_PRODUCT, chunk_type="span", status="active") == []


def test_plain_failure_without_append_outcome_keeps_one_failed_span_per_group(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    batch = _batch(tmp_path, ["A", "B"])
    result = PipelineResult(batch=batch, success=False, error="boom")

    SpanRecorder(manager).record_batch_failure(
        ctx=_plugin_ctx(),
        batch=batch,
        result=result,
        slice_meta={"plugin": "test"},
        run_id=_RUN_ID,
        product=_PRODUCT,
    )
    _finish_failed(manager, tmp_path)

    spans = _spans_by_group(manager)
    assert set(spans) == {"A", "B"}
    for record in spans.values():
        assert record["status"] == "failed"
        assert record["span"]["reason"] == "boom"
        assert record["span"]["time_index_ranges"] == []
        assert "repair" not in record["meta"]


def _coverage_metrics(group: str, ranges: list[list[int]]) -> dict:
    return {"zarr": {"coverage": [_entry(group, ranges)]}}


def test_active_batch_spans_carry_the_recorders_written_to(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    rows = {
        "direct_target": ("target", {"plugin": "test"}, "target"),
        "staged_workspace": ("workspace", {"plugin": "test"}, "workspace"),
        "unset_recorder_stamps_nothing": (None, {"plugin": "test"}, None),
        "slice_meta_cannot_override_recorder": (
            "workspace",
            {"plugin": "test", "written_to": "target"},
            "workspace",
        ),
        "slice_meta_cannot_stamp_when_recorder_unset": (
            None,
            {"plugin": "test", "written_to": "target"},
            None,
        ),
    }
    for name, (written_to, slice_meta, expected) in rows.items():
        with subtests.test(msg=name):
            case_dir = tmp_path / name
            manager = _manager(case_dir)
            batch = _batch(case_dir, ["A"])
            recorder = SpanRecorder(manager)
            recorder.written_to = written_to

            recorder.record_batch_success(
                ctx=_plugin_ctx(),
                batch=batch,
                result=PipelineResult(
                    batch=batch, success=True, metrics=_coverage_metrics("A", [[0, 3]])
                ),
                slice_meta=slice_meta,
                run_id=_RUN_ID,
                product=_PRODUCT,
            )
            _finish_failed(manager, case_dir)

            span = _spans_by_group(manager)["A"]
            assert span["status"] == "active"
            assert span["meta"].get("written_to") == expected


def test_failed_and_skipped_batch_spans_are_not_stamped(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    recorder = SpanRecorder(manager)
    recorder.written_to = "target"
    skipped_batch = _batch(tmp_path, ["A"])
    failed_batch = PipelineBatch(
        batch_id="batch_0002", data_path=tmp_path, items=["y.nc"], groups=["B"]
    )

    recorder.record_batch_success(
        ctx=_plugin_ctx(),
        batch=skipped_batch,
        result=PipelineResult(batch=skipped_batch, success=True, metrics={}),
        slice_meta={"plugin": "test"},
        run_id=_RUN_ID,
        product=_PRODUCT,
    )
    recorder.record_batch_failure(
        ctx=_plugin_ctx(),
        batch=failed_batch,
        result=PipelineResult(batch=failed_batch, success=False, error="boom"),
        slice_meta={"plugin": "test"},
        run_id=_RUN_ID,
        product=_PRODUCT,
    )
    _finish_failed(manager, tmp_path)

    spans = _spans_by_group(manager)
    assert spans["A"]["status"] == "skipped"
    assert spans["B"]["status"] == "failed"
    assert "written_to" not in spans["A"]["meta"]
    assert "written_to" not in spans["B"]["meta"]


def test_terminal_run_spans_ignore_a_written_to_in_slice_meta(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    rows = {"recorder_wins": ("workspace", "workspace"), "unset_recorder_removes": (None, None)}
    for name, (written_to, expected) in rows.items():
        with subtests.test(msg=name):
            case_dir = tmp_path / name
            manager = _manager(case_dir)
            recorder = SpanRecorder(manager)
            recorder.written_to = written_to
            ctx = RuntimeIngestContext.from_ingest_context(
                IngestContext(source="."),
                run_id=_RUN_ID,
                temp_root=Path("."),
                materializer=lambda p: Path(p),
            )
            result = IngestResult(
                output_format="zarr",
                outputs=OutputPaths(primary=str(case_dir / _PRODUCT)),
                metrics=_coverage_metrics("A", [[0, 3]]),
            )

            recorder.register_run(
                ctx=ctx,
                result=result,
                run_id=_RUN_ID,
                product=_PRODUCT,
                slice_meta={"plugin": "test", "written_to": "target"},
            )

            assert _spans_by_group(manager)["A"]["meta"].get("written_to") == expected


def test_terminal_run_spans_carry_written_to_but_run_meta_does_not(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    recorder = SpanRecorder(manager)
    recorder.written_to = "target"
    ctx = RuntimeIngestContext.from_ingest_context(
        IngestContext(source="."),
        run_id=_RUN_ID,
        temp_root=Path("."),
        materializer=lambda p: Path(p),
    )
    result = IngestResult(
        output_format="zarr",
        outputs=OutputPaths(primary=str(tmp_path / _PRODUCT)),
        metrics=_coverage_metrics("A", [[0, 3]]),
    )

    recorder.register_run(
        ctx=ctx,
        result=result,
        run_id=_RUN_ID,
        product=_PRODUCT,
        slice_meta={"plugin": "test"},
    )

    span = _spans_by_group(manager)["A"]
    assert span["key"] == f"span_{_RUN_ID}_single_A"
    assert span["meta"]["written_to"] == "target"
    (run,) = manager.list_chunks(product=_PRODUCT, chunk_type="run", include_replaced=True)
    assert run.status == "complete"
    assert "written_to" not in (run.meta or {})
