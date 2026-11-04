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

import dataclasses
import importlib
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from click.testing import CliRunner, Result

from firecube.cli.main import cli
from firecube.core.controlplane import ChunkManager
from firecube.ingestor.registry import loader as _loader
from tests.helpers.storage import make_test_binding

pytestmark = pytest.mark.integration

_PLUGIN = "direct_zarr_capable_test_plugin"
_PRODUCT = "direct_zarr_capable_test_product"
_ITEMS = 200
_BATCH = 20


@pytest.fixture(autouse=True)
def reset_plugin_registry() -> Iterator[None]:
    original_loaded = _loader._LOADED
    original_registry = dict(_loader.AVAILABLE_INGESTORS)
    _loader._LOADED = False
    _loader.AVAILABLE_INGESTORS.clear()
    importlib.reload(importlib.import_module("direct_zarr_capable_test_plugin"))
    importlib.reload(importlib.import_module("multi_group_capable_test_plugin"))
    _loader._LOADED = True
    yield
    _loader._LOADED = original_loaded
    _loader.AVAILABLE_INGESTORS.clear()
    _loader.AVAILABLE_INGESTORS.update(original_registry)


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(record.getMessage() for record in self.records)


@pytest.fixture
def engine_log() -> Iterator[_Records]:
    """Records of the ``firecube.ingestor`` loggers, which the resume filter logs to.

    A handler on the logger itself: the CLI resets the root logger's handlers, which
    would drop pytest's capture handler.
    """
    handler = _Records()
    logger = logging.getLogger("firecube.ingestor")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


@pytest.fixture
def decode_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "decoded.log"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DIRECT_ZARR_CAPABLE_DECODE_LOG", str(path))
    return path


def _take_decoded(log: Path) -> list[int]:
    """Return the items decoded since the last call, sorted, and reset the log."""
    items = sorted(int(line) for line in log.read_text(encoding="utf-8").split())
    log.write_text("", encoding="utf-8")
    return items


def _ingest(
    plugin: str,
    target: Path,
    *options: str,
    write_mode: str = "direct",
    resume: bool = False,
    flags: tuple[str, ...] = (),
) -> Result:
    args = [
        "ingest",
        plugin,
        "--target",
        target.as_uri(),
        "--product-name",
        _PRODUCT if plugin == _PLUGIN else "multi_group_capable_test_product",
        "--storage-type",
        "local",
        "--storage-driver",
        "fsspec",
        "--write-mode",
        write_mode,
        "--option",
        "no_progress=true",
        "--option",
        f"pipeline_batch_size={_BATCH}",
    ]
    if resume:
        args += ["--option", "resume_existing=true"]
    for option in options:
        args += ["--option", option]
    args += list(flags)
    return CliRunner().invoke(cli, args)


def _runs(target: Path) -> list[dict[str, Any]]:
    result = CliRunner().invoke(
        cli, ["chunks", "runs", "list", "--product-name", target.as_uri(), "-f", "json"]
    )
    assert result.exit_code == 0, result.output
    runs = json.loads(result.output)
    return sorted(runs, key=lambda run: run["started_at"])


def _current_spans(target: Path) -> list[Any]:
    """The spans the default projection shows: replaced ones are gone."""
    manager = ChunkManager(
        binding=make_test_binding(target.parent), workspace=target.parent / "workspace"
    )
    try:
        return manager.list_chunks(product=target.name, chunk_type="span")
    finally:
        manager.close()


def _spans(target: Path) -> list[Any]:
    manager = ChunkManager(
        binding=make_test_binding(target.parent), workspace=target.parent / "workspace"
    )
    try:
        return manager.list_chunks(
            product=target.name, chunk_type="span", status="active", include_replaced=True
        )
    finally:
        manager.close()


def _open_array(target: Path, group: str, name: str, mode: str = "r") -> zarr.Array:
    return zarr.open_array(store=str(target), path=f"{group}/{name}", mode=mode, zarr_format=3)


def _array(target: Path, group: str = "data", name: str = "data") -> np.ndarray:
    return np.asarray(_open_array(target, group, name)[:])


def _expected_rows(first: int, last: int, width: int = 10) -> np.ndarray:
    return np.repeat(np.arange(first, last, dtype=np.float32)[:, None], width, axis=1)


def _fail_batches_containing(
    monkeypatch: pytest.MonkeyPatch, plugin_class: Any, bad_items: set[Any]
) -> None:
    original = plugin_class.build_write_intents

    def failing(self: Any, batch: Any, ctx: Any) -> Any:
        if any(item in bad_items for item in batch.items):
            raise RuntimeError("injected decode failure")
        return original(self, batch, ctx)

    monkeypatch.setattr(plugin_class, "build_write_intents", failing)


_UNWRITTEN = set(range(100, 120))
_WRITTEN = set(range(_ITEMS)) - _UNWRITTEN


def _crash_at_item_100(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a direct ingest in which the batch of items 100..119 fails.

    The engine attempts every batch, so the other nine are written and recorded
    and the run ends failed.
    """
    # Imported inside the test: reset_plugin_registry reloads this module per test.
    import direct_zarr_capable_test_plugin as plugin_module

    with monkeypatch.context() as patched:
        _fail_batches_containing(
            patched, plugin_module.DirectZarrCapableTestIngestor, set(_UNWRITTEN)
        )
        result = _ingest(_PLUGIN, target)
    assert result.exit_code != 0, result.output
    assert "injected decode failure" in result.output


def _covered_slots(spans: list[Any], group: str = "data") -> set[int]:
    covered: set[int] = set()
    for span in spans:
        if (span.meta or {}).get("group") != group:
            continue
        for start, end in span.record["span"]["time_index_ranges"]:
            covered.update(range(start, end + 1))
    return covered


# --- the skip ---------------------------------------------------------------------------


def test_direct_resume_after_complete_run_skips_all_decodes(
    tmp_path: Path, decode_log: Path, engine_log: _Records
) -> None:
    target = tmp_path / "out.zarr"
    first = _ingest(_PLUGIN, target)
    assert first.exit_code == 0, first.output
    assert _take_decoded(decode_log) == list(range(_ITEMS))
    before = _array(target)
    assert np.array_equal(before[:_ITEMS], _expected_rows(0, _ITEMS))

    resumed = _ingest(_PLUGIN, target, resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == []
    assert (
        "stage=resume_written_filter group=data required_groups=['data'] "
        "original_count=200 filtered_count=0 dropped_count=200"
    ) in engine_log.text
    after = _array(target)
    assert np.array_equal(after, before)
    assert np.array_equal(after[:_ITEMS], _expected_rows(0, _ITEMS))
    first_run, second_run = _runs(target)
    assert (first_run["status"], first_run["timestamps_skipped"]) == ("complete", 0)
    assert (second_run["status"], second_run["timestamps_skipped"]) == ("complete", _ITEMS)


def test_direct_resume_with_parallel_pipeline_workers_skips_all_decodes(
    tmp_path: Path, decode_log: Path
) -> None:
    target = tmp_path / "out.zarr"
    assert _ingest(_PLUGIN, target).exit_code == 0
    _take_decoded(decode_log)

    resumed = _ingest(_PLUGIN, target, "pipeline_workers=2", resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == []
    assert _runs(target)[-1]["timestamps_skipped"] == _ITEMS


def test_direct_resume_after_crash_skips_written_slots(
    tmp_path: Path, decode_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.zarr"
    _crash_at_item_100(target, monkeypatch)
    assert _take_decoded(decode_log) == sorted(_WRITTEN)

    # The invariant behind trusting a crashed run's span: a span is recorded only
    # after its batch's writes are durable, so every covered slot already holds
    # its value, and the slots of the failed batch do not.
    crashed_rows = _array(target)
    spans = _spans(target)
    assert _covered_slots(spans) == _WRITTEN
    assert {(span.meta or {}).get("written_to") for span in spans} == {"target"}
    for slot in sorted(_WRITTEN):
        assert (crashed_rows[slot] == float(slot)).all(), slot
    assert not crashed_rows[100:120].any()
    assert _runs(target)[0]["status"] == "failed"

    resumed = _ingest(_PLUGIN, target, resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == sorted(_UNWRITTEN)
    assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))
    resumed_run = _runs(target)[-1]
    assert (resumed_run["status"], resumed_run["timestamps_skipped"]) == (
        "complete",
        len(_WRITTEN),
    )


def test_resume_skip_written_false_decodes_every_item(tmp_path: Path, decode_log: Path) -> None:
    target = tmp_path / "out.zarr"
    assert _ingest(_PLUGIN, target).exit_code == 0
    _take_decoded(decode_log)

    resumed = _ingest(_PLUGIN, target, "resume_skip_written=false", resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == list(range(_ITEMS))
    assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))
    assert _runs(target)[-1]["timestamps_skipped"] == 0


# --- the bypasses -----------------------------------------------------------------------


def test_force_reingest_decodes_everything(tmp_path: Path, decode_log: Path) -> None:
    target = tmp_path / "out.zarr"
    assert _ingest(_PLUGIN, target).exit_code == 0
    _take_decoded(decode_log)

    forced = _ingest(_PLUGIN, target, "force_reingest=true", resume=True)

    assert forced.exit_code == 0, forced.output
    assert _take_decoded(decode_log) == list(range(_ITEMS))
    assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))
    first_run, forced_run = _runs(target)
    assert forced_run["timestamps_skipped"] == 0
    # The forced run replaced the first run's spans: only its own remain active.
    active = _current_spans(target)
    assert {(span.meta or {}).get("run_id") for span in active} == {forced_run["run_id"]}
    assert first_run["run_id"] != forced_run["run_id"]
    assert _covered_slots(active) == set(range(_ITEMS))


def test_staged_resume_never_filters(
    tmp_path: Path, decode_log: Path, engine_log: _Records
) -> None:
    target = tmp_path / "out.zarr"
    assert _ingest(_PLUGIN, target, write_mode="staged").exit_code == 0
    assert _take_decoded(decode_log) == list(range(_ITEMS))

    resumed = _ingest(_PLUGIN, target, write_mode="staged", resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == list(range(_ITEMS))
    assert "resume_written_filter" not in engine_log.text
    assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))
    assert [run["timestamps_skipped"] for run in _runs(target)] == [0, 0]


def test_unstamped_crashed_run_is_fully_redecoded(
    tmp_path: Path, decode_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.zarr"
    _crash_at_item_100(target, monkeypatch)
    _take_decoded(decode_log)

    # Rewrite the crashed run's WAL as a release without the stamp wrote it.
    stripped = 0
    for events_file in (target / ".firecube" / "runs").glob("*/events-*.jsonl"):
        lines = []
        for line in events_file.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            for holder in (event.get("meta"), event.get("record", {}).get("meta")):
                if isinstance(holder, dict) and holder.pop("written_to", None) is not None:
                    stripped += 1
            lines.append(json.dumps(event, separators=(",", ":")))
        events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert stripped > 0
    assert {(span.meta or {}).get("written_to") for span in _spans(target)} == {None}

    resumed = _ingest(_PLUGIN, target, resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == list(range(_ITEMS))
    assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))
    assert _runs(target)[-1]["timestamps_skipped"] == 0


# --- statics, groups, surfaces ----------------------------------------------------------


def test_partial_skip_still_replays_statics(
    tmp_path: Path, decode_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Imported inside the test: reset_plugin_registry reloads this module per test.
    import direct_zarr_capable_test_plugin as plugin_module

    plugin_class = plugin_module.DirectZarrCapableTestIngestor
    target = tmp_path / "out.zarr"
    _crash_at_item_100(target, monkeypatch)
    _take_decoded(decode_log)
    assert _open_array(target, "data", "lat").attrs["firecube_static_written"]

    # A resumed batch that emits a static diverging from the stored one is refused,
    # so the remaining batches really do replay the static intent.
    original = plugin_class.build_write_intents

    def diverging_static(self: Any, batch: Any, ctx: Any) -> Any:
        return [
            dataclasses.replace(intent, data=np.asarray(intent.data) + 1.0)
            if intent.kind == "static"
            else intent
            for intent in original(self, batch, ctx)
        ]

    with monkeypatch.context() as patched:
        patched.setattr(plugin_class, "build_write_intents", diverging_static)
        diverged = _ingest(_PLUGIN, target, resume=True)
    assert diverged.exit_code != 0, diverged.output
    assert "lat" in diverged.output
    assert _take_decoded(decode_log) == sorted(_UNWRITTEN)

    resumed = _ingest(_PLUGIN, target, resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == sorted(_UNWRITTEN)
    assert np.array_equal(_array(target, name="lat"), np.arange(10, dtype=np.float64))
    assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))


def test_multi_group_item_kept_unless_all_groups_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Imported inside the test: reset_plugin_registry reloads this module per test.
    import multi_group_capable_test_plugin as plugin_module

    plugin = "multi_group_capable_test_plugin"
    target = tmp_path / "out.zarr"
    # Items run group_a 0..199 then group_b 0..199. The engine attempts every batch:
    # group_a is fully written, group_b everywhere except slots 40..59, whose batch fails.
    with monkeypatch.context() as patched:
        _fail_batches_containing(
            patched,
            plugin_module.MultiGroupCapableTestIngestor,
            {("group_b", slot) for slot in range(40, 60)},
        )
        crashed = _ingest(plugin, target)
    assert crashed.exit_code != 0, crashed.output
    spans = _spans(target)
    assert _covered_slots(spans, "group_a") == set(range(200))
    assert _covered_slots(spans, "group_b") == set(range(200)) - set(range(40, 60))

    # Mark rows with a sentinel the plugin never writes: a skipped item leaves its
    # sentinel, a decoded one overwrites it with the slot number.
    _open_array(target, "group_a", "primary", "r+")[10, :] = -1.0
    _open_array(target, "group_b", "primary", "r+")[10, :] = -1.0
    _open_array(target, "group_a", "primary", "r+")[45, :] = -1.0

    resumed = _ingest(plugin, target, resume=True)

    assert resumed.exit_code == 0, resumed.output
    group_a = _array(target, "group_a", "primary")
    group_b = _array(target, "group_b", "primary")
    # Slot 10 is written in both groups: both of its items are skipped.
    assert (group_a[10] == -1.0).all()
    assert (group_b[10] == -1.0).all()
    # Slot 45 is written in group_a only: its group_a item is decoded, not skipped.
    assert (group_a[45] == 45.0).all()
    assert (group_b[45] == 45.0).all()
    assert (group_b[40:60] == _expected_rows(40, 60, width=5)).all()
    # 180 slots are covered in both groups, two items each.
    assert _runs(target)[-1]["timestamps_skipped"] == 360


# --- maintenance runs, groups, inspect errors -------------------------------------------


def _record_maintenance(target: Path, op: str) -> None:
    manager = ChunkManager(
        binding=make_test_binding(target.parent), workspace=target.parent / "workspace"
    )
    try:
        run_id = f"maintenance-{op}-001"
        manager.record_maintenance_started(product=target.name, run_id=run_id, op=op)
        manager.record_maintenance_completed(product=target.name, run_id=run_id, op=op)
    finally:
        manager.close()


def test_data_changing_maintenance_run_revokes_trust_in_crashed_run_spans(
    tmp_path: Path, decode_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.zarr"
    _crash_at_item_100(target, monkeypatch)
    _take_decoded(decode_log)
    assert _covered_slots(_spans(target)) == _WRITTEN
    # A delete may have emptied slots the crashed run's spans still claim.
    _record_maintenance(target, "delete")

    resumed = _ingest(_PLUGIN, target, resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == list(range(_ITEMS))
    assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))
    assert _runs(target)[-1]["timestamps_skipped"] == 0


def test_archive_create_maintenance_run_keeps_trust_in_crashed_run_spans(
    tmp_path: Path, decode_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.zarr"
    _crash_at_item_100(target, monkeypatch)
    _take_decoded(decode_log)
    _record_maintenance(target, "archive_create")

    resumed = _ingest(_PLUGIN, target, resume=True)

    assert resumed.exit_code == 0, resumed.output
    assert _take_decoded(decode_log) == sorted(_UNWRITTEN)
    assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))
    assert _runs(target)[-1]["timestamps_skipped"] == len(_WRITTEN)


def test_serial_resume_with_slot_group_but_no_window_decodes_every_uncovered_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Imported inside the test: reset_plugin_registry reloads this module per test.
    import multi_group_capable_test_plugin as plugin_module

    plugin = "multi_group_capable_test_plugin"
    target = tmp_path / "out.zarr"
    with monkeypatch.context() as patched:
        _fail_batches_containing(
            patched,
            plugin_module.MultiGroupCapableTestIngestor,
            {("group_b", slot) for slot in range(40, 60)},
        )
        assert _ingest(plugin, target).exit_code != 0
    # Slot 45 is written in group_a only; the sentinel shows whether its item is decoded.
    _open_array(target, "group_a", "primary", "r+")[45, :] = -1.0
    _open_array(target, "group_a", "primary", "r+")[10, :] = -1.0

    resumed = _ingest(plugin, target, resume=True, flags=("--slot-group", "group_a"))

    assert resumed.exit_code == 0, resumed.output
    group_a = _array(target, "group_a", "primary")
    # A serial run writes every group, so naming one group must not narrow what counts as
    # written: the group_a item of slot 45 is decoded because group_b lacks that slot.
    assert (group_a[45] == 45.0).all()
    assert (_array(target, "group_b", "primary")[40:60] == _expected_rows(40, 60, 5)).all()
    # Slot 10 is written in both groups: skipped, sentinel intact.
    assert (group_a[10] == -1.0).all()
    assert _runs(target)[-1]["timestamps_skipped"] == 360


def test_serial_resume_keeps_items_whose_inspect_item_raises(
    tmp_path: Path,
    decode_log: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine_log: _Records,
    subtests: pytest.Subtests,
) -> None:
    # Imported inside the test: reset_plugin_registry reloads this module per test.
    import direct_zarr_capable_test_plugin as plugin_module

    plugin_class = plugin_module.DirectZarrCapableTestIngestor
    original = plugin_class.inspect_item
    unreadable = set(range(10, 20))
    # ValueError is one of the index's own placement errors; RuntimeError and OSError are
    # the plugin's: the filter keeps the item for every one of them.
    for error_type in (ValueError, RuntimeError, OSError):
        with subtests.test(msg=error_type.__name__):
            target = tmp_path / f"{error_type.__name__}.zarr"
            assert _ingest(_PLUGIN, target).exit_code == 0
            _take_decoded(decode_log)
            engine_log.records.clear()
            seen: set[int] = set()

            def flaky_inspect(
                self: Any,
                item: Any,
                ctx: Any,
                error_type: type[Exception] = error_type,
                seen: set[int] = seen,
            ) -> Any:
                # Fails the engine's pre-decode look at an item, then behaves: the plugin's
                # own decode of a kept item must still work.
                if item in unreadable and item not in seen:
                    seen.add(item)
                    raise error_type(f"cannot read header of item {item}")
                return original(self, item, ctx)

            with monkeypatch.context() as patched:
                patched.setattr(plugin_class, "inspect_item", flaky_inspect)
                resumed = _ingest(_PLUGIN, target, resume=True)

            assert resumed.exit_code == 0, resumed.output
            assert _take_decoded(decode_log) == sorted(unreadable)
            assert np.array_equal(_array(target)[:_ITEMS], _expected_rows(0, _ITEMS))
            run = _runs(target)[-1]
            assert (run["status"], run["timestamps_skipped"]) == (
                "complete",
                _ITEMS - len(unreadable),
            )
            warnings = [
                record.getMessage()
                for record in engine_log.records
                if record.levelno == logging.WARNING
                and "could not be placed" in record.getMessage()
            ]
            assert len(warnings) == 1
            assert "10 item(s) kept" in warnings[0]
            assert f"{error_type.__name__}('cannot read header of item 10')" in warnings[0]


def test_real_delete_span_records_the_delete_maintenance_run_the_filter_keys_on(
    tmp_path: Path, decode_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Imported inside the test: reset_plugin_registry reloads this module per test.
    import direct_zarr_capable_test_plugin as plugin_module

    target = tmp_path / "out.zarr"
    assert _ingest(_PLUGIN, target).exit_code == 0
    first_run_id = _runs(target)[0]["run_id"]
    # A forced rewrite crashes on the batch of items 100..119: its target-stamped spans
    # claim every other slot, which the first run's delete-span would empty.
    with monkeypatch.context() as patched:
        _fail_batches_containing(
            patched, plugin_module.DirectZarrCapableTestIngestor, set(_UNWRITTEN)
        )
        assert _ingest(_PLUGIN, target, "force_reingest=true").exit_code != 0

    # The command runs the real deletion engine. Its exit code is not asserted: this
    # fixture's spans list the bare array name "data" for the array "data/data", which
    # the engine currently cannot resolve, so nothing is deleted. The record is what the
    # resume filter reads: a maintenance run of op "delete", written before the engine
    # touches any cell. (A crashed run's own spans are not selectable by delete-span, so
    # a delete of an earlier complete run is how its claimed slots get emptied.)
    CliRunner().invoke(
        cli,
        [
            "chunks",
            "delete-span",
            "--product-name",
            target.as_uri(),
            "--run-id",
            first_run_id,
            "--force",
            "--yes-i-really-mean-it",
        ],
    )

    manager = ChunkManager(
        binding=make_test_binding(target.parent), workspace=target.parent / "workspace"
    )
    try:
        maintenance = manager.list_chunks(
            product=target.name,
            chunk_type="run",
            include_replaced=True,
            meta={"kind": "maintenance"},
        )
    finally:
        manager.close()
    assert [(run.meta or {}).get("op") for run in maintenance] == ["delete"]


def test_skip_count_surfaces_in_runs_list(tmp_path: Path, decode_log: Path) -> None:
    target = tmp_path / "out.zarr"
    assert _ingest(_PLUGIN, target).exit_code == 0

    resumed = _ingest(_PLUGIN, target, resume=True)

    assert resumed.exit_code == 0, resumed.output
    skipped = [run.get("timestamps_skipped", 0) for run in _runs(target)]
    assert skipped == [0, _ITEMS]


def test_direct_run_spans_stamp_written_to_target(tmp_path: Path) -> None:
    target = tmp_path / "out.zarr"
    assert _ingest(_PLUGIN, target).exit_code == 0

    spans = _spans(target)

    assert _covered_slots(spans) == set(range(_ITEMS))
    assert {(span.meta or {})["written_to"] for span in spans} == {"target"}


def test_staged_run_spans_stamp_written_to_workspace(tmp_path: Path) -> None:
    target = tmp_path / "out.zarr"
    assert _ingest(_PLUGIN, target, write_mode="staged").exit_code == 0

    spans = _spans(target)

    assert _covered_slots(spans) == set(range(_ITEMS))
    assert {(span.meta or {})["written_to"] for span in spans} == {"workspace"}


def test_show_options_lists_resume_skip_written() -> None:
    result = CliRunner().invoke(cli, ["ingest", _PLUGIN, "--show-options"])

    assert result.exit_code == 0, result.output
    assert re.search(r"--option resume_skip_written\b", result.output)
