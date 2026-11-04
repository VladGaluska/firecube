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

"""Pre-decode resume filter: span coverage query and slot-based item dropping."""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cftime
import pytest

from firecube.core.controlplane import ChunkManager, SpanCoverage
from firecube.core.index_resolve import ResolvedIndex, resolve_index_spec
from firecube.core.index_spec import IndexSpec, RegularTimeAxis
from firecube.ingestor.runtime.resume_filter import (
    UnplaceableItemError,
    active_position_group,
    filter_items_by_written_slots,
    written_slot_coverage,
)
from tests.helpers.resume_spans import record_run_with_span
from tests.helpers.storage import make_test_binding

pytestmark = pytest.mark.unit

PRODUCT = "resume_product"
PLUGIN = "resume_plugin"
_LOG = logging.getLogger("firecube.test.resume_filter")


@pytest.fixture
def manager(tmp_path: Path) -> Iterator[ChunkManager]:
    mgr = ChunkManager(binding=make_test_binding(tmp_path), workspace=tmp_path)
    yield mgr
    mgr.close()


def _record_run(
    manager: ChunkManager,
    *,
    run_id: str,
    group: str = "data",
    ranges: list[Any],
    terminal: str | None,
    span_meta: dict[str, Any] | None = None,
    plugin: str = PLUGIN,
) -> None:
    record_run_with_span(
        manager,
        product=PRODUCT,
        plugin=plugin,
        run_id=run_id,
        group=group,
        ranges=ranges,
        terminal=terminal,
        span_meta=span_meta,
    )


def _coverage(manager: ChunkManager, groups: tuple[str, ...] = ("data",)) -> dict[str, Any]:
    return written_slot_coverage(
        chunk_manager=manager, product=PRODUCT, plugin_name=PLUGIN, groups=groups, log=_LOG
    )


# --- written_slot_coverage -------------------------------------------------


def test_coverage_merges_complete_run_spans_per_group(manager: ChunkManager) -> None:
    _record_run(manager, run_id="r1", group="a", ranges=[[0, 4], [10, 12]], terminal="complete")
    _record_run(manager, run_id="r2", group="a", ranges=[[5, 6]], terminal="complete")
    _record_run(manager, run_id="r3", group="b", ranges=[[20, 20]], terminal="complete")

    coverage = _coverage(manager, groups=("a", "b"))

    # Inclusive [0,4] and [5,6] touch and merge; [10,12] stays apart.
    assert coverage == {"a": [(0, 7), (10, 13)], "b": [(20, 21)]}


def test_coverage_includes_history_spans_stamped_target(manager: ChunkManager) -> None:
    _record_run(
        manager,
        run_id="crashed",
        ranges=[[3, 5]],
        terminal=None,
        span_meta={"written_to": "target"},
    )
    _record_run(
        manager,
        run_id="failed",
        ranges=[[8, 8]],
        terminal="failed",
        span_meta={"written_to": "target"},
    )

    assert _coverage(manager) == {"data": [(3, 6), (8, 9)]}


def test_coverage_ignores_unstamped_history_spans(manager: ChunkManager) -> None:
    _record_run(manager, run_id="crashed-old", ranges=[[0, 9]], terminal=None)
    _record_run(manager, run_id="failed-old", ranges=[[20, 29]], terminal="failed")

    assert _coverage(manager) == {"data": []}


def test_coverage_ignores_workspace_stamped_history_spans(manager: ChunkManager) -> None:
    _record_run(
        manager,
        run_id="staged-crash",
        ranges=[[0, 9]],
        terminal=None,
        span_meta={"written_to": "workspace"},
    )
    _record_run(
        manager,
        run_id="staged-failed",
        ranges=[[20, 29]],
        terminal="failed",
        span_meta={"written_to": "workspace"},
    )

    assert _coverage(manager) == {"data": []}


def test_coverage_trusts_workspace_stamped_span_of_complete_run(manager: ChunkManager) -> None:
    _record_run(
        manager,
        run_id="staged-done",
        ranges=[[0, 9]],
        terminal="complete",
        span_meta={"written_to": "workspace"},
    )

    assert _coverage(manager) == {"data": [(0, 10)]}


def test_coverage_ignores_unknown_written_to_values(manager: ChunkManager) -> None:
    for index, value in enumerate(["Target", "target ", "durable", "", "TARGET"]):
        _record_run(
            manager,
            run_id=f"odd-{index}",
            ranges=[[index * 10, index * 10 + 9]],
            terminal=None,
            span_meta={"written_to": value},
        )

    assert _coverage(manager) == {"data": []}


def test_coverage_ignores_other_plugin_spans(manager: ChunkManager) -> None:
    _record_run(manager, run_id="mine", ranges=[[0, 1]], terminal="complete")
    _record_run(manager, run_id="theirs", ranges=[[2, 9]], terminal="complete", plugin="other")
    _record_run(
        manager,
        run_id="theirs-crash",
        ranges=[[10, 19]],
        terminal=None,
        span_meta={"written_to": "target"},
        plugin="other",
    )

    assert _coverage(manager) == {"data": [(0, 2)]}


def test_coverage_tolerates_malformed_span_ranges(manager: ChunkManager, tmp_path: Path) -> None:
    # The recorder refuses malformed ranges, so a valid span is recorded and its
    # stored ranges are then damaged in the WAL, as a hand-edited or foreign
    # writer's record would be.
    _record_run(manager, run_id="messy", ranges=[[0, 1]], terminal="complete")
    _record_run(manager, run_id="clean", ranges=[[50, 51]], terminal="complete")
    damaged = [[0, 1], [5], "xy", [None, 3], ["a", "b"], [4, 4, 4], [7, 8]]
    damaged_events = 0
    for events_file in (tmp_path / PRODUCT / ".firecube" / "runs" / "messy").glob("events-*.jsonl"):
        lines = []
        for line in events_file.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            payload = event.get("record", {}).get("span")
            if isinstance(payload, dict):
                payload["time_index_ranges"] = damaged
                damaged_events += 1
            lines.append(json.dumps(event))
        events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert damaged_events == 1

    assert _coverage(manager) == {"data": [(0, 2), (7, 9), (50, 52)]}


def test_coverage_skips_groups_not_requested(manager: ChunkManager) -> None:
    _record_run(manager, run_id="r1", group="other", ranges=[[0, 9]], terminal="complete")

    assert _coverage(manager, groups=("data",)) == {"data": []}


def test_coverage_empty_control_plane_returns_empty_intervals(manager: ChunkManager) -> None:
    assert _coverage(manager, groups=()) == {}
    assert _coverage(manager, groups=("data",)) == {"data": []}


def _record_maintenance(manager: ChunkManager, *, op: str, state: str) -> None:
    run_id = f"maintenance-{op}-{state}"
    manager.record_maintenance_started(product=PRODUCT, run_id=run_id, op=op)
    if state == "completed":
        manager.record_maintenance_completed(product=PRODUCT, run_id=run_id, op=op)
    elif state == "failed":
        manager.record_maintenance_failed(product=PRODUCT, run_id=run_id, op=op, error="boom")
    else:
        manager.close()


def _crashed_target_run(manager: ChunkManager) -> None:
    _record_run(
        manager,
        run_id="crashed",
        ranges=[[0, 9]],
        terminal="failed",
        span_meta={"written_to": "target"},
    )


def test_coverage_ignores_history_after_a_data_changing_maintenance_run(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    # A delete, scrub or archive restore may have emptied slots the crashed run's
    # span still claims, in whatever state the maintenance run ended.
    for op in ("delete", "scrub", "archive_restore"):
        for state in ("completed", "failed", "started"):
            with subtests.test(msg=f"{op}_{state}"):
                mgr = ChunkManager(
                    binding=make_test_binding(tmp_path / f"{op}_{state}"),
                    workspace=tmp_path / f"{op}_{state}",
                )
                try:
                    _crashed_target_run(mgr)
                    _record_maintenance(mgr, op=op, state=state)

                    assert _coverage(mgr) == {"data": []}
                finally:
                    mgr.close()


def test_coverage_keeps_history_after_archive_create_maintenance_run(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    # Creating an archive reads the store and changes none of its cells.
    for state in ("completed", "failed", "started"):
        with subtests.test(msg=state):
            mgr = ChunkManager(
                binding=make_test_binding(tmp_path / state), workspace=tmp_path / state
            )
            try:
                _crashed_target_run(mgr)
                _record_maintenance(mgr, op="archive_create", state=state)

                assert _coverage(mgr) == {"data": [(0, 10)]}
            finally:
                mgr.close()


def test_coverage_keeps_complete_run_spans_after_a_data_changing_maintenance_run(
    manager: ChunkManager,
) -> None:
    # Deletion marks the spans it removes replaced, so the complete-run query
    # needs no maintenance gate: what is still active there is still written.
    _record_run(manager, run_id="complete", ranges=[[20, 29]], terminal="complete")
    _crashed_target_run(manager)
    _record_maintenance(manager, op="delete", state="completed")

    assert _coverage(manager) == {"data": [(20, 30)]}


def test_coverage_ignores_non_active_spans_of_complete_runs(manager: ChunkManager) -> None:
    output_path = f"{manager.base_uri.rstrip('/')}/{PRODUCT}"
    manager.record_run_started(
        product=PRODUCT,
        run_id="mixed",
        output_path=output_path,
        output_format="zarr",
        size=0,
        meta={"plugin": PLUGIN},
    )
    for batch, status, slot in (("b1", "active", 0), ("b2", "failed", 10), ("b3", "skipped", 20)):
        manager.record_span(
            product=PRODUCT,
            run_id="mixed",
            batch_id=batch,
            group="data",
            status=status,
            reason=None if status == "active" else "not written",
            coverage=SpanCoverage(
                group="data",
                arrays=["data/values"],
                time_index_ranges=[[slot, slot + 4]],
                time_min=f"2026-01-01T00:{slot:02d}:00Z",
                time_max=f"2026-01-01T01:{slot:02d}:00Z",
            ),
            meta={
                "plugin": PLUGIN,
                "group": "data",
                "time_min": f"2026-01-01T00:{slot:02d}:00Z",
                "time_max": f"2026-01-01T01:{slot:02d}:00Z",
            },
        )
    manager.record_run_terminal(
        product=PRODUCT,
        run_id="mixed",
        output_path=output_path,
        output_format="zarr",
        size=1,
        meta={"plugin": PLUGIN},
        status="complete",
    )

    assert _coverage(manager) == {"data": [(0, 5)]}


class _ListChunksSpy:
    """A control plane that delegates to a real one and records every ``list_chunks`` call."""

    def __init__(self, inner: ChunkManager) -> None:
        self._inner = inner
        self.history_queries = 0

    def list_chunks(self, **kwargs: Any) -> Any:
        if kwargs.get("include_replaced"):
            self.history_queries += 1
        return self._inner.list_chunks(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_coverage_scans_history_only_when_a_run_did_not_complete(tmp_path: Path) -> None:
    rows = {"all_runs_complete": ("complete", 0), "a_run_failed": ("failed", 2)}
    for name, (terminal, expected_history_queries) in rows.items():
        mgr = ChunkManager(binding=make_test_binding(tmp_path / name), workspace=tmp_path / name)
        try:
            _record_run(
                mgr,
                run_id="only",
                ranges=[[0, 9]],
                terminal=terminal,
                span_meta={"written_to": "target"},
            )
            spy = _ListChunksSpy(mgr)

            coverage = written_slot_coverage(
                chunk_manager=spy,  # type: ignore[arg-type]
                product=PRODUCT,
                plugin_name=PLUGIN,
                groups=("data",),
                log=_LOG,
            )

            assert coverage == {"data": [(0, 10)]}, name
            # With a failed run: one query for the target-stamped spans, one for the
            # maintenance runs.
            assert spy.history_queries == expected_history_queries, name
        finally:
            mgr.close()


# --- active_position_group --------------------------------------------------


def _axis(slot_count: int = 100, *, epoch: str = "2026-01-01T00:00:00Z") -> RegularTimeAxis:
    return RegularTimeAxis(
        coordinate="timestamp",
        epoch=epoch,
        cadence_s=60,
        mode="exact",
        slot_count=slot_count,
    )


def _resolved(groups: dict[str, RegularTimeAxis]) -> ResolvedIndex:
    return resolve_index_spec(
        IndexSpec(name="resume_filter_v1", groups=groups), time_dim_name="timestamp"
    )


def _at(minute: int) -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(minutes=minute)


def test_single_group_resolves() -> None:
    assert active_position_group(_resolved({"data": _axis()}), None) == "data"


def test_slot_group_wins_and_unknown_slot_group_disables() -> None:
    shared = _axis()
    resolved = _resolved({"a": shared, "b": _axis(slot_count=50)})

    assert active_position_group(resolved, "b") == "b"
    assert active_position_group(resolved, "missing") is None


def test_shared_axis_multi_group_resolves() -> None:
    shared = _axis()

    assert active_position_group(_resolved({"a": shared, "b": shared}), None) == "a"


def test_distinct_axes_returns_none() -> None:
    # Value-equal but distinct instances: the slot-range filter refuses these.
    resolved = _resolved({"a": _axis(), "b": _axis()})

    assert active_position_group(resolved, None) is None


# --- filter_items_by_written_slots ----------------------------------------


def _filter(
    items: list[int],
    *,
    resolved: ResolvedIndex,
    coverage: dict[str, list[tuple[int, int]]],
    required: list[str],
    position: Any = None,
    active: str = "data",
) -> tuple[list[int], int]:
    return filter_items_by_written_slots(
        items,
        resolved=resolved,
        coverage=coverage,
        active_group=active,
        required_groups=required,
        position_for_item=position or (lambda item: _at(item)),
        log=_LOG,
    )


def test_drops_item_only_when_every_required_group_covered() -> None:
    shared = _axis()
    resolved = _resolved({"a": shared, "b": shared})
    coverage = {"a": [(0, 10)], "b": [(0, 5)]}

    kept, dropped = _filter(
        [2, 7], resolved=resolved, coverage=coverage, required=["a", "b"], active="a"
    )

    # Slot 2 is in both groups; slot 7 is covered in a only.
    assert kept == [7]
    assert dropped == 1


def test_keeps_item_with_none_coordinate() -> None:
    kept, dropped = _filter(
        [0, 1, 2],
        resolved=_resolved({"data": _axis()}),
        coverage={"data": [(0, 100)]},
        required=["data"],
        position=lambda item: None if item == 1 else _at(item),
    )

    assert kept == [1]
    assert dropped == 2


def test_keeps_item_whose_index_cannot_place_the_coordinate(
    subtests: pytest.Subtests, caplog: pytest.LogCaptureFixture
) -> None:
    resolved = _resolved({"data": _axis(slot_count=10)})
    rows = {
        "predates_epoch": _at(-5),
        "off_cadence": _at(0) + dt.timedelta(seconds=7),
        "not_a_datetime": "not-a-time",
    }
    for name, coordinate in rows.items():
        with subtests.test(msg=name):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=_LOG.name):
                kept, dropped = _filter(
                    [0],
                    resolved=resolved,
                    coverage={"data": [(0, 10)]},
                    required=["data"],
                    position=lambda _item, c=coordinate: c,
                )
            assert kept == [0]
            assert dropped == 0
            assert "1 item(s) kept because they could not be placed" in caplog.text


def test_keeps_item_whose_position_for_item_raises_a_placement_error(
    subtests: pytest.Subtests, caplog: pytest.LogCaptureFixture
) -> None:
    errors = {
        "ValueError": ValueError("bad item"),
        "TypeError": TypeError("bad item"),
        "IndexError": IndexError("bad item"),
        "KeyError": KeyError("bad item"),
    }
    for name, error in errors.items():

        def position(item: int, error: Exception = error) -> Any:
            if item % 2:
                raise error
            return _at(item)

        with subtests.test(msg=name):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=_LOG.name):
                kept, dropped = _filter(
                    [0, 1, 2, 3, 4],
                    resolved=_resolved({"data": _axis()}),
                    coverage={"data": [(0, 100)]},
                    required=["data"],
                    position=position,
                )
            # Even items sit on written slots; the two odd ones cannot be placed.
            assert kept == [1, 3]
            assert dropped == 3
            assert "2 item(s) kept because they could not be placed" in caplog.text
            assert f"errors: {{'{name}': 2}}" in caplog.text
            assert f"first error: {error!r}" in caplog.text


def test_warning_counts_each_error_type(caplog: pytest.LogCaptureFixture) -> None:
    def position(item: int) -> Any:
        if item == 1:
            raise TypeError("bad item")
        if item in (2, 3):
            raise KeyError("bad item")
        return _at(item)

    with caplog.at_level(logging.WARNING, logger=_LOG.name):
        kept, dropped = _filter(
            [0, 1, 2, 3, 4],
            resolved=_resolved({"data": _axis()}),
            coverage={"data": [(0, 100)]},
            required=["data"],
            position=position,
        )

    assert kept == [1, 2, 3]
    assert dropped == 2
    assert "errors: {'TypeError': 1, 'KeyError': 2}" in caplog.text
    assert "first error: TypeError('bad item')" in caplog.text


def test_keeps_item_the_plugin_failed_to_inspect_and_names_the_plugins_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def position(item: int) -> Any:
        if item % 2:
            raise UnplaceableItemError from OSError("disk gone")
        return _at(item)

    with caplog.at_level(logging.WARNING, logger=_LOG.name):
        kept, dropped = _filter(
            [0, 1, 2, 3, 4],
            resolved=_resolved({"data": _axis()}),
            coverage={"data": [(0, 100)]},
            required=["data"],
            position=position,
        )

    assert kept == [1, 3]
    assert dropped == 3
    assert "2 item(s) kept because they could not be placed" in caplog.text
    assert "errors: {'OSError': 2}" in caplog.text
    assert "first error: OSError('disk gone')" in caplog.text


def test_unexpected_position_error_propagates() -> None:
    def position(item: int) -> Any:
        raise RuntimeError("plugin bug")

    with pytest.raises(RuntimeError, match="plugin bug"):
        _filter(
            [0],
            resolved=_resolved({"data": _axis()}),
            coverage={"data": [(0, 100)]},
            required=["data"],
            position=position,
        )


def test_boundary_slot_at_coverage_edge(subtests: pytest.Subtests) -> None:
    resolved = _resolved({"data": _axis()})
    # Half-open [5, 10): slots 5..9 covered.
    rows = [(4, False), (5, True), (9, True), (10, False)]
    for slot, covered in rows:
        with subtests.test(msg=f"slot_{slot}"):
            kept, dropped = _filter(
                [slot],
                resolved=resolved,
                coverage={"data": [(5, 10)]},
                required=["data"],
            )
            assert dropped == (1 if covered else 0)
            assert kept == ([] if covered else [slot])


def test_calendar_axis_coordinate_positions_via_encode() -> None:
    axis = RegularTimeAxis(
        coordinate="timestamp",
        epoch="2049-01-01T00:00:00Z",
        cadence_s=86400,
        mode="exact",
        slot_count=90,
        calendar="360_day",
    )
    resolved = _resolved({"data": axis})

    def day_of_360_year(item: int) -> cftime.Datetime360Day:
        # Day offset 30 on a 360_day calendar is 2049-02-01, which a Gregorian
        # calendar would call 2049-01-31.
        return cftime.Datetime360Day(2049, 1 + item // 30, 1 + item % 30)

    kept, dropped = _filter(
        [0, 29, 30, 31, 59, 60],
        resolved=resolved,
        coverage={"data": [(30, 60)]},
        required=["data"],
        position=day_of_360_year,
    )

    assert kept == [0, 29, 60]
    assert dropped == 3


def test_all_items_dropped_returns_empty_and_count() -> None:
    kept, dropped = _filter(
        [0, 1, 2, 3],
        resolved=_resolved({"data": _axis()}),
        coverage={"data": [(0, 4)]},
        required=["data"],
    )

    assert kept == []
    assert dropped == 4


def test_no_coverage_keeps_every_item_and_logs_evidence(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=_LOG.name):
        kept, dropped = _filter(
            [0, 1],
            resolved=_resolved({"data": _axis()}),
            coverage={},
            required=["data"],
        )

    assert (kept, dropped) == ([0, 1], 0)
    assert (
        "stage=resume_written_filter group=data required_groups=['data'] "
        "original_count=2 filtered_count=2 dropped_count=0"
    ) in caplog.text
