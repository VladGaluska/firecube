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

import time
from unittest.mock import MagicMock

import pytest

from firecube.core.controlplane.types import RunInfo
from firecube.ingestor.errors import RangeOverlapError, ResumeConflictError
from firecube.ingestor.runtime.resume_guard import ResumeGuard


def _make_ctx(*, force_reingest: bool = False, **options):
    ctx = MagicMock()
    ctx.force_reingest = force_reingest
    ctx.option.side_effect = lambda name, default=None: options.get(name, default)
    return ctx


def _make_run(
    run_id: str = "run-123",
    *,
    slot_range: tuple[int, int] | None = None,
    stale: bool = False,
) -> RunInfo:
    now = time.time()
    return RunInfo(
        product="P",
        run_id=run_id,
        status="running",
        run_dir=f"/tmp/{run_id}",
        run_uri=f"file:///tmp/{run_id}",
        started_at=now,
        updated_at=0.0 if stale else now,
        completed_at=None,
        events=1,
        parts=1,
        stale_threshold_s=3600,
        slot_range=slot_range,
    )


def _make_guard(chunk_manager: MagicMock) -> ResumeGuard:
    return ResumeGuard(
        plugin_name="test_product",
        chunk_manager=chunk_manager,
        log=MagicMock(),
        slice_meta_keys=(),
    )


def _make_chunk_manager(runs: list[RunInfo]) -> MagicMock:
    chunk_manager = MagicMock()
    chunk_manager.list_runs.return_value = runs
    chunk_manager.list_chunks.return_value = []
    return chunk_manager


@pytest.mark.unit
@pytest.mark.parametrize(
    ("runs", "slot_range"),
    [
        pytest.param([], (0, 100), id="no_active_runs_allows"),
        pytest.param([_make_run(slot_range=(0, 100))], (100, 200), id="disjoint_ranges_allowed"),
        pytest.param(
            [_make_run(slot_range=None, stale=True)], (0, 100), id="stale_non_range_allows_range"
        ),
        pytest.param([_make_run(slot_range=(0, 1))], (1, 2), id="single_slot_adjacent_ranges"),
    ],
)
def test_range_invocation_allowed(runs: list[RunInfo], slot_range: tuple[int, int]) -> None:
    chunk_manager = _make_chunk_manager(runs)
    guard = _make_guard(chunk_manager)

    guard.enforce(ctx=_make_ctx(), product="P", slot_range=slot_range, slot_group=None)

    chunk_manager.list_chunks.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("run_range", "requested", "error", "match"),
    [
        pytest.param(
            (0, 100), (0, 100), ResumeConflictError, "same slot_range", id="same_range_no_resume"
        ),
        pytest.param(
            (0, 100), (50, 150), RangeOverlapError, "overlaps", id="overlapping_different_range"
        ),
        pytest.param(None, (0, 100), ResumeConflictError, "Non-range run", id="non_range_active"),
    ],
)
def test_range_invocation_blocked(
    run_range: tuple[int, int] | None,
    requested: tuple[int, int],
    error: type[Exception],
    match: str,
) -> None:
    chunk_manager = _make_chunk_manager([_make_run(slot_range=run_range)])
    guard = _make_guard(chunk_manager)

    with pytest.raises(error, match=match):
        guard.enforce(ctx=_make_ctx(), product="P", slot_range=requested, slot_group=None)

    chunk_manager.list_chunks.assert_not_called()


@pytest.mark.unit
def test_same_range_with_resume_allows():
    chunk_manager = _make_chunk_manager([_make_run(slot_range=(0, 100))])
    guard = _make_guard(chunk_manager)

    guard.enforce(
        ctx=_make_ctx(resume_existing=True), product="P", slot_range=(0, 100), slot_group=None
    )

    guard.log.warning.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
    chunk_manager.list_chunks.assert_called_once()


@pytest.mark.unit
def test_same_range_with_force_reingest_allows_without_run_scan():
    chunk_manager = _make_chunk_manager([_make_run(slot_range=(0, 100))])
    guard = _make_guard(chunk_manager)

    guard.enforce(
        ctx=_make_ctx(force_reingest=True), product="P", slot_range=(0, 100), slot_group=None
    )

    guard.log.warning.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]
    chunk_manager.list_runs.assert_not_called()
    chunk_manager.list_chunks.assert_not_called()


@pytest.mark.unit
def test_single_pod_mode_blocks_as_before():
    chunk_manager = _make_chunk_manager([_make_run(slot_range=None)])
    guard = _make_guard(chunk_manager)

    with pytest.raises(ResumeConflictError, match="Non-terminal run"):
        guard.enforce(ctx=_make_ctx(), product="P", slot_range=None)

    chunk_manager.list_chunks.assert_not_called()
