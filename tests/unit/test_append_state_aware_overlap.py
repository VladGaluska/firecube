# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for state-aware append overlap classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import LocalStore

from firecube.ingestor.errors import (
    AppendOverwriteRefused,
    DuplicateExistingTimestampsError,
    InsertRefusedError,
)
from firecube.ingestor.runtime.zarr.append_services import AppendResumeService
from tests.helpers.rows import table_row

pytestmark = pytest.mark.unit


def _make_service(
    *,
    append_dim: str = "timestamp",
    read_source_uri: str | None = None,
) -> AppendResumeService:
    return AppendResumeService(
        read_source_uri=read_source_uri,
        read_storage_options=None,
        resume_existing=True,
        append_dim=append_dim,
        chunk_shape=None,
        shard_shape=None,
        sharding=False,
        logger=logging.getLogger("test-append-classification"),
        state_var_name="firecube_timestamp_state",
    )


def _times(*values: str) -> np.ndarray:
    return np.array(values, dtype="datetime64[s]")


def _open_group_with_coord(
    tmp_path: Path,
    *,
    times: np.ndarray,
    state: np.ndarray | None = None,
    append_dim: str = "timestamp",
    state_array_name: str = "firecube_timestamp_state",
) -> zarr.Group:
    store_path = tmp_path / "store.zarr"
    root = zarr.open_group(store=LocalStore(store_path), mode="w", zarr_format=3)
    n = int(times.shape[0])
    time_arr = root.create_array(
        append_dim,
        shape=(n,),
        dtype=times.dtype,
        chunks=(max(n, 1),),
    )
    time_arr[:] = times

    state_vals = state if state is not None else np.ones((n,), dtype=np.uint8)
    state_arr = root.create_array(
        state_array_name,
        shape=(n,),
        dtype=np.uint8,
        chunks=(max(n, 1),),
    )
    state_arr[:] = state_vals
    return zarr.open_group(store=LocalStore(store_path), mode="r", zarr_format=3)


def _assert_slice(actual: slice | None, start: int, stop: int) -> None:
    assert actual is not None
    assert actual.start == start
    assert actual.stop == stop
    assert actual.step is None


def test_no_overlap_returns_append_only(tmp_path: Path) -> None:
    group = _open_group_with_coord(tmp_path, times=_times("2024-01-01T00", "2024-01-01T01"))
    batch = _times("2024-01-01T02", "2024-01-01T03")

    result = _make_service().classify_incoming(batch, group)

    assert result.mode == "append_only"
    assert result.overwrite_slice is None
    assert result.new_values == batch.tolist()


def test_full_overlap_returns_region_overwrite(subtests, tmp_path: Path) -> None:
    times = _times("2024-01-01T00", "2024-01-01T01", "2024-01-01T02")
    for row_id, state in [
        ("full_overlap_contiguous_state1_returns_region_overwrite", [1, 1, 1]),
        ("full_overlap_state2_refill_returns_region_overwrite", [2, 2, 2]),
        ("full_overlap_state3_failed_returns_region_overwrite", [3, 3, 3]),
    ]:
        with table_row(subtests, row_id, tmp_path) as (workdir, _):
            group = _open_group_with_coord(
                workdir, times=times, state=np.array(state, dtype=np.uint8)
            )

            result = _make_service().classify_incoming(times, group)

            assert result.mode == "region_overwrite"
            _assert_slice(result.overwrite_slice, 0, 3)
            assert result.new_values == []


def test_split_prefix_overlap_plus_tail(tmp_path: Path) -> None:
    group = _open_group_with_coord(
        tmp_path,
        times=_times("2024-01-01T00", "2024-01-01T01", "2024-01-01T02"),
        state=np.array([1, 2, 3], dtype=np.uint8),
    )
    batch = _times(
        "2024-01-01T01",
        "2024-01-01T02",
        "2024-01-01T03",
        "2024-01-01T04",
    )

    result = _make_service().classify_incoming(batch, group)

    assert result.mode == "split_region_plus_append"
    _assert_slice(result.overwrite_slice, 1, 3)
    assert result.new_values == batch[2:].tolist()


def _hours(*hours: int | None) -> np.ndarray:
    return _times(*("NaT" if h is None else f"2024-01-01T{h:02d}" for h in hours))


def test_classify_incoming_refusals(subtests, tmp_path: Path) -> None:
    append_refused = AppendOverwriteRefused
    rows = [
        ("insert_refused", [0, 2], None, [1], InsertRefusedError, "insert"),
        (
            "non_contiguous_overlap_refused",
            [0, 1, 2, 3, 4, 5],
            None,
            [0, 5],
            append_refused,
            "non_contiguous",
        ),
        ("incoming_duplicates_refused", [0], None, [1, 1], append_refused, "duplicates_incoming"),
        (
            "existing_duplicates_refused",
            [0, 0, 1],
            None,
            [2],
            DuplicateExistingTimestampsError,
            "duplicates_existing",
        ),
        ("incoming_nat_refused", [0], None, [None], append_refused, "nat_incoming"),
        ("existing_nat_refused", [0, None], None, [1], append_refused, "nat_existing"),
        # An unsorted axis makes ``coord.values[-1]`` an unreliable tail marker, so the
        # classifier refuses before any state-aware overlap logic, whatever the state.
        (
            "unsorted_existing_refused_regardless_of_state",
            [2, 0, 1],
            [1, 2, 3],
            [0, 1, 3],
            append_refused,
            "unsorted_existing_coord",
        ),
    ]
    for row_id, existing, state, batch, exc, reason in rows:
        with table_row(subtests, row_id, tmp_path) as (workdir, _):
            group = _open_group_with_coord(
                workdir,
                times=_hours(*existing),
                state=None if state is None else np.array(state, dtype=np.uint8),
            )

            with pytest.raises(exc) as exc_info:
                _make_service().classify_incoming(_hours(*batch), group)

            assert exc_info.value.reason == reason


def test_classifier_uses_configured_append_dim(tmp_path: Path) -> None:
    group = _open_group_with_coord(
        tmp_path,
        times=_times("2024-01-01T00"),
        append_dim="time",
    )
    batch = _times("2024-01-01T01")

    result = _make_service(append_dim="time").classify_incoming(batch, group)

    assert result.mode == "append_only"
    assert result.new_values == batch.tolist()


def test_malformed_units_propagates_loudly(tmp_path: Path) -> None:
    store_path = tmp_path / "store.zarr"
    root = zarr.open_group(store=LocalStore(store_path), mode="w", zarr_format=3)
    group = root.create_group("daily")
    time_arr = group.create_array(
        "timestamp",
        shape=(1,),
        dtype=np.dtype("int64"),
        chunks=(1,),
    )
    time_arr[:] = np.array([0], dtype=np.int64)
    time_arr.attrs["units"] = "totally not valid"
    state_arr = group.create_array(
        "firecube_timestamp_state",
        shape=(1,),
        dtype=np.uint8,
        chunks=(1,),
    )
    state_arr[:] = np.array([1], dtype=np.uint8)
    reopened = zarr.open_group(store=LocalStore(store_path), mode="r", zarr_format=3)
    source_uri = f"file://{store_path}"

    with pytest.raises(ValueError) as exc_info:
        _make_service(read_source_uri=source_uri).overlapping_values_state_aware(
            {pd.Timestamp("1970-01-01T00:00:00")},
            cast(zarr.Group, reopened["daily"]),
        )

    message = str(exc_info.value)
    assert source_uri in message
    assert "daily" in message
    assert "int64" in message
    assert "totally not valid" in message


def test_overlapping_values_state_aware(subtests, tmp_path: Path) -> None:
    # state 1 counts as an overlap; states 2 (deleted) and 3 (failed) are refillable.
    rows = [
        ("state1_slot_skipped_on_resume", [0, 1, 2], [1, 1, 1], [1], [1]),
        ("state2_slot_allowed_through_on_resume", [0, 1, 2], [1, 2, 1], [1], []),
        ("state3_slot_allowed_through_on_resume", [0, 1], [3, 3], [0, 1], []),
        ("partial_skip_mixed_states", [0, 1, 2], [1, 2, 3], [0, 1, 2, 3], [0]),
        ("empty_incoming_returns_empty", [0], [1], [], []),
        ("no_overlap_returns_empty", [0, 1], [1, 1], [5], []),
    ]
    stamp = "2024-01-01T{:02d}:00:00".format
    for row_id, existing, state, incoming, expected in rows:
        with table_row(subtests, row_id, tmp_path) as (workdir, _):
            group = _open_group_with_coord(
                workdir, times=_hours(*existing), state=np.array(state, dtype=np.uint8)
            )

            result = _make_service().overlapping_values_state_aware(
                {pd.Timestamp(stamp(h)) for h in incoming}, group
            )

            assert result == {pd.Timestamp(stamp(h)) for h in expected}
