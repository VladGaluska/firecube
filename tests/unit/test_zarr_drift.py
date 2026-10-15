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

"""Tests for :mod:`firecube.core.zarr._drift` helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from firecube.core.zarr._drift import (
    GroupAttrsDiff,
    arrays_equal_missing_aware,
    chunk_by_chunk_equal,
    group_attrs_diff,
    touched_chunks_for_slice,
)

# ── arrays_equal_missing_aware ────────────────────────────────────────────────

_NAT_AB = np.array(["NaT", "2020-01-01"], dtype="datetime64[ns]")
_NAT_BA = np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]")
_NAN_MID = np.array([1.0, float("nan"), 3.0])
_ARRAY_EQUALITY_ROWS = (
    ("identical_float64_equal", np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), True),
    ("nan_same_positions_equal", _NAN_MID.copy(), np.array([1.0, float("nan"), 3.0]), True),
    (
        "nan_different_positions_not_equal",
        _NAN_MID.copy(),
        np.array([1.0, 2.0, float("nan")]),
        False,
    ),
    (
        "float32_vs_float64_same_values_equal",
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float64),
        True,
    ),
    (
        "float32_vs_bytes_not_equal",
        np.array([1.0], dtype=np.float32),
        np.array([b"\x00"], dtype="S1"),
        False,
    ),
    ("shape_mismatch_not_equal", np.array([1.0, 2.0]), np.array([1.0]), False),
    ("shape_mismatch_multidim_not_equal", np.zeros((3, 4)), np.zeros((3, 5)), False),
    ("nat_same_positions_equal", _NAT_AB.copy(), _NAT_AB.copy(), True),
    ("nat_different_positions_not_equal", _NAT_AB.copy(), _NAT_BA.copy(), False),
    ("string_arrays_equal", np.array(["foo", "bar"]), np.array(["foo", "bar"]), True),
    (
        "string_arrays_different_not_equal",
        np.array(["foo", "bar"]),
        np.array(["foo", "baz"]),
        False,
    ),
    (
        "int_vs_float_same_values_equal",
        np.array([1, 2, 3], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]),
        True,
    ),
    ("int_arrays_equal", np.array([1, 2, 3]), np.array([1, 2, 3]), True),
    ("int_arrays_different_not_equal", np.array([1, 2, 3]), np.array([1, 2, 4]), False),
    ("multidim_float_equal", np.arange(12.0).reshape(3, 4), np.arange(12.0).reshape(3, 4), True),
    ("empty_arrays_equal", np.array([], dtype=np.float64), np.array([], dtype=np.float32), True),
)


@pytest.mark.unit
def test_arrays_equal_missing_aware(subtests: Any) -> None:
    for row_id, a, b, expected in _ARRAY_EQUALITY_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            assert arrays_equal_missing_aware(a, b) is expected


# ── group_attrs_diff ──────────────────────────────────────────────────────────

_ARR12 = np.array([1.0, 2.0])
_GROUP_ATTRS_ROWS = (
    # (row id, stored, incoming, added, removed, changed)
    ("identical_attrs_is_empty", {"title": "A", "x": 1}, {"title": "A", "x": 1}, (), (), ()),
    ("empty_dicts_is_empty", {}, {}, (), (), ()),
    ("added_key_detected", {"title": "A"}, {"title": "A", "history": "new"}, ("history",), (), ()),
    ("removed_key_detected", {"title": "A", "old": 1}, {"title": "A"}, (), ("old",), ()),
    ("changed_value_detected", {"title": "Run 1"}, {"title": "Run 2"}, (), (), ("title",)),
    ("tuple_vs_list_same_values_is_not_a_diff", {"k": [1, 2, 3]}, {"k": (1, 2, 3)}, (), (), ()),
    (
        "nested_tuple_of_lists_is_not_a_diff",
        {"k": {"outer": [[1, 2], [3, 4]], "flag": True}},
        {"k": {"outer": ([1, 2], (3, 4)), "flag": True}},
        (),
        (),
        (),
    ),
    ("actual_value_change_is_still_flagged", {"k": [1, 2]}, {"k": [1, 3]}, (), (), ("k",)),
    (
        "mixed_diff",
        {"title": "Run 1", "Conventions": "CF-1.8"},
        {"title": "Run 2", "Conventions": "CF-1.8", "history": "appended"},
        ("history",),
        (),
        ("title",),
    ),
    ("ndarray_attr_equal_not_changed", {"b": _ARR12.copy()}, {"b": _ARR12.copy()}, (), (), ()),
    (
        "ndarray_attr_dtype_promoted_equal",
        {"b": _ARR12.astype(np.float32)},
        {"b": _ARR12.copy()},
        (),
        (),
        (),
    ),
    (
        "ndarray_attr_diff_reported_changed",
        {"b": _ARR12.copy()},
        {"b": np.array([1.0, 3.0])},
        (),
        (),
        ("b",),
    ),
    (
        # Engine is generic first-write-wins; no key is silently ignored.
        "no_volatile_attr_filter_baked_in",
        {"history": "run-1", "firecube_run_id": "aaa"},
        {"history": "run-2", "firecube_run_id": "bbb"},
        (),
        (),
        ("firecube_run_id", "history"),
    ),
    ("added_keys_sorted", {}, {"z": 1, "a": 2, "m": 3}, ("a", "m", "z"), (), ()),
)


@pytest.mark.unit
def test_group_attrs_diff(subtests: Any) -> None:
    for row_id, stored, incoming, added, removed, changed in _GROUP_ATTRS_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            d = group_attrs_diff(stored, incoming)
            assert isinstance(d, GroupAttrsDiff)
            assert (d.added, d.removed, d.changed) == (added, removed, changed)
            assert d.is_empty is (not (added or removed or changed))


# ── touched_chunks_for_slice ──────────────────────────────────────────────────

_TOUCHED_CHUNK_ROWS = (
    # (row id, array shape, chunk shape, region, expected chunk indices)
    ("touched_chunks_single_chunk", (10,), (5,), {0: slice(0, 3)}, [(0,)]),
    ("touched_chunks_spanning_two", (10,), (2,), {0: slice(1, 4)}, [(0,), (1,)]),
    (
        "touched_chunks_all_dims_when_not_in_region",
        (4, 6),
        (2, 3),
        {0: slice(0, 2)},
        [(0, 0), (0, 1)],
    ),
    ("touched_chunks_unaligned_boundary", (7,), (3,), {0: slice(4, 7)}, [(1,), (2,)]),
    (
        "touched_chunks_no_region_returns_full_grid",
        (4, 6),
        (2, 3),
        {},
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ),
    ("touched_chunks_slice_none_start_treated_as_zero", (10,), (5,), {0: slice(None, 3)}, [(0,)]),
    ("touched_chunks_slice_none_stop_treated_as_shape", (10,), (5,), {0: slice(5, None)}, [(1,)]),
    (
        "touched_chunks_full_region_covers_all",
        (10,),
        (2,),
        {0: slice(0, 10)},
        [(i,) for i in range(5)],
    ),
)


@pytest.mark.unit
def test_touched_chunks_for_slice(subtests: Any) -> None:
    for row_id, shape, chunk_shape, region, expected in _TOUCHED_CHUNK_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            chunks = touched_chunks_for_slice(
                array_shape=shape, chunk_shape=chunk_shape, region=region
            )
            assert sorted(chunks) == expected


@pytest.mark.unit
def test_touched_chunks_zero_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        touched_chunks_for_slice(array_shape=(10,), chunk_shape=(0,), region={0: slice(0, 3)})


# ── chunk_by_chunk_equal ──────────────────────────────────────────────────────

_CHUNK_COMPARE_ROWS = (
    # (row id, shape, chunks, shards, edit on both, edit on b only, compare kwargs, expected)
    ("chunk_by_chunk_identical_equal", (20,), (5,), None, None, None, {}, True),
    (
        "chunk_by_chunk_mismatch_detected",
        (20,),
        (5,),
        None,
        None,
        (slice(15, None), 999.0),
        {},
        False,
    ),
    ("chunk_by_chunk_multidim_equal", (6, 8), (3, 4), None, None, None, {}, True),
    (
        "chunk_by_chunk_multidim_last_chunk_mismatch",
        (6, 8),
        (3, 4),
        None,
        None,
        ((5, 7), -99.0),
        {},
        False,
    ),
    ("chunk_by_chunk_sharded_equal", (100,), (10,), (50,), None, None, {}, True),
    ("chunk_by_chunk_sharded_mismatch_detected", (100,), (10,), (50,), None, (75, -1.0), {}, False),
    ("chunk_by_chunk_nan_positions_equal", (10,), (4,), None, ([3, 7], np.nan), None, {}, True),
    ("chunk_by_chunk_unaligned_last_chunk_equal", (7,), (3,), None, None, None, {}, True),
    (
        "chunk_by_chunk_chunk_shape_override_used",
        (20,),
        (5,),
        None,
        None,
        None,
        {"chunk_shape": (4,)},
        True,
    ),
)


@pytest.mark.unit
def test_chunk_by_chunk_equal(subtests: Any, tmp_path: Path) -> None:
    for row_id, shape, chunks, shards, both, only_b, kwargs, expected in _CHUNK_COMPARE_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            vals = np.arange(float(np.prod(shape))).reshape(shape)
            arrays = [
                zarr.create_array(
                    f"{tmp_path}/{row_id}-{name}.zarr",
                    shape=shape,
                    dtype="f8",
                    chunks=chunks,
                    shards=shards,
                )
                for name in "ab"
            ]
            for arr in arrays:
                arr[:] = vals
                if both:
                    arr[both[0]] = both[1]
            if only_b:
                arrays[1][only_b[0]] = only_b[1]
            assert chunk_by_chunk_equal(*arrays, **kwargs) is expected


_INCOMING_NDARRAY_ROWS = (
    # (row id, stored length, incoming values)
    ("chunk_by_chunk_ndarray_incoming_equal", 20, np.arange(20.0)),
    (
        "chunk_by_chunk_ndarray_incoming_mismatch",
        20,
        np.where(np.arange(20) == 7, -1.0, np.arange(20.0)),
    ),
    ("chunk_by_chunk_shape_mismatch_not_equal", 10, np.arange(8.0)),
)


@pytest.mark.unit
def test_chunk_by_chunk_ndarray_incoming(subtests: Any, tmp_path: Path) -> None:
    for row_id, length, incoming in _INCOMING_NDARRAY_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            a = zarr.create_array(
                f"{tmp_path}/{row_id}.zarr", shape=(length,), dtype="f8", chunks=(5,)
            )
            a[:] = np.arange(float(length))
            assert chunk_by_chunk_equal(a, incoming) is (
                row_id == "chunk_by_chunk_ndarray_incoming_equal"
            )
