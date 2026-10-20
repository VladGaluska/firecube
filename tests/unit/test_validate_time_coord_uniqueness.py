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

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from firecube.core.zarr.validation import validate_group_with_fs
from tests.helpers.storage import make_local_session

pytestmark = pytest.mark.unit


def _seed_group(
    store_path: Path,
    *,
    timestamps: np.ndarray,
    materialize_data_chunks: bool,
) -> None:
    root = zarr.open_group(store=str(store_path), mode="w", zarr_format=3)
    group = root.require_group("G")
    group.create_array(
        "timestamp",
        data=timestamps,
        chunks=(2,),
        dimension_names=("timestamp",),
    )
    group.create_array(
        "firecube_timestamp_state",
        data=np.ones(len(timestamps), dtype=np.uint8),
        chunks=(2,),
        dimension_names=("timestamp",),
    )
    data = group.create_array(
        "data",
        shape=(len(timestamps), 2),
        chunks=(2, 2),
        dtype=np.float32,
        fill_value=0.0,
        dimension_names=("timestamp", "x"),
    )
    if materialize_data_chunks:
        data[:] = np.arange(len(timestamps) * 2, dtype=np.float32).reshape(len(timestamps), 2)


def _validate(store_path: Path):
    session = make_local_session(str(store_path))
    return validate_group_with_fs(
        session.fs(), session.product.product_uri, "G", time_dim_name="timestamp"
    )


def test_duplicate_time_coord_values_are_invalid(tmp_path: Path) -> None:
    store_path = tmp_path / "duplicate.zarr"
    _seed_group(
        store_path,
        timestamps=np.array([0, 1, 1, 2], dtype=np.int64),
        materialize_data_chunks=True,
    )

    report = _validate(store_path)

    assert report.is_valid is False
    assert any("duplicate" in issue.lower() for issue in report.validity_issues)


def test_absent_chunk_keys_are_informational_only(tmp_path: Path) -> None:
    store_path = tmp_path / "absent-chunks.zarr"
    _seed_group(
        store_path,
        timestamps=np.array([0, 1, 2, 3], dtype=np.int64),
        materialize_data_chunks=False,
    )

    report = _validate(store_path)

    assert report.is_valid is True
    assert report.validity_issues == []
    assert report.absent_chunk_indices["G/data"] == [0, 1]


def test_sparse_direct_zarr_coordinate_with_nat_slots_is_valid(tmp_path: Path) -> None:
    """A dense DirectZarr coordinate carries NaT for never-written slots.

    Those slots are absent, not out of order or duplicated, so a store whose
    filled slots are unique and increasing must validate clean.
    """
    store_path = tmp_path / "sparse.zarr"
    root = zarr.open_group(store=str(store_path), mode="w", zarr_format=3)
    group = root.require_group("G")
    timestamps = np.array(
        ["2026-06-16T14:50", "NaT", "2026-06-16T15:10", "NaT"], dtype="datetime64[ns]"
    )
    group.create_array("timestamp", data=timestamps, chunks=(2,), dimension_names=("timestamp",))
    group.create_array(
        "data",
        shape=(4, 2),
        chunks=(2, 2),
        dtype=np.float32,
        fill_value=0.0,
        dimension_names=("timestamp", "x"),
    )

    report = _validate(store_path)

    assert report.is_valid is True, report.validity_issues
    assert report.validity_issues == []


def test_duplicate_filled_values_beside_nat_slots_are_still_invalid(tmp_path: Path) -> None:
    store_path = tmp_path / "sparse-dup.zarr"
    root = zarr.open_group(store=str(store_path), mode="w", zarr_format=3)
    group = root.require_group("G")
    timestamps = np.array(["2026-06-16T14:50", "NaT", "2026-06-16T14:50"], dtype="datetime64[ns]")
    group.create_array("timestamp", data=timestamps, chunks=(2,), dimension_names=("timestamp",))

    report = _validate(store_path)

    assert report.is_valid is False
    assert any("duplicate" in issue.lower() for issue in report.validity_issues)


_FILL = int(np.iinfo(np.int64).min)
_CALENDAR_ATTRS = {"units": "seconds since 1850-01-01 00:00:00", "calendar": "360_day"}


@pytest.mark.parametrize(
    ("values", "fill_value", "attributes", "diagnostic"),
    [
        pytest.param(
            [0, 86400, _FILL, _FILL, 345600], _FILL, _CALENDAR_ATTRS, None, id="fill-slots"
        ),
        pytest.param(
            [0, 86400, _FILL, 86400, 345600],
            _FILL,
            _CALENDAR_ATTRS,
            "duplicate",
            id="duplicate-beside-fill",
        ),
        pytest.param(
            [-1, -1, 3, 4, 5],
            -1,
            {"units": _CALENDAR_ATTRS["units"]},
            "duplicate",
            id="fill-is-an-ordinary-value-without-calendar",
        ),
        *[
            pytest.param(v, 0, {**_CALENDAR_ATTRS, "calendar": c}, d, id=f"{c}-{v}")
            for c, v, d in (
                ("gregorian", [0, 0, 1], "duplicate"),
                ("standard", [0, 0, 1], "duplicate"),
                ("gregorian", [1, 0, 2], "monotonic"),
            )
        ],
    ],
)
def test_only_encoded_calendar_fill_slots_are_absent_not_duplicates(
    tmp_path: Path,
    values: list[int],
    fill_value: int,
    attributes: dict[str, Any],
    diagnostic: str | None,
) -> None:
    store_path = tmp_path / "encoded.zarr"
    group = zarr.open_group(store=str(store_path), mode="w", zarr_format=3).require_group("G")
    coord = group.create_array(
        "timestamp",
        shape=(len(values),),
        chunks=(2,),
        dtype=np.int64,
        fill_value=fill_value,
        dimension_names=("timestamp",),
        attributes=attributes,
    )
    coord[:] = np.array(values, dtype=np.int64)

    report = _validate(store_path)

    assert report.is_valid is (diagnostic is None), report.validity_issues
    assert diagnostic is None or any(diagnostic in i.lower() for i in report.validity_issues)
