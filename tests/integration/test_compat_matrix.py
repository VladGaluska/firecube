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

"""Compatibility matrix integration tests for time_dim_name behavior.

Each test corresponds to one row of the Existing-Cube Compatibility Matrix
defined in the CF-1.8 compliance contract decisions.

| # | Existing cube dim                  | Plugin-declared dim | Expected action                                    |
|---|------------------------------------|---------------------|----------------------------------------------------|
| 1 | (new cube, no existing)            | ``timestamp``       | write new cube with ``timestamp``                  |
| 2 | (new cube, no existing)            | ``time``            | write new cube with ``time``                       |
| 3 | (new cube, no existing)            | ``foo`` (any other) | write new cube with ``foo``                        |
| 4 | ``timestamp``                      | ``timestamp``       | append (back-compat path)                          |
| 5 | ``timestamp``                      | ``time``            | FAIL with exact migration guidance                 |
| 6 | ``time``                           | ``time``            | append                                             |
| 7 | ``time``                           | ``timestamp``       | FAIL with exact migration guidance                 |
| 8 | both ``time`` AND ``timestamp``    | (any)               | FAIL with ambiguous-state diagnostic               |
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from firecube.ingestor.errors import ConfigurationError
from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility
from tests.fixtures.cf_dataset_fixtures import (
    make_cf_compliant_dataset,
    make_legacy_timestamp_dataset,
)
from tests.helpers.rows import table_row

# New-cube rows 1-3: (row id = former test name, declared dim).
_NEW_CUBE_ROWS = (
    ("test_row1_new_cube_timestamp_dim_allowed", "timestamp"),
    ("test_row2_new_cube_time_dim_allowed", "time"),
    ("test_row3_new_cube_arbitrary_dim_allowed", "foo"),
)


def _legacy_timestamp_cube(target: str) -> None:
    make_legacy_timestamp_dataset().to_zarr(target, mode="w", zarr_format=3, consolidated=False)


def _cf_time_cube(target: str) -> None:
    make_cf_compliant_dataset(time_dim="time").to_zarr(
        target, mode="w", zarr_format=3, consolidated=False
    )


def _ambiguous_cube(target: str) -> None:
    ambiguous = xr.Dataset(
        {
            "temperature": (
                ["time", "timestamp", "lat"],
                np.zeros((2, 2, 3), dtype=np.float32),
            )
        },
        coords={"time": [0, 1], "timestamp": [0, 1], "lat": [10, 20, 30]},
    )
    ambiguous.to_zarr(target, mode="w", zarr_format=3, consolidated=False)


# Existing-cube rows 4-7: (row id = former test name, cube writer, declared dim,
# whether the append must be refused with migration guidance).
_EXISTING_CUBE_ROWS = (
    ("test_row4_existing_timestamp_matches_timestamp", _legacy_timestamp_cube, "timestamp", False),
    (
        "test_row5_existing_timestamp_mismatch_time_fails_with_guidance",
        _legacy_timestamp_cube,
        "time",
        True,
    ),
    ("test_row6_existing_time_matches_time", _cf_time_cube, "time", False),
    (
        "test_row7_existing_time_mismatch_timestamp_fails_with_guidance",
        _cf_time_cube,
        "timestamp",
        True,
    ),
)


@pytest.mark.integration
def test_new_cube_accepts_any_declared_dim(tmp_path, subtests: pytest.Subtests):
    """Rows 1-3: no existing cube, any declared dim -> no error (written fresh)."""
    for row_id, declared in _NEW_CUBE_ROWS:
        with table_row(subtests, row_id, tmp_path) as (workdir, _mp):
            target = str(workdir / "cube.zarr")
            verify_dim_compatibility(target, declared, group_paths=["."], storage_config=None)


@pytest.mark.integration
def test_existing_cube_dim_compatibility(tmp_path, subtests: pytest.Subtests):
    """Rows 4-8: matching dims append, mismatches and ambiguity are refused."""
    for row_id, make_cube, declared, refused in _EXISTING_CUBE_ROWS:
        with table_row(subtests, row_id, tmp_path) as (workdir, _mp):
            target = str(workdir / "cube.zarr")
            make_cube(target)
            if not refused:
                verify_dim_compatibility(target, declared, group_paths=["."], storage_config=None)
                continue
            with pytest.raises(ConfigurationError) as exc:
                verify_dim_compatibility(target, declared, group_paths=["."], storage_config=None)
            msg = str(exc.value)
            assert "Refusing to append" in msg
            assert "declared time dimension" in msg
            assert "migrate the existing cube" in msg

    with table_row(subtests, "test_row8_ambiguous_both_dims_fails", tmp_path) as (workdir, _mp):
        target = str(workdir / "cube.zarr")
        _ambiguous_cube(target)
        with pytest.raises(ConfigurationError) as exc:
            verify_dim_compatibility(target, "time", group_paths=["."], storage_config=None)
        msg = str(exc.value)
        assert any(token in msg.lower() for token in ("ambiguous", "both", "conflicting"))
