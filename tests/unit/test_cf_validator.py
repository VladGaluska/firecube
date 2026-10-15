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

# pyright: reportMissingImports=false

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from firecube.core.cf import CFReport, CFSeverity
from firecube.core.cf.validator import validate_cf18
from tests.fixtures.cf_dataset_fixtures import (
    make_broken_dataset,
    make_cf_compliant_dataset,
    make_legacy_timestamp_dataset,
)
from tests.helpers.rows import table_row


def _ids(report: CFReport, check_id: str) -> list[str]:
    return [f.id for f in report.findings if f.id == check_id]


def _by_id(report: CFReport, check_id: str):
    return [f for f in report.findings if f.id == check_id]


def _make_2d_lat_lon_dataset() -> xr.Dataset:
    n_y, n_x = 4, 5
    yy, xx = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")
    lat2d = -30 + 60 * yy / (n_y - 1)
    lon2d = -20 + 40 * xx / (n_x - 1)
    times = np.arange(2, dtype=np.float64)
    return xr.Dataset(
        {
            "temperature": (
                ("time", "y", "x"),
                np.zeros((2, n_y, n_x)),
                {"units": "K", "long_name": "T"},
            )
        },
        coords={
            "time": (
                "time",
                times,
                {
                    "units": "days since 2000-01-01",
                    "calendar": "standard",
                    "standard_name": "time",
                    "axis": "T",
                },
            ),
            "lat": (("y", "x"), lat2d, {"standard_name": "latitude", "units": "degrees_north"}),
            "lon": (("y", "x"), lon2d, {"standard_name": "longitude", "units": "degrees_east"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "2D aux coords",
            "institution": "x",
            "source": "x",
            "history": "x",
        },
    )


def test_validate_returns_cfreport_with_product_group() -> None:
    ds = make_cf_compliant_dataset()
    report = validate_cf18(ds, "TEST_PRODUCT", "level1")

    assert isinstance(report, CFReport)
    assert report.product == "TEST_PRODUCT"
    assert report.group == "level1"


def test_compliant_dataset_zero_errors() -> None:
    ds = make_cf_compliant_dataset()
    report = validate_cf18(ds, "p", "g")

    assert report.summary.errors == 0


def _time_only_dataset() -> xr.Dataset:
    """A CF time axis and one variable, no latitude or longitude coordinate."""
    time_attrs = {
        "units": "days since 2000-01-01",
        "standard_name": "time",
        "axis": "T",
        "calendar": "standard",
    }
    return xr.Dataset(
        {"v": (("time",), np.zeros(2), {"units": "K", "long_name": "v"})},
        coords={"time": ("time", np.arange(2, dtype=np.float64), time_attrs)},
        attrs={"Conventions": "CF-1.8"},
    )


def _no_time_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"x_var": (("x",), np.zeros(3), {"units": "K", "long_name": "x"})},
        coords={"x": ("x", np.arange(3), {"units": "m"})},
        attrs={"Conventions": "CF-1.8"},
    )


def _without_global_attr(key: str) -> xr.Dataset:
    ds = make_cf_compliant_dataset()
    ds.attrs = {k: v for k, v in ds.attrs.items() if k != key}
    return ds


def _with_time_attr(key: str, value: str | None) -> xr.Dataset:
    """Compliant dataset with one time attr replaced (``None`` removes it)."""
    ds = make_cf_compliant_dataset()
    attrs = {k: v for k, v in ds["time"].attrs.items() if k != key}
    if value is not None:
        attrs[key] = value
    ds["time"].attrs = attrs
    return ds


def _unnamed_temperature() -> xr.Dataset:
    ds = make_cf_compliant_dataset()
    ds["temperature"].attrs = {"units": "K"}
    return ds


def _bad_time_bounds() -> xr.Dataset:
    ds = make_cf_compliant_dataset()
    ds = ds.assign(time_bnds=xr.DataArray(np.zeros((3, 5)), dims=("time", "extra")))
    ds["time"].attrs = dict(ds["time"].attrs)
    ds["time"].attrs["bounds"] = "time_bnds"
    return ds


# (row id, dataset, check id, severity, substrings of the finding message)
_ONE_FINDING: tuple[tuple[str, Callable[[], xr.Dataset], str, CFSeverity, tuple[str, ...]], ...] = (
    (
        "test_cf001_missing_conventions_error",
        lambda: make_broken_dataset("conventions"),
        "CF001",
        CFSeverity.error,
        (),
    ),
    (
        "test_cf002_missing_title_warning",
        lambda: _without_global_attr("title"),
        "CF002",
        CFSeverity.warning,
        (),
    ),
    (
        "test_cf003_missing_recommended_attrs_warning",
        make_cf_compliant_dataset,
        "CF003",
        CFSeverity.warning,
        ("institution", "source", "history"),
    ),
    ("test_cf004_no_time_coord_error", _no_time_dataset, "CF004", CFSeverity.error, ()),
    (
        "test_cf005_bad_time_units_error",
        lambda: _with_time_attr("units", "foo bar baz"),
        "CF005",
        CFSeverity.error,
        (),
    ),
    (
        "test_cf006_missing_calendar_warning",
        lambda: _with_time_attr("calendar", None),
        "CF006",
        CFSeverity.warning,
        (),
    ),
    (
        "test_cf007_missing_axis_T_info",
        lambda: _with_time_attr("axis", None),
        "CF007",
        CFSeverity.info,
        (),
    ),
    ("test_cf008_no_latitude_warning", _time_only_dataset, "CF008", CFSeverity.warning, ()),
    ("test_cf009_no_longitude_warning", _time_only_dataset, "CF009", CFSeverity.warning, ()),
    (
        "test_cf010_missing_var_units_error",
        lambda: make_broken_dataset("var_units"),
        "CF010",
        CFSeverity.error,
        ("temperature",),
    ),
    (
        "test_cf011_missing_long_or_standard_name_warning",
        _unnamed_temperature,
        "CF011",
        CFSeverity.warning,
        (),
    ),
    (
        "test_cf012_broken_reference_error",
        lambda: make_broken_dataset("broken_reference"),
        "CF012",
        CFSeverity.error,
        ("nonexistent_var",),
    ),
    ("test_cf013_bad_bounds_shape_error", _bad_time_bounds, "CF013", CFSeverity.error, ()),
    (
        "test_cf014_bad_variable_name_warning",
        lambda: make_cf_compliant_dataset().rename({"temperature": "1bad-name"}),
        "CF014",
        CFSeverity.warning,
        (),
    ),
)


def test_cf_check_reports_one_finding(subtests: Any, tmp_path: Path) -> None:
    for row_id, build, check_id, severity, message_parts in _ONE_FINDING:
        with table_row(subtests, row_id, tmp_path):
            report = validate_cf18(build(), "p", "g")

            findings = _by_id(report, check_id)
            assert len(findings) == 1
            assert findings[0].severity is severity
            for part in message_parts:
                assert part in findings[0].message


# (row id, dataset, check ids that must not report)
_NO_FINDING: tuple[tuple[str, Callable[[], xr.Dataset], tuple[str, ...]], ...] = (
    ("test_cf001_compliant_no_finding", make_cf_compliant_dataset, ("CF001",)),
    (
        "test_cf004_legacy_timestamp_with_standard_name_passes",
        make_legacy_timestamp_dataset,
        ("CF004", "CF005"),
    ),
    ("test_cf008_2d_aux_lat_passes", _make_2d_lat_lon_dataset, ("CF008", "CF009")),
    ("test_cf015_monotonic_coord_no_finding", make_cf_compliant_dataset, ("CF015",)),
)


def test_cf_check_reports_nothing(subtests: Any, tmp_path: Path) -> None:
    for row_id, build, check_ids in _NO_FINDING:
        with table_row(subtests, row_id, tmp_path):
            report = validate_cf18(build(), "p", "g")

            for check_id in check_ids:
                assert _ids(report, check_id) == []


def test_cf005_missing_time_units_error_via_fixture() -> None:
    ds = make_broken_dataset("time_units")
    report = validate_cf18(ds, "p", "g")

    findings = _by_id(report, "CF005")
    assert len(findings) >= 1


def test_cf010_exempts_grid_mapping_container() -> None:
    """A variable referenced via `grid_mapping` is a CF grid-mapping container —
    it carries the CRS in attributes and has no units by design, so CF010 must
    not flag it. A unitless var that is NOT referenced still errors (the
    exemption stays narrow)."""
    ds = xr.Dataset(
        {
            "temperature": (
                ("time", "y", "x"),
                np.zeros((1, 2, 2)),
                {"units": "K", "long_name": "T", "grid_mapping": "crs"},
            ),
            "crs": ((), np.int32(0), {"grid_mapping_name": "latitude_longitude"}),
            "no_units_var": (("time", "y", "x"), np.zeros((1, 2, 2)), {"long_name": "x"}),
        },
        coords={
            "time": (
                "time",
                np.array([0.0]),
                {"standard_name": "time", "units": "days since 2000-01-01"},
            ),
        },
        attrs={"Conventions": "CF-1.8"},
    )

    report = validate_cf18(ds, "p", "g")
    flagged = {f.target for f in _by_id(report, "CF010")}

    assert "crs" not in flagged, "grid-mapping container must be exempt from CF010"
    assert "no_units_var" in flagged, "non-referenced unitless var must still error"


def test_cf015_non_monotonic_coord_warning() -> None:
    ds = xr.Dataset(
        {"v": (("y",), np.zeros(4), {"units": "K", "long_name": "v"})},
        coords={
            "y": (
                "y",
                np.array([10.0, 20.0, 5.0, 30.0]),
                {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"},
            )
        },
        attrs={"Conventions": "CF-1.8"},
    )
    report = validate_cf18(ds, "p", "g")

    findings = _by_id(report, "CF015")
    assert any(f.target == "y" for f in findings)
    assert all(f.severity is CFSeverity.warning for f in findings)


def test_aggregation_multiple_findings() -> None:
    ds = make_cf_compliant_dataset()
    ds.attrs = {}
    ds["temperature"].attrs = {}
    report = validate_cf18(ds, "p", "g")

    assert report.summary.errors >= 1
    assert report.summary.warnings >= 1
    ids = {f.id for f in report.findings}
    assert "CF001" in ids
    assert "CF010" in ids
    assert "CF011" in ids
