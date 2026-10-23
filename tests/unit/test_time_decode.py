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

"""Tests for firecube.core.zarr.time_decode."""

# pyright: reportMissingImports=false

import numpy as np
import pytest

from firecube.core.zarr.time_decode import (
    decode_or_passthrough,
    decode_time_array,
    missing_time_mask,
)

pytestmark = pytest.mark.unit


def test_datetime64_passthrough_preserves_resolution() -> None:
    values = np.array(["2023-12-01"], dtype="datetime64[ns]")

    out = decode_time_array(values, {})

    # Native resolution is preserved rather than coarsened to seconds.
    assert out.dtype == np.dtype("datetime64[ns]")
    assert out[0] == np.datetime64("2023-12-01", "ns")


def test_datetime64_subsecond_not_truncated() -> None:
    """Distinct sub-second timestamps must survive decode (no second-floor).

    Coverage bounds and dedup keys are derived from this output; collapsing
    sub-second values to the same second would corrupt dedup decisions.
    """
    values = np.array(
        ["2023-12-01T00:00:00.100", "2023-12-01T00:00:00.200"],
        dtype="datetime64[ns]",
    )

    out = decode_time_array(values, {})

    assert out[0] != out[1]
    assert out[0] == np.datetime64("2023-12-01T00:00:00.100", "ns")


_DECODE_DATE_ROWS = (
    # (row id, values, attrs, expected ISO dates of the decoded elements)
    (
        "float64_seconds_since_epoch",
        np.array([1701388800.0]),
        {"units": "seconds since 1970-01-01"},
        ["2023-12-01"],
    ),
    (
        "float64_days_since_2000",
        np.array([0.0, 1.0, 9.0]),
        {"units": "days since 2000-01-01", "calendar": "standard"},
        ["2000-01-01", "2000-01-02", "2000-01-10"],
    ),
    (
        "int64_seconds_since_epoch",
        np.array([1701388800], dtype="int64"),
        {"units": "seconds since 1970-01-01"},
        ["2023-12-01"],
    ),
)


def test_decode_time_array_dates(subtests) -> None:
    for row_id, values, attrs, expected in _DECODE_DATE_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            out = decode_time_array(values, attrs)
            assert [str(x)[:10] for x in out] == expected


def test_calendar_proleptic_gregorian() -> None:
    values = np.array([0.0, 1.0])

    out = decode_time_array(
        values,
        {"units": "days since 2000-01-01", "calendar": "proleptic_gregorian"},
    )

    # decode_cf_datetime selects a range-aware datetime64 resolution; assert the
    # kind rather than a fixed unit, which varies by xarray version.
    assert out.dtype.kind == "M"
    assert out[0] == np.datetime64("2000-01-01")


_DECODE_ERROR_ROWS = (
    ("no_units_raises_valueerror", np.array([1.0, 2.0]), {}, "units"),
    ("units_without_since_raises", np.array([1.0, 2.0]), {"units": "kelvin"}, "since"),
    ("unsupported_dtype_raises", np.array([1.0 + 2.0j], dtype=np.complex64), {}, r"complex64"),
)


def test_decode_time_array_errors(subtests) -> None:
    for row_id, values, attrs, match in _DECODE_ERROR_ROWS:
        with subtests.test(msg=f"test_{row_id}"), pytest.raises(ValueError, match=match):
            decode_time_array(values, attrs)


class TestDecodeOrPassthrough:
    """Six permutations of (dtype, attrs.units) covering the dispatch table."""

    def test_datetime64_no_units(self) -> None:
        values = np.array(["2023-12-01"], dtype="datetime64[ns]")
        out = decode_or_passthrough(values, {})
        assert out.dtype.kind == "M"
        assert out[0] == np.datetime64("2023-12-01", "ns")

    def test_datetime64_units_without_since(self) -> None:
        values = np.array(["2023-12-01"], dtype="datetime64[ns]")
        out = decode_or_passthrough(values, {"units": "kelvin"})
        assert out.dtype.kind == "M"

    def test_datetime64_units_with_since(self) -> None:
        values = np.array(["2023-12-01"], dtype="datetime64[ns]")
        out = decode_or_passthrough(values, {"units": "seconds since 1970-01-01"})
        assert out.dtype.kind == "M"
        assert out[0] == np.datetime64("2023-12-01", "ns")

    def test_numeric_no_units_passthrough(self) -> None:
        values = np.array([1.0, 2.0, 3.0], dtype="float64")
        out = decode_or_passthrough(values, {})
        assert out.dtype.kind == "f"
        assert out[0] == 1.0

    def test_numeric_units_without_since_raises(self) -> None:
        values = np.array([1.0, 2.0, 3.0], dtype="float64")
        with pytest.raises(ValueError, match="since"):
            decode_or_passthrough(values, {"units": "kelvin"})

    def test_numeric_units_with_since_decodes(self) -> None:
        values = np.array([0.0, 1.0], dtype="float64")
        out = decode_or_passthrough(
            values, {"units": "days since 2000-01-01", "calendar": "standard"}
        )
        assert out.dtype.kind == "M"
        assert str(out[0])[:10] == "2000-01-01"

    def test_numeric_units_with_since_but_malformed_raises(self) -> None:
        values = np.array([1.0, 2.0], dtype="float64")
        with pytest.raises(ValueError):
            decode_or_passthrough(values, {"units": "days since not-a-real-date"})


def test_360_day_values_decode_to_calendar_objects_including_february_30() -> None:
    values = np.array([0, 86400, 59 * 86400], dtype="int64")

    decoded = decode_time_array(
        values, {"units": "seconds since 2049-01-01", "calendar": "360_day"}
    )

    assert decoded.dtype.kind == "O"
    assert str(decoded[0]) == "2049-01-01 00:00:00"
    # Day 59 (0-based) of a 360_day year is February 30.
    assert decoded[2].calendar == "360_day"
    assert decoded[2].isoformat() == "2049-02-30T00:00:00"


def test_missing_time_mask_flags_nat_none_and_nan_in_every_decoded_shape() -> None:
    nat = np.array(["2024-01-01", "NaT"], dtype="datetime64[s]")
    calendar_values = decode_time_array(
        np.array([0, 86400, 172800, 259200], dtype="int64"),
        {"units": "seconds since 2049-01-01", "calendar": "360_day"},
    )
    mixed = np.array([calendar_values[0], None, float("nan"), calendar_values[3]], dtype=object)

    assert missing_time_mask(nat).tolist() == [False, True]
    assert missing_time_mask(mixed).tolist() == [False, True, True, False]
    assert missing_time_mask(np.array([1, 2], dtype="int64")).tolist() == [False, False]
