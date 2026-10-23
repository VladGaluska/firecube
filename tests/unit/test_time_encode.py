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

"""Tests for the ``firecube.core.time`` package (calendar names, encoding, range guard).

New module because ``firecube.core.time`` is a new package with no existing test module.
"""

from __future__ import annotations

import datetime as dt

import cftime
import numpy as np
import pandas as pd
import pytest

from firecube.core.errors import ConfigurationError
from firecube.core.index_spec import RegularTimeAxis
from firecube.core.time.calendar import (
    is_gregorian_like,
    normalise_calendar,
    validate_calendar_units,
)
from firecube.core.time.convert import coord_to_datetime64
from firecube.core.time.encode import encode_coordinate
from firecube.core.time.encoding import coordinate_encoding_for
from firecube.core.time.gregorian import Datetime64RangeError, gregorian_to_datetime64_ns
from firecube.ingestor.templates.direct_zarr import ZarrArraySpec

pytestmark = pytest.mark.unit

_UNITS = "seconds since 2049-01-01 12:00:00"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("360_Day", "360_day"),
        ("NoLeap", "noleap"),
        ("365_day", "noleap"),
        ("366_day", "all_leap"),
        ("Standard", "proleptic_gregorian"),
        (" GREGORIAN ", "proleptic_gregorian"),
    ],
)
def test_calendar_names_fold_case_and_aliases(raw: str, expected: str) -> None:
    assert normalise_calendar(raw) == expected
    assert is_gregorian_like(raw) is (expected == "proleptic_gregorian")


def test_unknown_calendar_name_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="not_a_calendar"):
        validate_calendar_units(units=_UNITS, calendar="not_a_calendar")


class TestEncodeCoordinateOnCalendarAxis:
    def test_calendar_value_encodes_to_seconds_since_the_units_epoch(self) -> None:
        value = cftime.Datetime360Day(2049, 2, 30, 12, 0, 0)
        assert encode_coordinate(value, units=_UNITS, calendar="360_day") == 5097600
        noleap = cftime.DatetimeNoLeap(2049, 1, 2)
        assert encode_coordinate(noleap, units=_UNITS, calendar="365_day") == 43200

    def test_value_of_another_calendar_is_refused_naming_both(self) -> None:
        with pytest.raises(ValueError, match="noleap") as excinfo:
            encode_coordinate(cftime.DatetimeNoLeap(2049, 1, 2), units=_UNITS, calendar="360_day")
        assert "360_day" in str(excinfo.value)

    @pytest.mark.parametrize(
        "value", ["2049-01-01T00:00:00Z", dt.datetime(2049, 1, 1), pd.Timestamp("2049-01-01")]
    )
    def test_gregorian_typed_value_is_refused(self, value: object) -> None:
        with pytest.raises(TypeError, match="Gregorian"):
            encode_coordinate(value, units=_UNITS, calendar="360_day")

    @pytest.mark.parametrize(
        ("value", "expected", "expected_type"),
        [
            (5097600, 5097600, int),
            (np.float64(5097600.0), 5097600, int),
            (5097600.5, 5097600.5, float),
        ],
    )
    def test_number_is_taken_as_already_encoded(
        self, value: object, expected: object, expected_type: type
    ) -> None:
        result = encode_coordinate(value, units=_UNITS, calendar="360_day")
        assert result == expected
        assert type(result) is expected_type

    @pytest.mark.parametrize("value", [True, np.bool_(True)])
    def test_bool_is_not_slot_one(self, value: object) -> None:
        with pytest.raises(TypeError, match="bool"):
            encode_coordinate(value, units=_UNITS, calendar="360_day")


class TestDatetime64NsRange:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2262-04-11T23:47:16.854775807", "2262-04-11T23:47:16.854775807"),
            ("1677-09-21T00:12:43.145224193", "1677-09-21T00:12:43.145224193"),
            (np.datetime64("2262-04-11", "D"), "2262-04-11"),
            (cftime.DatetimeProlepticGregorian(1677, 9, 22), "1677-09-22"),
        ],
    )
    def test_value_just_inside_either_end_converts_exactly(
        self, value: object, expected: str
    ) -> None:
        assert gregorian_to_datetime64_ns(value) == np.datetime64(expected, "ns")

    @pytest.mark.parametrize(
        "value",
        [
            "2262-04-11T23:47:16.854775808",
            "1677-09-21T00:12:43.145224192",
            "2500-01-01",
            np.datetime64("2500-01-01", "D"),
            dt.datetime(1600, 1, 1),
            cftime.DatetimeGregorian(2262, 4, 12),
        ],
        ids=["upper+1ns", "lower-1ns", "str-2500", "dt64-2500", "datetime-1600", "cftime-upper+1d"],
    )
    def test_value_just_outside_either_end_is_refused_not_wrapped(self, value: object) -> None:
        with pytest.raises(Datetime64RangeError, match=r"datetime64\[ns\] range") as excinfo:
            gregorian_to_datetime64_ns(value)
        assert isinstance(excinfo.value, ValueError)
        assert "1677-09-21" in str(excinfo.value) and "2262-04-11" in str(excinfo.value)

    def test_refusal_carries_the_offending_value(self) -> None:
        with pytest.raises(Datetime64RangeError) as excinfo:
            gregorian_to_datetime64_ns("2500-01-01")
        assert excinfo.value.value == "2500-01-01"

    def test_calendar_value_is_refused_with_declare_calendar_guidance(self) -> None:
        with pytest.raises(ValueError, match="calendar=") as excinfo:
            coord_to_datetime64(cftime.DatetimeNoLeap(2049, 1, 1))
        assert "noleap" in str(excinfo.value)


class TestCoordinateEncodingForSpec:
    @staticmethod
    def _axis(**overrides: object) -> RegularTimeAxis:
        kwargs: dict[str, object] = {
            "coordinate": "time",
            "epoch": "2049-01-01T00:00:00Z",
            "cadence_s": 86400,
            "slot_count": 90,
            "calendar": "360_day",
        }
        kwargs.update(overrides)
        return RegularTimeAxis(**kwargs)  # type: ignore[arg-type]

    def test_calendar_axis_is_int64_with_cf_attrs_and_int64_min_fill(self) -> None:
        encoding = coordinate_encoding_for(self._axis(), None)
        assert encoding.dtype == np.dtype("int64")
        assert encoding.extra_attrs == {
            "units": "seconds since 2049-01-01 00:00:00",
            "calendar": "360_day",
        }
        assert encoding.fill_value == np.iinfo(np.int64).min

    @pytest.mark.parametrize(
        ("spec_kwargs", "reason"),
        [
            ({"dtype": np.dtype("datetime64[ns]")}, "datetime64"),
            ({"dtype": np.int64, "attrs": {"calendar": "noleap"}}, "contradicts"),
            ({"dtype": np.int64, "attrs": {"units": "days since 1850-01-01"}}, "contradicts"),
        ],
    )
    def test_spec_contradicting_the_axis_is_refused(
        self, spec_kwargs: dict[str, object], reason: str
    ) -> None:
        spec = ZarrArraySpec(name="time", shape=(0,), **spec_kwargs)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match=reason):
            coordinate_encoding_for(self._axis(), spec)
