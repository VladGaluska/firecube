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

"""Validation tests for irregular index resolution."""

from __future__ import annotations

import datetime as dt

import cftime
import numpy as np
import pytest

from firecube.core.errors import ConfigurationError
from firecube.core.index_resolve import (
    IrregularTimeResolver,
    RegularTimeResolver,
    resolve_index_spec,
)
from firecube.core.index_spec import IndexSpec, IrregularTimeAxis, RegularTimeAxis


def _spec(*, coordinate: str) -> IndexSpec:
    return IndexSpec(
        name="irregular_v1",
        groups={
            "data": IrregularTimeAxis(
                coordinate=coordinate,
                values=("2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z"),
            )
        },
    )


def test_irregular_time_axis_coordinate_mismatch_raises_configuration_error() -> None:
    with pytest.raises(
        ConfigurationError, match=r"IrregularTimeAxis\.coordinate='timestamp'|IrregularTimeAxis"
    ) as exc_info:
        resolve_index_spec(_spec(coordinate="timestamp"), time_dim_name="time")

    message = str(exc_info.value)
    assert "IrregularTimeAxis.coordinate='timestamp'" in message
    assert "time_dim_name='time'" in message


def test_irregular_time_axis_coordinate_match_resolves() -> None:
    resolved = resolve_index_spec(_spec(coordinate="time"), time_dim_name="time")

    assert resolved.groups == ("data",)


_CAL_UNITS = "seconds since 2049-01-01 00:00:00"


def test_irregular_calendar_axis_positions_follow_declared_order() -> None:
    axis = IrregularTimeAxis(
        coordinate="time",
        values=[cftime.Datetime360Day(2049, 1, 3), cftime.Datetime360Day(2049, 1, 1)],
        calendar="360_day",
        units=_CAL_UNITS,
    )
    resolver = IrregularTimeResolver(axis)

    assert resolver.position(cftime.Datetime360Day(2049, 1, 3)) == 0
    assert resolver.position(cftime.Datetime360Day(2049, 1, 1)) == 1
    assert resolver.position(0) == 1
    assert resolver.coordinate(1) == 0
    with pytest.raises(ValueError, match="not present"):
        resolver.position(cftime.Datetime360Day(2049, 3, 1))


def test_regular_calendar_axis_places_hadgem3_values_on_february_30() -> None:
    axis = RegularTimeAxis(
        coordinate="time",
        epoch="2049-01-01T12:00:00Z",
        cadence_s=86400,
        slot_count=90,
        calendar="360_day",
    )
    resolver = RegularTimeResolver(axis)
    values = cftime.num2date(
        [71640.5, 71699.5, 71729.5], units="days since 1850-01-01", calendar="360_day"
    )

    assert [resolver.position(v) for v in values] == [0, 59, 89]
    assert type(resolver.coordinate(1)) is int
    with pytest.raises(ValueError, match="not cadence-aligned"):
        resolver.position(cftime.Datetime360Day(2049, 1, 1, 18))
    with pytest.raises(ValueError, match="predates epoch"):
        resolver.position(cftime.Datetime360Day(2048, 12, 30, 12))


@pytest.mark.parametrize("value", [3, 3.0, np.int64(60), np.float64(60.0)])
def test_bare_number_on_gregorian_regular_axis_is_a_type_error(value: object) -> None:
    axis = RegularTimeAxis(coordinate="t", epoch="2020-01-01T00:00:00Z", cadence_s=60, slot_count=5)

    with pytest.raises(TypeError, match="coordinate must be str, datetime"):
        RegularTimeResolver(axis).position(value)


class TestCftimeGregorianOperandOnIrregularAxis:
    @pytest.mark.parametrize(
        "stored",
        [
            (dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2, 6)),
            (np.datetime64("2020-01-01T00:00:00", "s"), np.datetime64("2020-01-02T06:00:00", "s")),
        ],
        ids=["datetime", "datetime64"],
    )
    def test_gregorian_cftime_value_matches_the_same_instant_in_the_axis(
        self, stored: tuple
    ) -> None:
        resolver = IrregularTimeResolver(IrregularTimeAxis(coordinate="t", values=stored))

        assert resolver.position(cftime.DatetimeGregorian(2020, 1, 2, 6)) == 1
        with pytest.raises(ValueError, match="not present in axis values"):
            resolver.position(cftime.DatetimeGregorian(2020, 1, 3))

    def test_cftime_lookup_skips_stored_values_outside_the_datetime64_ns_range(self) -> None:
        stored = (np.datetime64("1500-01-01", "D"), np.datetime64("2020-03-01", "D"))
        resolver = IrregularTimeResolver(IrregularTimeAxis(coordinate="t", values=stored))

        assert resolver.position(cftime.DatetimeGregorian(2020, 3, 1)) == 1
        with pytest.raises(ValueError, match="not present in axis values"):
            resolver.position(cftime.DatetimeGregorian(2020, 3, 2))

    def test_same_instant_stored_twice_is_refused_with_both_slots(self) -> None:
        from firecube.core.index_resolve import DuplicateAxisInstantError

        stored = (dt.datetime(2020, 1, 1), np.datetime64("2020-01-01", "ns"))
        resolver = IrregularTimeResolver(IrregularTimeAxis(coordinate="t", values=stored))
        operand = cftime.DatetimeGregorian(2020, 1, 1)

        with pytest.raises(DuplicateAxisInstantError, match=r"matches axis slots \[0, 1\]") as exc:
            resolver.position(operand)
        assert isinstance(exc.value, ValueError)
        assert exc.value.coordinate == operand
        assert list(exc.value.slots) == [0, 1]
