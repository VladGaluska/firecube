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

"""Encode a coordinate value onto a calendar-declared time axis.

`encode_coordinate` fast-paths Gregorian-like calendars (``"standard"``,
``"gregorian"``, ``"proleptic_gregorian"``): a ``str``, ``datetime.datetime``,
``numpy.datetime64``, ``pandas.Timestamp``, or calendar-valued value on a
Gregorian-like axis is converted straight to epoch seconds and offset
against the axis's ``units``, importing neither ``xarray`` nor ``cftime``.
A non-Gregorian calendar, or an axis ``units`` string that is not
``"seconds since "``-prefixed, falls through to xarray's CF time coder,
which brings in ``cftime`` transitively through the hard ``netcdf4``
dependency.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Literal

import numpy as np

from firecube.core.time._codecs import _xarray_time_codecs
from firecube.core.time.calendar import (
    is_calendar_valued,
    is_gregorian_like,
    normalise_calendar,
)
from firecube.core.time.gregorian import (
    _as_utc_explicit,
    _gregorian_epoch_s,
    _seconds_since_epoch_s,
    coerce_to_epoch_s,
)

__all__ = ["encode_coordinate"]


def _canonicalise_encoded_number(value: Any) -> int | float:
    """Coerce a numeric value to the canonical `encode_coordinate` result.

    Returns a Python ``int`` when the value is integral, else a Python
    ``float``. Raises `TypeError` for ``bool``/``np.bool_`` (a boolean is
    not a coordinate). Raises `ValueError` for a non-finite ``float``
    (``NaN`` or infinity).

    ``np.bool_`` must be rejected *before* the ``np.generic`` unboxing:
    ``np.bool_(True).item()`` returns Python ``True``, and ``int(True) == 1``
    would otherwise silently coerce a boolean into a valid coordinate.
    """

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"encoded coordinate must not be bool; got {value!r}")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"encoded coordinate must be finite; got {value!r}")
        if value.is_integer():
            return int(value)
        return value
    return int(value)


def _apply_mode(result: int | float, *, mode: str) -> int | float:
    """Apply ``mode="floor"`` truncation to a non-Gregorian encode result.

    Args:
        result: The canonicalised `encode_coordinate` result (xarray-encoded
            or already-encoded numeric passthrough).
        mode: ``"exact"`` returns *result* unchanged. ``"floor"`` floors a
            non-integral ``float`` result to an ``int``.

    Returns:
        *result* unchanged (``mode="exact"``, or *result* is already
        integral), or its floor as an ``int`` (``mode="floor"`` and *result*
        is a non-integral ``float``).
    """

    if mode == "floor" and isinstance(result, float):
        return math.floor(result)
    return result


def encode_coordinate(
    value: Any,
    *,
    units: str,
    calendar: str,
    mode: Literal["exact", "floor"] = "exact",
) -> int | float:
    """Encode a plugin-supplied coordinate value onto a calendar time axis.

    Implements the acceptance table for values addressing a calendar axis:

    * A calendar-valued object (see `is_calendar_valued`) whose own
      calendar normalises to the same value as *calendar* is encoded. When
      *calendar* is Gregorian-like (see `is_gregorian_like`) and *units* is
      a ``"seconds since "`` string, this goes through the no-xarray,
      no-cftime fast path; otherwise it is encoded via xarray's CF coder
      against *units* and *calendar*.
    * A calendar-valued object on a *different* calendar raises
      `ValueError` naming both calendars -- xarray would otherwise encode
      it silently, mislabeling the value.
    * ``bool`` raises `TypeError`.
    * Any other ``int``/``float``/numpy number on a non-Gregorian axis is
      taken as already encoded in the axis units and returned
      canonicalised. On a Gregorian-like axis it raises `TypeError`.
    * A ``str``, ``datetime.datetime``, ``numpy.datetime64``, or
      ``pandas.Timestamp`` value on a Gregorian-like axis declared with
      ``"seconds since "`` *units* is encoded through the same no-xarray,
      no-cftime fast path.
    * Any other Gregorian-shaped value (or a Gregorian-shaped value against
      a non-Gregorian axis, or non-``"seconds since "`` *units*) raises
      `TypeError`.

    Args:
        value: The coordinate value supplied by a plugin.
        units: The axis's CF ``units`` string.
        calendar: The axis's (already normalised) CF calendar name.
        mode: ``"exact"`` (default) or ``"floor"``. On the Gregorian fast
            path, ``"exact"`` raises for a sub-second-precision ``str``,
            ``datetime.datetime``, or ``pandas.Timestamp`` value (a
            ``numpy.datetime64`` value always floors); ``"floor"`` truncates
            instead. Off the fast path, ``"floor"`` floors a non-integral
            xarray-encoded or already-encoded numeric result to an ``int``;
            ``"exact"`` leaves it as a non-integral ``float``.

    Returns:
        The canonical encoded value: a Python ``int`` when the encoded
        number is integral, else a Python ``float``.

    Raises:
        TypeError: *value* is ``bool``, is a bare number on a Gregorian-like
            axis, or is a Gregorian-shaped value that cannot address
            *calendar*/*units* as given.
        ValueError: *value* is a calendar-valued object on a different
            calendar than *calendar*, the encoded result is not finite, or
            (``mode="exact"`` on the Gregorian fast path) *value* has
            sub-second precision.
    """

    if isinstance(value, bool):
        raise TypeError(f"encode_coordinate does not accept bool; got {value!r}")

    # ``np.bool_`` is not an instance of ``bool``, so it skips the guard
    # above; a boolean is not a coordinate on any calendar, so it gets the
    # same refusal as ``bool`` (naming the numpy type), not the
    # "Gregorian-typed value" message meant for datetime-like inputs.
    if isinstance(value, np.bool_):
        raise TypeError(f"encode_coordinate does not accept bool (np.bool_); got {value!r}")

    axis_calendar_norm = normalise_calendar(calendar)
    axis_is_gregorian = is_gregorian_like(axis_calendar_norm)

    if is_calendar_valued(value):
        value_calendar_norm = normalise_calendar(value.calendar)
        # Exact match is always accepted; a Gregorian-like value against a
        # Gregorian-like axis is also accepted even when the two spellings
        # differ (e.g. a `cftime.DatetimeGregorian`, calendar="standard",
        # addressing an axis declared calendar="proleptic_gregorian") --
        # `_GREGORIAN_LIKE_CALENDARS` already treats the three spellings as
        # the same calendar; only a genuine cross-calendar mismatch (e.g.
        # noleap on a 360_day axis) is refused.
        calendars_compatible = value_calendar_norm == axis_calendar_norm or (
            axis_is_gregorian and is_gregorian_like(value_calendar_norm)
        )
        if not calendars_compatible:
            guidance = (
                f"; declare calendar={value.calendar!r} on the time axis "
                "(RegularTimeAxis/IrregularTimeAxis, or TimeAxis.grid/.explicit/.discovered)"
                if axis_is_gregorian
                else ""
            )
            raise ValueError(
                f"coordinate calendar {value.calendar!r} does not match axis calendar "
                f"{calendar!r}{guidance}"
            )
        if axis_is_gregorian:
            axis_epoch_s = _seconds_since_epoch_s(units)
            if axis_epoch_s is not None:
                iso_text = _as_utc_explicit(value.isoformat())
                value_epoch_s = _gregorian_epoch_s(iso_text, mode=mode)
                return _canonicalise_encoded_number(value_epoch_s - axis_epoch_s)
        _, encode_cf_datetime = _xarray_time_codecs()
        encoded, _, _ = encode_cf_datetime(
            np.array([value], dtype=object), units=units, calendar=calendar
        )
        return _apply_mode(_canonicalise_encoded_number(encoded.reshape(-1)[0]), mode=mode)

    if isinstance(value, (numbers.Integral, numbers.Real)):
        if axis_is_gregorian:
            # A bare number is not a Gregorian coordinate; raise the same
            # TypeError `coerce_to_epoch_s` has always raised for it.
            coerce_to_epoch_s(value, mode=mode)
        return _apply_mode(_canonicalise_encoded_number(value), mode=mode)

    if axis_is_gregorian:
        axis_epoch_s = _seconds_since_epoch_s(units)
        if axis_epoch_s is not None:
            value_epoch_s = _gregorian_epoch_s(value, mode=mode)
            return _canonicalise_encoded_number(value_epoch_s - axis_epoch_s)

    raise TypeError(
        f"a Gregorian-typed value ({type(value).__name__}) cannot address a calendar axis "
        f"(calendar={calendar!r}, units={units!r}); pass a calendar-valued object on "
        "that calendar, a recognized Gregorian-shaped value against a Gregorian axis "
        "declared with 'seconds since ' units, or an already-encoded int/float"
    )
