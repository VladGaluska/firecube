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

"""Gregorian fast path: epoch-second and ``datetime64[ns]`` conversion.

`coerce_to_epoch_s` is the general-purpose epoch-seconds converter for
``str``/``datetime``/``numpy.datetime64``/``pandas.Timestamp`` input. It has
no xarray/cftime import on its call path and performs no I/O.
`gregorian_to_datetime64_ns` refuses instants outside the ``datetime64[ns]``
range instead of wrapping.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import numpy as np

from firecube.core.errors import FirecubeError
from firecube.core.slot_index import iso_to_epoch_s
from firecube.core.time.calendar import (
    _UTC_EXPLICIT_SUFFIXES,
    is_calendar_valued,
    is_gregorian_like,
)

__all__ = ["coerce_to_epoch_s", "gregorian_to_datetime64_ns"]

_SECONDS_SINCE_PREFIX = "seconds since "


def _seconds_since_epoch_s(units: str) -> int | None:
    """Return the Gregorian axis epoch in seconds, or ``None`` if unfit.

    Only a ``"seconds since "``-prefixed *units* string is fast-pathable;
    any other CF ``units`` string returns ``None`` so the caller falls back
    to the xarray-based path.

    Args:
        units: A CF ``units`` string.

    Returns:
        The epoch in seconds since the Unix epoch, or ``None`` if *units*
        does not start with ``"seconds since "``.
    """

    if not units.startswith(_SECONDS_SINCE_PREFIX):
        return None
    epoch_body = units[len(_SECONDS_SINCE_PREFIX) :]
    epoch_iso = f"{epoch_body.replace(' ', 'T', 1)}Z"
    return iso_to_epoch_s(epoch_iso)


def _as_utc_explicit(text: str) -> str:
    """Append ``"Z"`` to *text* unless it already ends with a UTC suffix."""

    if text.endswith(_UTC_EXPLICIT_SUFFIXES):
        return text
    return f"{text}Z"


def _gregorian_epoch_s(value: Any, *, mode: str) -> int:
    """Compute epoch seconds for a value addressing a Gregorian-like axis.

    Delegates to `coerce_to_epoch_s`, so every accepted type keeps its
    established semantics: a ``str`` and a ``numpy.datetime64`` are truncated
    to whole seconds in both modes, while a ``datetime.datetime`` or
    ``pandas.Timestamp`` with a fractional second raises in ``"exact"`` mode.

    Args:
        value: A ``str``, ``datetime.datetime``, ``numpy.datetime64``, or
            ``pandas.Timestamp`` coordinate value.
        mode: ``"exact"`` or ``"floor"``.

    Returns:
        Whole seconds since the Unix epoch.

    Raises:
        TypeError: If *value* is not one of the accepted types.
        ValueError: In ``"exact"`` mode, if a ``datetime.datetime`` or
            ``pandas.Timestamp`` value has a fractional second.
    """

    return coerce_to_epoch_s(value, mode=mode)


def coerce_to_epoch_s(value: Any, *, mode: str = "floor") -> int:
    """Coerce a coordinate value to seconds since the Unix epoch (UTC).

    Accepts the following types:

    - ``str``: UTC-explicit ISO 8601 string (via ``iso_to_epoch_s``).
    - ``datetime.datetime``: naive treated as UTC (FCI pattern);
      aware converted to UTC.
    - ``numpy.datetime64``: any unit, converted to seconds.
    - ``pandas.Timestamp``: naive treated as UTC; aware converted to UTC.

    Args:
        value: The coordinate value to coerce.
        mode: ``"floor"`` (default) or ``"exact"``. In ``"exact"`` mode,
            fractional seconds raise ``ValueError``.

    Returns:
        Integer seconds since the Unix epoch (UTC).

    Raises:
        TypeError: If ``value`` is not one of the accepted types.
        ValueError: In ``"exact"`` mode, if the value has sub-second precision.
    """
    import pandas as pd  # lazy import: keep at function scope to defer import cost

    if isinstance(value, str):
        return iso_to_epoch_s(value)

    if isinstance(value, pd.Timestamp):
        if value.tz is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        ts = value.timestamp()
        if mode == "exact" and ts != int(ts):
            raise ValueError(
                f"coordinate {value!r} has sub-second precision; "
                "use mode='floor' or provide a whole-second value"
            )
        return int(ts)

    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            # Default: naive datetime treated as UTC (FCI pattern)
            value = value.replace(tzinfo=dt.UTC)
        else:
            value = value.astimezone(dt.UTC)
        ts = value.timestamp()
        if mode == "exact" and ts != int(ts):
            raise ValueError(
                f"coordinate {value!r} has sub-second precision; "
                "use mode='floor' or provide a whole-second value"
            )
        return int(ts)

    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[s]").astype("int64"))

    if is_calendar_valued(value):
        raise TypeError(
            "coordinate must be str, datetime, numpy.datetime64, or pandas.Timestamp; "
            f"got {type(value).__name__!r}; the value is on calendar {value.calendar!r}; "
            "declare calendar= on the time axis"
        )

    raise TypeError(
        "coordinate must be str, datetime, numpy.datetime64, or pandas.Timestamp; "
        f"got {type(value).__name__!r}"
    )


_NS_PER_UNIT = {
    "W": 7 * 86_400 * 10**9,
    "D": 86_400 * 10**9,
    "h": 3_600 * 10**9,
    "m": 60 * 10**9,
    "s": 10**9,
    "ms": 10**6,
    "us": 10**3,
    "ns": 1,
}
_UNITS_FINER_THAN_NS = {"ps": 10**3, "fs": 10**6, "as": 10**9}
_NS_MIN = -(2**63) + 1  # -2**63 is NaT
_NS_MAX = 2**63 - 1
_NS_RANGE_TEXT = "1677-09-21T00:12:43.145224193 to 2262-04-11T23:47:16.854775807"
_FRACTION = re.compile(r"(?<=:\d\d)\.(\d+)")
_LEADING_YEAR = re.compile(r"^\s*([+-]?\d+)")
# Years/months far beyond the ns range; bounding them first keeps NumPy's
# intermediate day/second conversions from wrapping.
_MAX_ABS_YEAR = 10_000
_MAX_ABS_MONTH = 12 * _MAX_ABS_YEAR


def gregorian_to_datetime64_ns(value: Any) -> np.datetime64:
    """Convert a Gregorian time value to ``datetime64[ns]``, refusing overflow.

    ``datetime64[ns]`` holds instants from 1677-09-21T00:12:43.145224193 to
    2262-04-11T23:47:16.854775807 UTC. NumPy wraps (or returns ``NaT``) outside
    that range, which would store a different instant than the one supplied;
    this helper raises instead. In-range values convert exactly as NumPy does.

    Args:
        value: A ``numpy.datetime64`` (any unit), ``datetime.datetime``
            (including ``pandas.Timestamp``; naive is read as UTC), ISO 8601
            ``str``, or a calendar-valued object (e.g. ``cftime``) on a
            Gregorian-like calendar. Any other object is converted through
            ``str(value)``.

    Returns:
        The instant as ``numpy.datetime64`` at nanosecond resolution; ``NaT``
        in gives ``NaT`` out.

    Raises:
        ValueError: *value* is outside the representable range (the message
            names the value and the range), is calendar-valued on a
            non-Gregorian calendar, or is not a parseable time.
    """
    if isinstance(value, np.datetime64):
        return _from_datetime64(value)
    passed = value
    if is_calendar_valued(value):
        if not is_gregorian_like(value.calendar):
            raise ValueError(
                f"value {value!r} is on calendar {value.calendar!r}, "
                "which is not a Gregorian calendar"
            )
        text = value.isoformat()
    elif isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC).replace(tzinfo=None)
        text = value.isoformat()
    else:
        text = str(value)
    return _from_text(text, value, passed)


class _Label(str):
    """Pre-formatted description of an input, shown verbatim by ``!r``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


class Datetime64RangeError(FirecubeError, ValueError):
    """A time value lies outside the ``datetime64[ns]`` range.

    The range runs from about 1677-09-21 to 2262-04-11 UTC, so the value cannot
    be stored as a Gregorian time coordinate.

    Attributes:
        value: The offending value, as the caller passed it.
    """

    def __init__(self, value: Any, shown: Any = None) -> None:
        """Build the error and its sentence.

        Args:
            value: The offending value, as the caller passed it.
            shown: What the sentence names in place of *value*, when the two differ.
        """
        self.value = value
        shown = value if shown is None else shown
        super().__init__(
            f"time value {shown!r} is outside the datetime64[ns] range "
            f"({_NS_RANGE_TEXT} UTC); it cannot be stored as a Gregorian time coordinate"
        )


def _checked_ns(total: int, shown: Any, original: Any) -> np.datetime64:
    if not _NS_MIN <= total <= _NS_MAX:
        raise Datetime64RangeError(original, shown)
    return np.datetime64(total, "ns")


def _from_text(text: str, shown: Any, original: Any) -> np.datetime64:
    """Parse ISO text with exact integer arithmetic, so nothing wraps."""
    text = text.removesuffix("Z").removesuffix("+00:00")
    fraction = _FRACTION.search(text)
    fraction_ns = 0
    if fraction is not None:
        fraction_ns = int(fraction.group(1)[:9].ljust(9, "0"))
        text = text[: fraction.start()] + text[fraction.end() :]
    year = _LEADING_YEAR.match(text)
    if year is not None and abs(int(year.group(1))) > _MAX_ABS_YEAR:
        raise Datetime64RangeError(original, shown)
    whole_seconds = np.datetime64(text, "s")
    if np.isnat(whole_seconds):
        return np.datetime64("NaT", "ns")
    return _checked_ns(int(whole_seconds.astype("int64")) * 10**9 + fraction_ns, shown, original)


def _from_datetime64(value: np.datetime64) -> np.datetime64:
    if np.isnat(value):
        return np.datetime64("NaT", "ns")
    unit, count = np.datetime_data(value.dtype)
    raw = int(value.astype("int64")) * count
    # NumPy's own repr of a huge value is not the integer the caller passed.
    value_label = _Label(f"numpy.datetime64({raw}, {unit!r})")
    original = value
    if unit in ("M", "Y"):
        if abs(raw) > (_MAX_ABS_YEAR if unit == "Y" else _MAX_ABS_MONTH):
            raise Datetime64RangeError(original, value_label)
        value = value.astype("datetime64[D]")
        unit, raw = "D", int(value.astype("int64"))
    if unit in _UNITS_FINER_THAN_NS:
        return _checked_ns(raw // _UNITS_FINER_THAN_NS[unit], value_label, original)
    return _checked_ns(raw * _NS_PER_UNIT[unit], value_label, original)
