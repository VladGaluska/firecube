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

"""Calendar vocabulary for calendar-declared time axes.

Names, normalises and classifies CF calendars (``normalise_calendar``,
``is_gregorian_like``, ``is_gregorian_axis``, ``is_calendar_valued``) and
derives or validates the CF ``units`` an axis stores
(``derive_regular_axis_units``, ``validate_calendar_units``). Nothing here
imports ``cftime``; ``validate_calendar_units`` reaches xarray lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from firecube.core.time._codecs import _xarray_time_codecs

__all__ = [
    "derive_regular_axis_units",
    "is_calendar_valued",
    "is_gregorian_axis",
    "is_gregorian_like",
    "normalise_calendar",
    "validate_calendar_units",
]

_GREGORIAN_LIKE_CALENDARS = frozenset({"standard", "gregorian", "proleptic_gregorian"})
_CALENDAR_ALIASES = {
    "365_day": "noleap",
    "366_day": "all_leap",
    # Firecube stores Gregorian-like time as ``datetime64``, which is
    # proleptic Gregorian; the three CF spellings name the same encoding.
    "standard": "proleptic_gregorian",
    "gregorian": "proleptic_gregorian",
}
_UTC_EXPLICIT_SUFFIXES = ("Z", "+00:00", "-00:00")


def normalise_calendar(calendar: str) -> str:
    """Normalise a CF calendar name for comparison and storage.

    Lowercases the name and translates aliases to one canonical spelling:
    ``"365_day"`` -> ``"noleap"``, ``"366_day"`` -> ``"all_leap"``, and
    ``"standard"`` / ``"gregorian"`` -> ``"proleptic_gregorian"`` (firecube
    stores Gregorian-like time as ``datetime64``, which is proleptic
    Gregorian; dates before 1582-10-15 follow that rule, not the CF mixed
    calendar). Every other name is returned lowercased and otherwise
    unchanged.

    Args:
        calendar: A CF calendar name, e.g. ``"360_Day"`` or ``"NoLeap"``.

    Returns:
        The normalised calendar name.

    Raises:
        ValueError: If *calendar* is not a non-empty string.
    """

    if not isinstance(calendar, str) or not calendar.strip():
        raise ValueError(
            f"calendar must be a CF calendar name such as 'proleptic_gregorian' or "
            f"'360_day'; got {calendar!r} (omit calendar for Gregorian time)"
        )
    text = calendar.strip().lower()
    return _CALENDAR_ALIASES.get(text, text)


def is_gregorian_like(calendar: str) -> bool:
    """Return whether *calendar* denotes a Gregorian-shaped calendar.

    Args:
        calendar: A CF calendar name; normalised internally before the
            comparison, so callers may pass a raw or already-normalised name.

    Returns:
        ``True`` for ``"standard"``, ``"gregorian"``, or
        ``"proleptic_gregorian"`` (after normalisation); ``False`` otherwise.
    """

    return normalise_calendar(calendar) in _GREGORIAN_LIKE_CALENDARS


def is_gregorian_axis(axis: Any) -> bool:
    """Return whether *axis* addresses Gregorian (datetime64) time.

    Args:
        axis: An axis-like object; only its optional ``calendar`` attribute
            is inspected.

    Returns:
        ``True`` when *axis* has no ``calendar`` attribute, ``axis.calendar``
        is ``None``, or ``axis.calendar`` normalises to a Gregorian-like
        name; ``False`` otherwise.
    """

    calendar = getattr(axis, "calendar", None)
    return calendar is None or is_gregorian_like(calendar)


def is_calendar_valued(value: Any) -> bool:
    """Return whether *value* duck-types as a calendar-valued object.

    A calendar-valued object has a string ``calendar`` attribute and a
    callable ``isoformat`` method, the shape every ``cftime`` datetime class
    has. This module never imports ``cftime`` to perform the check.

    Args:
        value: Any candidate coordinate value.

    Returns:
        ``True`` if *value* has both a string ``calendar`` attribute and a
        callable ``isoformat`` attribute.
    """

    calendar = getattr(value, "calendar", None)
    isoformat = getattr(value, "isoformat", None)
    return isinstance(calendar, str) and callable(isoformat)


def derive_regular_axis_units(epoch: str) -> str:
    """Derive CF ``units`` for a regular axis from its UTC-explicit epoch.

    Turns ``"2049-01-01T12:00:00Z"`` into
    ``"seconds since 2049-01-01 12:00:00"``. The epoch is never parsed as a
    date (not with ``numpy``, not with ``datetime``): it is only sliced as
    text, so an epoch that is invalid in the Gregorian calendar (e.g.
    ``"1850-02-30T00:00:00Z"``) still derives units cleanly -- validity is
    the calendar's own coder's concern, checked separately by
    `validate_calendar_units`.

    Args:
        epoch: UTC-explicit ISO 8601 epoch string; must end with ``"Z"``,
            ``"+00:00"``, or ``"-00:00"``.

    Returns:
        The derived CF ``units`` string, ``"seconds since <epoch body>"``,
        with the ``"T"`` separator rewritten to a space.

    Raises:
        ValueError: If *epoch* is not a non-empty, UTC-explicit string.
    """

    if not isinstance(epoch, str) or not epoch.strip():
        raise ValueError(f"epoch must be a non-empty UTC ISO string; got {epoch!r}")
    text = epoch.strip()
    for suffix in _UTC_EXPLICIT_SUFFIXES:
        if text.endswith(suffix):
            body = text[: -len(suffix)]
            break
    else:
        raise ValueError(
            f"epoch must be UTC-explicit ('Z', '+00:00', or '-00:00' offset); got {epoch!r}"
        )
    return f"seconds since {body.replace('T', ' ')}"


def validate_calendar_units(*, units: str, calendar: str) -> None:
    """Validate that *units* and *calendar* are jointly usable.

    Asks xarray's CF decoder to decode the value ``0`` against *units* and
    *calendar*. An unknown calendar name or a reference date in *units* that
    does not exist in *calendar* (e.g. ``2049-02-31`` in ``360_day``) both
    surface here as a decode failure, which this function re-raises as a
    single, clear `ValueError`.

    Args:
        units: A CF ``units`` string, e.g.
            ``"seconds since 2049-01-01 12:00:00"``.
        calendar: A CF calendar name.

    Raises:
        ValueError: If *units* and *calendar* cannot be decoded together.
    """

    decode_cf_datetime, _ = _xarray_time_codecs()
    try:
        decode_cf_datetime(np.array([0]), units=units, calendar=calendar)
    except Exception as exc:
        raise ValueError(f"units={units!r} is not valid for calendar={calendar!r}: {exc}") from exc
