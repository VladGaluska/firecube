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

"""Canonicalisation of plugin-supplied coordinate values.

`_canonical_coordinate_value` and `_canonicalise_irregular_value` are the
bodies behind the same-named functions in ``firecube.core.index_spec`` (which
delegate here); ``firecube.ingestor.api`` aliases the first as
``canonical_coordinate_value``. The calendar-dependent imports stay inside
`_canonicalise_irregular_value` so importing this module costs only numpy.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np


def _canonical_coordinate_value(value: Any) -> Any:
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="ns", timezone="UTC")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        else:
            value = value.astimezone(dt.UTC)
        return value.isoformat().replace("+00:00", "Z")
    return value


def _canonicalise_irregular_value(value: Any, *, calendar: str, units: str | None) -> Any:
    """Canonicalise a single ``IrregularTimeAxis`` coordinate value.

    The single canonicalisation rule for every value an ``IrregularTimeAxis``
    (or its resolver, or planning-time discovery) accepts, shared by
    ``IrregularTimeAxis.__post_init__``, ``IrregularTimeResolver.position``,
    and ``index_binding._discover_auto_irregular_axis``.

    Args:
        value: A plugin-supplied or discovered coordinate value.
        calendar: The axis's already-normalised CF calendar name.
        units: The axis's CF ``units`` string. Required and used only when
            *calendar* is not Gregorian-like; ignored for a Gregorian-like
            *calendar*.

    Returns:
        For a non-Gregorian *calendar*: the ``encode_coordinate`` result (a
        Python ``int`` or ``float``). For a Gregorian-like *calendar*:
        *value* unchanged, except a calendar-valued object (e.g. a
        ``cftime`` instance) whose own calendar is also Gregorian-like,
        which is converted to ``numpy.datetime64`` at nanosecond resolution.

    Raises:
        ValueError: *value* is a calendar-valued object whose own calendar
            is not Gregorian-like and this axis's *calendar* is
            Gregorian-like -- declare ``calendar=`` and ``units=`` naming the
            value's own calendar instead. Also raised when a Gregorian
            calendar-valued object falls outside the ``datetime64[ns]`` range
            (about 1677-09-21 to 2262-04-11).
    """
    from firecube.core.time.calendar import is_calendar_valued, is_gregorian_like
    from firecube.core.time.encode import encode_coordinate
    from firecube.core.time.gregorian import gregorian_to_datetime64_ns

    if not is_gregorian_like(calendar):
        assert units is not None  # caller-guaranteed: non-Gregorian requires units
        return encode_coordinate(value, units=units, calendar=calendar)

    if is_calendar_valued(value):
        if is_gregorian_like(value.calendar):
            return gregorian_to_datetime64_ns(value)
        raise ValueError(
            f"value {value!r} is on calendar {value.calendar!r}, which this axis "
            "does not declare; declare calendar= and units= on this "
            "IrregularTimeAxis (or TimeAxis.explicit) to address it"
        )
    return value
