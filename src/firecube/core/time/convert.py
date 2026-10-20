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

"""Conversion of plugin-supplied coordinate values to ``datetime64[ns]``."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from firecube.core.time.calendar import is_calendar_valued
from firecube.core.time.canonical import (
    _canonical_coordinate_value as canonical_coordinate_value,
)
from firecube.core.time.gregorian import gregorian_to_datetime64_ns
from firecube.core.time.guard import reject_non_gregorian_calendar_value


def coord_to_datetime64(value: Any) -> np.datetime64:
    """Convert a plugin-supplied coordinate value to ``datetime64[ns]``.

    Raises:
        ValueError: If *value* is calendar-valued (see
            `firecube.core.time.calendar.is_calendar_valued`) on a
            non-Gregorian calendar; see
            `firecube.core.time.guard.reject_non_gregorian_calendar_value`.
            Gregorian-like calendar values keep converting as before.
        ValueError: If a time value falls outside the ``datetime64[ns]`` range
            (about 1677-09-21 to 2262-04-11); see
            `firecube.core.time.gregorian.gregorian_to_datetime64_ns`.
    """
    reject_non_gregorian_calendar_value(value)
    if isinstance(value, str | dt.datetime | np.datetime64) or is_calendar_valued(value):
        return gregorian_to_datetime64_ns(value)
    return np.datetime64(canonical_coordinate_value(value), "ns")
