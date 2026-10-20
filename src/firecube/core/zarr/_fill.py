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

"""Missing-aware fill-value helpers for Zarr arrays.

``NaN`` and ``NaT`` fills are never equal to themselves under ``==``, so every
comparison of an array against its fill value goes through these helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _fill_value_is_missing(value: Any) -> bool:
    """Return True when *value* is a not-a-value sentinel for its dtype.

    Covers floating/complex ``NaN`` and datetime64/timedelta64 ``NaT`` — the fill
    sentinels that are never equal to themselves under ``==``. Dispatch is by
    numpy dtype kind so every such type is handled uniformly; all other kinds
    (ints, bytes, object, ``None``) are never "missing".
    """
    arr = np.asarray(value)
    if arr.dtype.kind in ("f", "c"):  # floating, complex
        return bool(np.isnan(arr))
    if arr.dtype.kind in ("M", "m"):  # datetime64, timedelta64
        return bool(np.isnat(arr))
    return False


def _fill_mask(arr: np.ndarray, fill_value: Any) -> np.ndarray:
    """Return the elementwise mask of *arr* positions that hold *fill_value*.

    Missing-aware, so a NaN or NaT fill is detected with ``isnan``/``isnat``
    rather than equality; every other dtype compares elementwise. This is
    the one place that compares an array against its fill value; callers
    that need a mask use it instead of writing ``arr == fill_value``.

    Args:
        arr: Array to test; object and structured dtypes are out of contract.
        fill_value: The array's declared fill value.

    Returns:
        A boolean array of ``arr``'s shape, ``True`` where the element is fill.
    """
    if _fill_value_is_missing(fill_value):
        kind = np.asarray(fill_value).dtype.kind
        if kind in ("f", "c"):
            return np.isnan(arr)
        if kind in ("M", "m"):
            return np.isnat(arr)
    return np.asarray(arr == fill_value)


def _array_is_all_fill(arr: np.ndarray, fill_value: Any) -> bool:
    """Return True iff every element of *arr* equals *fill_value* under
    missing-aware equality.

    Mirrors `_fill_value_is_missing` semantics:
      - float / complex NaN fill: every element must be NaN
      - datetime64 / timedelta64 NaT fill: every element must be NaT
      - all other dtypes (int, bool, bytes/string): plain element-wise equality

    Empty arrays return True (consistent with ``np.all([])`` semantics; an
    empty static array trivially satisfies "all elements equal fill".

    Out of contract: object dtype, structured/record dtypes. Callers MUST
    not pass these.
    """
    return bool(np.all(_fill_mask(arr, fill_value)))
