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

"""Lazy access to xarray's CF time coders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _xarray_time_codecs() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Lazily import xarray's CF time coders.

    Mirrors ``firecube.core.zarr.time_decode._xarray_time_codecs`` so this
    module stays cheap to import for callers that never touch a
    non-Gregorian calendar axis.
    """

    from xarray.coding.times import decode_cf_datetime, encode_cf_datetime

    return decode_cf_datetime, encode_cf_datetime
