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

"""Helpers shared by coordinate materialization and its re-materialization checks.

Array lookup and schema comparison, and the coordinate attrs each axis kind is
written with. Nothing here writes to a store.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr

from firecube.core.time.calendar import is_gregorian_axis
from firecube.core.zarr._reserved_attrs import RESERVED_ARRAY_ATTRS
from firecube.core.zarr.region_writer import RegionZarrWriter, _open_root_read_only


def open_existing_root_read_only(writer: RegionZarrWriter) -> Any | None:
    """Open the writer's store root read-only and live, or return ``None`` when it is absent.

    The root is never created, so a check that reads through it cannot write.
    """
    return _open_root_read_only(writer)


def existing_array(root: Any, array_path: str) -> Any | None:
    """Return the array at *array_path* below *root*, or ``None`` when absent or a group.

    The array is read by its full path from *root*, never by walking child
    groups. A child group reached through its parent keeps its own consolidated
    snapshot, which can hide an array created after it. Open *root* live
    (``use_consolidated=False``).
    """
    path = "/".join(part for part in array_path.split("/") if part)
    try:
        node = root[path]
    except KeyError:
        return None
    return None if isinstance(node, zarr.Group) else node


def array_schema_mismatches(
    *,
    existing: Any,
    expected_shape: tuple[int, ...],
    expected_dtype: Any,
    expected_chunks: tuple[int, ...] | None,
) -> list[str]:
    """List how *existing* differs from the expected shape, dtype and chunks; empty if none."""
    mismatches: list[str] = []

    found_shape = tuple(existing.shape)
    if found_shape != tuple(expected_shape):
        mismatches.append(f"shape: expected {tuple(expected_shape)}, found {found_shape}")

    expected_dtype_str = str(np.dtype(expected_dtype))
    found_dtype_str = str(np.dtype(existing.dtype))
    if found_dtype_str != expected_dtype_str:
        mismatches.append(f"dtype: expected {expected_dtype_str}, found {found_dtype_str}")

    if expected_chunks is not None:
        found_chunks = tuple(existing.chunks) if existing.chunks is not None else None
        if found_chunks != tuple(expected_chunks):
            mismatches.append(f"chunks: expected {tuple(expected_chunks)}, found {found_chunks}")

    return mismatches


def build_irregular_coord_attrs(spec: Any | None, axis: Any) -> dict[str, Any]:
    """Build irregular coord attrs, mirroring ``build_regular_coord_attrs``.

    Minimal defaults (``standard_name="time"``, ``axis="T"``) merged with
    ``spec.attrs`` (when provided) after stripping reserved firecube keys and
    the xarray-CF-encoding-owned keys ``units`` and ``calendar``. A
    non-Gregorian axis then gets ``units`` and ``calendar`` from the axis:
    CF encoding belongs to the axis, never to the plugin's attrs.
    """
    minimal: dict[str, Any] = {"standard_name": "time", "axis": "T"}
    merged = dict(minimal)
    if spec is not None and getattr(spec, "attrs", None) is not None:
        merged.update({k: v for k, v in spec.attrs.items() if k not in RESERVED_ARRAY_ATTRS})
        merged.pop("units", None)
        merged.pop("calendar", None)
    if not is_gregorian_axis(axis):
        merged["units"] = axis.units
        merged["calendar"] = axis.calendar
    return merged


def values_all_nat(values: np.ndarray) -> bool:
    """True when *values* is a datetime64 array with every element NaT."""
    if values.dtype.kind != "M":
        return False
    return bool(np.all(np.isnat(values)))


def build_regular_coord_attrs(spec: Any | None, axis: Any) -> dict[str, Any]:
    """Build coord attrs without injecting xarray-owned CF encoding attrs.

    A non-Gregorian axis gets ``units`` (``axis.encoded_units``, derived from
    the epoch) and ``calendar`` from the axis after the strip. CF encoding
    belongs to the axis, so a plugin-declared ``units`` or ``calendar`` is
    stripped either way.
    """
    minimal: dict[str, Any] = {"standard_name": "time", "axis": "T"}
    merged = dict(minimal)
    if spec is not None and getattr(spec, "attrs", None) is not None:
        merged.update({k: v for k, v in spec.attrs.items() if k not in RESERVED_ARRAY_ATTRS})
        merged.pop("units", None)
        merged.pop("calendar", None)
    if not is_gregorian_axis(axis):
        merged["units"] = axis.encoded_units
        merged["calendar"] = axis.calendar
    return merged
