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

"""Sealing marker attribute name constants for coordinate-array lifecycle management.

These constants name the reserved Zarr array attributes that firecube stamps
during preallocate and consolidate operations. They are re-exported from
``firecube.core.api`` for use by tooling that inspects cube state.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ATTR_CONSOLIDATED_AT",
    "ATTR_COORD_MANAGED",
    "ATTR_ENGINE_OWNED",
    "ATTR_ENGINE_OWNED_COMPLETE",
    "ATTR_ENGINE_OWNED_SLOTS",
    "ATTR_PREALLOCATED",
]

ATTR_PREALLOCATED: Final[str] = "firecube_preallocated"
"""Zarr array attr stamped on a time coord array after dense preallocate materialization.

Presence of this attr (with value ``True``) means the coord array was written
dense during ``firecube zarr preallocate``. Subsequent ``write_timestamp``
calls perform an equality drift check instead of creating new chunk files.
"""

ATTR_COORD_MANAGED: Final[str] = "firecube_coord_managed"
"""Zarr array attr for an engine-managed unsealed coordinate.

Presence of this attr (with value ``True``) means the materializer may fill
NaT holes during staged seeding, while pods verify-or-error on writes. It is
mutually exclusive with ``ATTR_PREALLOCATED``.
"""

ATTR_CONSOLIDATED_AT: Final[str] = "firecube_consolidated_at"
"""Zarr array attr stamped on a time coord array after ``consolidate-time-coord``.

Value is an ISO 8601 UTC timestamp string recording when consolidation ran.
A ``ConsolidatedTimeCoord`` WAL event accompanies this attr; ``ResumeGuard``
reads that event to block further ingest on the sealed cube.
"""

ATTR_ENGINE_OWNED: Final[str] = "firecube_engine_owned"
"""Zarr array attr reserving an engine-owned per-slot array for preallocate.

Presence of this attr (with value ``True``) means ``firecube zarr preallocate``
owns the array: ingest pods never write it, and a declaration that disagrees
with it is an ownership change, refused. Stamped before any value is written;
whether the values are complete is ``ATTR_ENGINE_OWNED_COMPLETE``. It never
appears on a time coordinate array.
"""

ATTR_ENGINE_OWNED_COMPLETE: Final[str] = "firecube_engine_owned_complete"
"""Zarr array attr stamped once preallocate finished writing an engine-owned array.

Value ``True`` is written only after the values and the slot list
(``ATTR_ENGINE_OWNED_SLOTS``) are stored, and set to ``False`` before a later
preallocate changes them. Ingest pods refuse a target where it is not ``True``.
"""

ATTR_ENGINE_OWNED_SLOTS: Final[str] = "firecube_engine_owned_slots"
"""Zarr array attr listing the slots preallocate has written on an engine-owned array.

Value: sorted, merged ``[start, end)`` slot ranges. A listed slot is stored data,
even when it equals the fill value, and a later preallocate must replay the same
value there. Only unlisted slots (no source item seen yet) may still be filled.
"""
