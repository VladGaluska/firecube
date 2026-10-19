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

"""Pure planning helpers for slot windows and coordinate chunk extents.

Slot intervals here are half-open ``[start, end)``. ``firecube.core.controlplane.ranges``
is inclusive ``[start, end]``; the conversion happens where stored coverage is read.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol, cast

from firecube.core.errors import ConfigurationError
from firecube.core.index_spec import RegularTimeAxis, effective_regular_time_policy
from firecube.core.zarr._coord_chunks import resolve_coord_chunks
from firecube.core.zarr.guard._layout import (
    ArraySpecLike,
    resolve_effective_layout,
    slot_window_layout,
)
from firecube.core.zarr.guard.engine_owned import is_engine_owned_spec


class _ResolvedIndexLike(Protocol):
    def axis_for(self, group: str) -> object | None: ...

    def size(self, group: str) -> int: ...


class _GroupSpecLike(Protocol):
    @property
    def group(self) -> str: ...

    @property
    def arrays(self) -> Sequence[ArraySpecLike]: ...


def coord_chunk_sizes_by_group(
    schema: Sequence[_GroupSpecLike],
    resolved_index: object,
    windows_by_group: dict[str, tuple[int, int]],
) -> dict[str, int]:
    """Resolve the coordinate-array chunk extent per windowed group.

    The extent is the chunk length the group's coordinate is stored with:

    * a group without a coordinate spec, or an observed ``RegularTimeAxis``
      (``mode="floor"``), gets the dense ``resolve_coord_chunks`` chunks.
      Coordinate materialization creates both unsharded, so the declared
      shards of an observed coordinate do not change them;
    * every other axis (grid regular, irregular) gets the chunks of
      ``resolve_effective_layout``, the layout its coordinate is created with.
    """
    sizes: dict[str, int] = {}
    index = cast(_ResolvedIndexLike, resolved_index)
    schema_by_group = {group_spec.group: group_spec for group_spec in schema}
    for group in windows_by_group:
        group_spec = schema_by_group.get(group)
        if group_spec is None:
            continue
        axis = index.axis_for(group)
        coord_name = getattr(axis, "coordinate", None)
        if not isinstance(coord_name, str):
            continue
        coord_spec = next(
            (
                array_spec
                for array_spec in group_spec.arrays
                if array_spec.name == coord_name
                and array_spec.time_indexed
                and len(array_spec.shape) == 1
            ),
            None,
        )
        slot_count = int(index.size(group))
        if coord_spec is None:
            # No spec: the materializer creates the coordinate with these chunks.
            chunks: tuple[int, ...] | None = resolve_coord_chunks(None, slot_count)
        elif (
            isinstance(axis, RegularTimeAxis) and effective_regular_time_policy(axis) == "observed"
        ):
            # Preallocate leaves an observed coordinate to the materializer,
            # which stores dense chunks and ignores declared shards.
            chunks = resolve_coord_chunks(coord_spec, slot_count)
        else:
            chunks = resolve_effective_layout(
                coord_spec, coord_name=coord_name, time_count=slot_count
            ).chunks
        if not chunks:
            raise ValueError(f"group {group!r}: no chunk layout for coordinate {coord_name!r}")
        sizes[group] = int(chunks[0])
    return sizes


def engine_owned_extents_by_group(
    schema: Sequence[_GroupSpecLike],
    extents: dict[str, int],
) -> dict[str, int]:
    """Return ``{group: slot count}`` for groups holding engine-owned arrays.

    Preallocate writes an engine-owned array whole, whatever its slot window,
    so its claim covers the group's full extent ``[0, slot count)``.
    """
    return {
        group_spec.group: extents[group_spec.group]
        for group_spec in schema
        if group_spec.group in extents
        and any(is_engine_owned_spec(spec) for spec in group_spec.arrays)
    }


def resolve_per_group_slot_sizes(
    schema: Sequence[_GroupSpecLike],
    explicit: int | None,
    *,
    stored_units: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, int]:
    """Return each group's slot size: the least common multiple of its time units.

    The units are the declared chunks or shards of its time-indexed arrays that are not
    engine-owned, plus the stored chunk or shard of each pod-written array the target
    already has (``stored_units``, time extents per group). An explicit size must be a
    multiple of that least common multiple.
    """
    if explicit is not None:
        if explicit <= 0:
            raise ConfigurationError(f"--slot-size must be > 0, got {explicit}")
        explicit = int(explicit)

    per_group: dict[str, int] = {}
    for group_spec in schema:
        # The schema does not name the time coordinate, so it is planned like
        # an array pods write: a one-slot unit when it declares neither chunks
        # nor shards, otherwise its declared chunks or shards count toward the
        # slot size. That is stricter than the capability gate, which does not
        # align windows to a coordinate that declares shards but no chunks.
        layouts = [
            layout
            for arr_spec in group_spec.arrays
            # Engine-owned arrays are written whole by preallocate, never by a window.
            if not is_engine_owned_spec(arr_spec)
            and (layout := slot_window_layout(arr_spec, coord_name=None)) is not None
        ]
        time_chunks = [
            int(unit[0])
            for layout in layouts
            if (unit := layout.shards if layout.shards is not None else layout.chunks)
        ]
        time_chunks.extend(int(unit) for unit in (stored_units or {}).get(group_spec.group, ()))
        if not time_chunks:
            continue
        per_group[group_spec.group] = math.lcm(*time_chunks)

    if explicit is not None:
        invalid = {g: req for g, req in per_group.items() if explicit % req != 0}
        if invalid:
            details = "; ".join(
                f"group '{g}' needs multiple of {req}" for g, req in sorted(invalid.items())
            )
            raise ConfigurationError(f"--slot-size={explicit} not valid: {details}")
        return dict.fromkeys(per_group, explicit)
    return per_group


def chunk_aligned_remaining(
    remaining: list[tuple[int, int]], slot_size: int, total: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split each ``remaining`` interval into chunk-aligned + blocked sub-intervals.

    Returns ``(aligned, blocked)`` where:
    - ``aligned`` is the executable list of chunk-aligned intervals (Phase 3.4 behavior preserved)
    - ``blocked`` is the list of sub-intervals that cannot produce any aligned executable range:
      * pre-alignment prefix when ``start`` is misaligned (e.g. ``[(73, 1000)]`` slot=100 →
        aligned=[(100, 1000)], blocked=[(73, 100)])
      * fully-dropped interval when ``aligned_start >= clamped_end`` (e.g. ``[(950, 1000)]``
        slot=100 total=1000 → aligned=[], blocked=[(950, 1000)])

    Callers MUST treat non-empty ``blocked`` as a fail-closed condition. Phase 3.5 introduced
    this signature change to close the silent-drop bug surfaced by external review.
    """
    aligned: list[tuple[int, int]] = []
    blocked: list[tuple[int, int]] = []
    for start, end in remaining:
        if start % slot_size != 0:
            aligned_start = ((start // slot_size) + 1) * slot_size
        else:
            aligned_start = start
        clamped_end = min(end, total)
        if aligned_start < clamped_end:
            aligned.append((aligned_start, clamped_end))
            if aligned_start > start:
                blocked.append((start, aligned_start))
        else:
            blocked.append((start, clamped_end))
    return aligned, blocked


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the sorted union of half-open intervals, joining those that touch or overlap."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged: list[tuple[int, int]] = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def complement_intervals(covered: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    """Return the parts of ``[0, total)`` that the sorted, merged ``covered`` intervals leave out."""
    if total <= 0:
        return []
    result: list[tuple[int, int]] = []
    cursor = 0
    for start, end in covered:
        clamped_start = max(0, min(start, total))
        clamped_end = max(0, min(end, total))
        if clamped_start > cursor:
            result.append((cursor, clamped_start))
        cursor = max(cursor, clamped_end)
        if cursor >= total:
            break
    if cursor < total:
        result.append((cursor, total))
    return result


def iter_partitions(
    remaining: Sequence[tuple[int, int]], slot_size: int
) -> Iterator[tuple[int, int]]:
    """Yield each interval's slot_size ranges in order, one at a time, holding none of them.

    Each range starts where the previous one ended; an interval's last range ends at the
    interval's end.
    """
    for interval_start, interval_end in remaining:
        cursor = interval_start
        while cursor < interval_end:
            next_cursor = min(cursor + slot_size, interval_end)
            yield cursor, next_cursor
            cursor = next_cursor


def partition_count(remaining: Sequence[tuple[int, int]], slot_size: int) -> int:
    """Count the ranges `iter_partitions` yields for the same arguments, without generating them."""
    return sum(-(-(end - start) // slot_size) for start, end in remaining if end > start)


def _divisors(n: int) -> list[int]:
    """Return the positive divisors of a positive n in ascending order."""
    small: list[int] = []
    large: list[int] = []
    for d in range(1, math.isqrt(n) + 1):
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
    return small + large[::-1]


def smallest_executable_slot_size(
    remaining: Mapping[str, Sequence[tuple[int, int]]],
    totals: Mapping[str, int],
    units: Mapping[str, int],
    ceiling: int,
) -> int | None:
    """Return the smallest `--slot-size` whose plan has no blocked interval and fits `ceiling`.

    Every interval must also end on a multiple of its group's unit or at the group total,
    else the last window of that interval splits a stored unit at any size.

    The plan is built as `zarr slots` builds it under `--slot-size s`: groups in `units` get
    slot size s, the others keep their whole axis, and `chunk_aligned_remaining` aligns each
    group's intervals. It counts ranges arithmetically and builds none.

    Args:
        remaining: Remaining intervals of each group, sorted and disjoint.
        totals: Slot count of each group.
        units: Resolved slot unit of each schema group that has one; s must be a multiple of
            every unit. A group absent from `remaining` constrains s only through its unit.
        ceiling: Largest range count the plan may have, all groups together.

    Returns:
        The smallest such s, or None when no size gives an executable plan within `ceiling`.
    """
    fixed = 0
    for group, intervals in remaining.items():
        if group in units:
            continue
        whole = totals[group]
        aligned, blocked = chunk_aligned_remaining(list(intervals), whole, whole)
        if blocked:
            return None
        fixed += partition_count(aligned, whole)
    if fixed > ceiling:
        return None

    step = math.lcm(*units.values())
    lengths: list[int] = []
    starts: list[int] = []
    for group in units:
        for start, end in remaining.get(group, ()):
            clamped_end = min(end, totals[group])
            if start >= clamped_end:
                # `chunk_aligned_remaining` blocks this interval at every slot size.
                return None
            if clamped_end != totals[group] and clamped_end % units[group] != 0:
                # The last window ends at the interval end, inside a stored unit, at every size.
                return None
            lengths.append(clamped_end - start)
            starts.append(start)

    def count(size: int) -> int:
        return fixed + sum(-(-length // size) for length in lengths)

    # A size leaves nothing blocked exactly when it divides every interval start.
    common = math.gcd(*starts)
    if common == 0:
        # Every start is 0, so every multiple of step is aligned and the count does not grow
        # with the size: search the smallest fitting multiple, up to one range per interval.
        widest = -(-max(lengths, default=step) // step)
        if count(widest * step) > ceiling:
            return None
        low, high = 1, widest
        while low < high:
            middle = (low + high) // 2
            if count(middle * step) <= ceiling:
                high = middle
            else:
                low = middle + 1
        return low * step
    if common % step != 0:
        return None
    for factor in _divisors(common // step):
        size = factor * step
        if count(size) <= ceiling:
            return size
    return None
