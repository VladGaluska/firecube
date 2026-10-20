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

"""Stored-unit ownership and staged-coverage checks for parallel Zarr writers.

A slot window must cover whole stored units (chunk, or shard when sharded) of every
pod-written time-indexed array, and a staged run must write every cell of each unit it
promotes. These functions read the stored layout and refuse; they write nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from operator import index as operator_index
from typing import Any

from firecube.core.errors import ConfigurationError, Message
from firecube.core.zarr.guard._unit_locks import canonical_array_path
from firecube.core.zarr.guard.chunk_geometry import (
    Bounds,
    incompletely_written_units,
    staged_partial_unit_refusal,
    unit_bounds,
    unit_ownership_refusal,
    units_touched,
    window_straddling_units,
    write_unit_shape,
)
from firecube.core.zarr.guard.engine_owned import _array_or_none


@dataclass(frozen=True)
class WriteFootprint:
    """The cells one write may change: an array and a box of its cells.

    Attributes:
        array_path: ``group/array`` written, relative to the store root;
            spelled zarr's one way on construction (``canonical_array_path``),
            so two spellings of one array cannot take different locks.
        box: Half-open ``(start, stop)`` per dimension of the stored array,
            any rank. An empty interval selects no cell.

    Raises:
        TypeError: A bound is not an integer.
        ValueError: A bound is negative or a stop lies before its start, or
            the path has a ``.`` or ``..`` segment.
    """

    array_path: str
    box: Bounds

    def __post_init__(self) -> None:
        box = tuple((operator_index(start), operator_index(stop)) for start, stop in self.box)
        for start, stop in box:
            if start < 0 or stop < start:
                raise ValueError(f"footprint box {box} has an interval outside 0 <= start <= stop")
        object.__setattr__(self, "array_path", canonical_array_path(self.array_path))
        object.__setattr__(self, "box", box)


def stored_write_unit(arr: Any) -> tuple[tuple[int, ...], str]:
    """Return the stored object shape a write to ``arr`` replaces: shard, else chunk."""
    return write_unit_shape(tuple(arr.chunks), getattr(arr, "shards", None))


def require_window_owns_stored_units(
    root: Any | None, array_paths: Sequence[str], window: tuple[int, int]
) -> None:
    """Refuse a slot window that holds only part of a stored chunk or shard.

    Judged on the layout actually stored, declared or not: two writers whose
    windows share one stored object rewrite it from stale reads and one of
    them loses its slots. Units wholly inside the window pass at any size.

    Args:
        root: Opened root group, or ``None`` when the store does not exist
            yet (its arrays are then created with the engine's layout).
        array_paths: ``group/array`` paths of the arrays the run writes.
            Absent arrays are skipped.
        window: Half-open ``(slot_start, slot_end)`` the run owns.

    Raises:
        ConfigurationError: One line per array whose stored units straddle
            the window, naming the unit, the window and what to do.
    """
    problems: list[str] = []
    for path in array_paths:
        arr = _array_or_none(root, path)
        if arr is None or arr.ndim == 0:
            continue
        unit_shape, unit_kind = stored_write_unit(arr)
        problems.extend(
            unit_ownership_refusal(
                array_path=path,
                unit_kind=unit_kind,
                unit_shape=unit_shape,
                unit_slots=span,
                window=window,
            )
            for span in window_straddling_units(
                time_unit=unit_shape[0], time_len=int(arr.shape[0]), window=window
            )
        )
    if problems:
        raise ConfigurationError(Message.join("\n", problems))


@dataclass(frozen=True)
class StagedSlotWrite:
    """One slot write of a staged run; becomes a box of cells once the shape is known.

    Attributes:
        array_path: ``group/array`` written.
        ts_index: The slot the write addressed, absolute: a slot counted from
            the end is resolved before it is recorded.
        rows: Region ``(start, stop)`` along the second axis (slice semantics,
            ``None`` ends allowed), or ``None`` for every row.
        channel: Region channel of a rank-4 array, or ``None`` for every channel.

    Raises:
        TypeError: ``ts_index`` is not an integer.
        ValueError: ``ts_index`` is negative.
    """

    array_path: str
    ts_index: int
    rows: tuple[int | None, int | None] | None = None
    channel: int | None = None

    def __post_init__(self) -> None:
        # One spelling per array, so two spellings of it cannot split its ledger.
        object.__setattr__(self, "array_path", canonical_array_path(self.array_path))
        # A slot counted from the end names another slot once the array grows.
        ts_index = operator_index(self.ts_index)
        if ts_index < 0:
            raise ValueError(
                f"staged slot write to {self.array_path} has ts_index {ts_index}; "
                "the ledger records absolute slots only"
            )
        object.__setattr__(self, "ts_index", ts_index)

    def bounds(self, shape: Sequence[int]) -> Bounds:
        """The cells this write covers in an array of *shape*.

        Raises:
            IndexError: The slot, or the channel, lies outside *shape*.
        """
        ts_index = _axis_index(self.ts_index, int(shape[0]))
        out: list[tuple[int, int]] = [(ts_index, ts_index + 1)]
        if len(shape) > 1:
            if self.rows is None:
                out.append((0, int(shape[1])))
            else:
                start, stop, _ = slice(*self.rows).indices(int(shape[1]))
                out.append((start, max(start, stop)))
        out.extend((0, int(n)) for n in shape[2:])
        if len(shape) == 4 and self.channel is not None:
            channel = _axis_index(self.channel, int(shape[3]))
            out[3] = (channel, channel + 1)
        return tuple(out)


def _axis_index(value: Any, length: int) -> int:
    """An integer selection as zarr applies it: negative counts from the end.

    Raises:
        IndexError: The index lies outside ``[-length, length)``; a slice
            built from it would silently select nothing.
    """
    index = operator_index(value)
    normalised = index + length if index < 0 else index
    if not 0 <= normalised < length:
        raise IndexError(f"index {index} is outside an axis of length {length}")
    return normalised


def require_staged_units_fully_written(
    root: Any | None, touched: Sequence[StagedSlotWrite], written: Sequence[StagedSlotWrite]
) -> None:
    """Refuse promotion of a workspace unit the run did not write in every cell.

    Promotion copies whole workspace objects over the target, so a unit the
    run wrote only in part would reset the target's other cells to the fill
    value.

    Args:
        root: The staged workspace, opened (``None``: nothing to promote).
        touched: Every slot write the run attempted.
        written: The slot writes known to have completed.

    Raises:
        ConfigurationError: One line per array with a partly written unit.
    """
    problems: list[str] = []
    for path in sorted({write.array_path for write in touched}):
        arr = _array_or_none(root, path)
        if arr is None or arr.ndim == 0:
            continue
        shape = tuple(int(n) for n in arr.shape)
        unit_shape, unit_kind = stored_write_unit(arr)
        units = incompletely_written_units(
            array_shape=shape,
            unit_shape=unit_shape,
            touched=[w.bounds(shape) for w in touched if w.array_path == path],
            written=[w.bounds(shape) for w in written if w.array_path == path],
        )
        if units:
            problems.append(
                staged_partial_unit_refusal(
                    array_path=path,
                    unit_kind=unit_kind,
                    unit_shape=unit_shape,
                    unit_cells=unit_bounds(units[0], unit_shape, shape),
                    detail=f"{len(units)} {unit_kind}(s) are written only in part",
                )
            )
    if problems:
        raise ConfigurationError(Message.join("\n", problems))


def _touched_units_per_array(
    root: Any | None, touched: Sequence[StagedSlotWrite]
) -> Iterator[tuple[Any, tuple[int, ...], tuple[int, ...], str, set[tuple[int, ...]]]]:
    """``(array, shape, unit_shape, unit_kind, units)`` per workspace array *touched* names.

    Arrays the workspace lacks and rank-0 arrays are skipped.
    """
    for path in sorted({write.array_path for write in touched}):
        arr = _array_or_none(root, path)
        if arr is None or arr.ndim == 0:
            continue
        shape = tuple(int(n) for n in arr.shape)
        unit_shape, unit_kind = stored_write_unit(arr)
        units = {
            unit
            for write in touched
            if write.array_path == path
            for unit in units_touched(write.bounds(shape), unit_shape)
        }
        yield arr, shape, unit_shape, unit_kind, units


def _unit_object_key(arr: Any, unit: tuple[int, ...]) -> str:
    key = arr.metadata.chunk_key_encoding.encode_chunk_key(unit)
    return f"{arr.path}/{key}" if arr.path else key


def staged_unit_object_keys(root: Any | None, touched: Sequence[StagedSlotWrite]) -> frozenset[str]:
    """Store key of the object of every stored unit a staged run touched.

    Only meaningful after :func:`require_staged_units_fully_written` has
    passed: every key returned then belongs to a unit the run wrote in every
    cell of the workspace extent, whether or not zarr left an object for it
    in the workspace. A unit the workspace array ends inside is proved only
    up to that extent; see :func:`require_target_within_clipped_units`.

    Args:
        root: The staged workspace, opened (``None``: no keys).
        touched: Every slot write the run attempted.
    """
    return frozenset(
        _unit_object_key(arr, unit)
        for arr, _, _, _, units in _touched_units_per_array(root, touched)
        for unit in units
    )


def require_target_within_clipped_units(
    root: Any | None,
    touched: Sequence[StagedSlotWrite],
    target_shape: Callable[[str], Sequence[int] | None],
) -> None:
    """Refuse promotion of a unit the workspace ends inside when the target reaches further into it.

    The staged coverage check proves a unit written only within the workspace
    array's shape. Promotion puts or deletes the whole target object, so when
    the target array grew past the workspace extent along an axis that ends
    inside the unit (another writer grew it after the workspace was seeded),
    the target cells past that extent would be replaced by fill although the
    run never wrote them. Units wholly inside the workspace extent pass, as
    do arrays the target lacks, holds no longer on those axes, or holds with
    another rank (a structural difference this check does not judge).

    Limit: a target that grows between this check and the put is not
    detected; closing it needs a cross-process claim.

    Args:
        root: The staged workspace, opened (``None``: nothing to check).
        touched: Every slot write the run attempted.
        target_shape: Shape of the target array at a ``group/array`` path,
            or ``None`` when the target holds no array there. Called once per
            array that has a clipped unit, and for no other.

    Raises:
        ConfigurationError: One line per array with such a unit, naming the
            array, the unit's object key and both extents.
    """
    problems: list[str] = []
    for arr, shape, unit_shape, unit_kind, units in _touched_units_per_array(root, touched):
        clipped = sorted(
            unit
            for unit in units
            if any((i + 1) * s > n for i, s, n in zip(unit, unit_shape, shape, strict=True))
        )
        if not clipped:
            continue
        target = target_shape(arr.path)
        if target is None or len(target) != len(shape):
            continue
        target = tuple(int(n) for n in target)
        for unit in clipped:
            if any(
                (i + 1) * s > n and t > n
                for i, s, n, t in zip(unit, unit_shape, shape, target, strict=True)
            ):
                problems.append(
                    f"{arr.path}: {unit_kind} {_unit_object_key(arr, unit)} ends past the "
                    f"workspace array's extent {shape} and the target array has grown into it "
                    f"(target extent {target}) since the workspace was seeded; promotion "
                    "replaces the whole object and would erase target cells this run never "
                    "wrote. Rerun once the writer that grew the target has finished."
                )
                break
    if problems:
        raise ConfigurationError(
            "staged promotion refused, target untouched: " + "\n".join(problems)
        )


def require_planned_slots_fill_units(
    units: Sequence[tuple[str, tuple[int, ...], str, int]],
    planned_slots: frozenset[int],
    window: tuple[int, int],
) -> None:
    """Refuse a staged window whose planned slots fill only part of a unit.

    Args:
        units: ``(array_path, unit_shape, unit_kind, time_len)`` per array the
            run writes: the stored layout, or the one it will be created with.
        planned_slots: Slot of every source item the window keeps.
        window: The run's ``(slot_start, slot_end)``, for the message.

    Raises:
        ConfigurationError: One line per array with a partly planned unit.
    """
    problems: list[str] = []
    for path, unit_shape, unit_kind, time_len in units:
        size = unit_shape[0]
        for index in sorted({slot // size for slot in planned_slots if 0 <= slot < time_len}):
            first, end = index * size, min((index + 1) * size, time_len)
            missing = [slot for slot in range(first, end) if slot not in planned_slots]
            if missing:
                problems.append(
                    staged_partial_unit_refusal(
                        array_path=path,
                        unit_kind=unit_kind,
                        unit_shape=unit_shape,
                        unit_cells=((first, end),),
                        detail=Message(
                            "window [{slot_window[0]}, {slot_window[1]}) has no source item for "
                            "slots {slots}",
                            slot_window=tuple(window),
                            slots=_format_slots(missing),
                        ),
                    )
                )
                break
    if problems:
        raise ConfigurationError(Message.join("\n", problems))


def _format_slots(slots: Sequence[int], limit: int = 6) -> str:
    """``[0, 1, 2, 5]`` -> ``"0-2, 5"``; long lists are cut after ``limit`` runs."""
    runs: list[tuple[int, int]] = []
    for slot in slots:
        if runs and slot == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], slot)
        else:
            runs.append((slot, slot))
    text = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs[:limit])
    return text + (", ..." if len(runs) > limit else "")
