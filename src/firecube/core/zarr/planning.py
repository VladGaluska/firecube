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

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from firecube.core.index_spec import RegularTimeAxis, effective_regular_time_policy
from firecube.core.zarr._coord_chunks import resolve_coord_chunks
from firecube.core.zarr.guard._layout import ArraySpecLike, resolve_effective_layout
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
