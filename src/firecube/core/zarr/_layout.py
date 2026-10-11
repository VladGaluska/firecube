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

"""The one definition of the chunk layout a DirectZarr array is created with.

Every site that creates a time-indexed array (``zarr preallocate``, pod
startup, the serial write path) and every planner that aligns slot windows to
an array's write units resolves the layout here, so they cannot drift apart.

Rules, in order:

* static arrays (``time_indexed=False``) keep the spec's chunks and shards;
* declared chunks are used verbatim, with the declared shards;
* the time coordinate and engine-owned rank-1 arrays with no declared chunks
  get dense chunks (``resolve_coord_chunks``): pods never write engine-owned
  arrays, and only verify a time coordinate that carries a preallocate marker;
* a rank-1 array with declared shards but no chunks, whoever writes it, gets
  the largest divisor of its shard that is at most ``min(256, slots)`` as its
  inner chunk (``_divisor_chunks``), so the chunks always fit the shard;
* any other array with declared shards but no chunks keeps the declared layout;
* any other time-indexed array is written by pods. With no declared chunks
  it keeps the chunks zarr's own heuristic picks for the whole array when
  those already hold one slot each (what firecube created before this
  default existed; nothing is defaulted and nothing warns); otherwise it gets
  one slot per chunk (time chunk 1, the other dimensions from zarr's heuristic
  for one slot), unsharded. Slot-disjoint writers then never share a chunk.

The layout applies when an array is CREATED. An existing array is opened as
stored and never compared with this default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import zarr
from zarr.storage import MemoryStore

from firecube.core.zarr._coord_chunks import resolve_coord_chunks

log = logging.getLogger("firecube.core.zarr.layout")

DEFAULT_TIME_CHUNK = 1
"""Time extent of one chunk for an undeclared array that pods write."""

LayoutSource = Literal["static", "declared", "dense", "zarr", "default"]


class ArraySpecLike(Protocol):
    # Structural view of the ingestor-owned ZarrArraySpec: core must not
    # import the ingestor layer.
    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def chunks(self) -> tuple[int, ...] | None: ...

    @property
    def shards(self) -> tuple[int, ...] | None: ...

    @property
    def time_indexed(self) -> bool: ...


@dataclass(frozen=True)
class EffectiveLayout:
    """Chunks and shards to create an array with, and where they came from.

    Attributes:
        chunks: Chunk shape to pass at creation; ``None`` lets zarr choose
            (static arrays without declared chunks, or declared shards only on
            an array of rank above 1). A rank-1 time-indexed array with declared
            shards and no chunks gets the largest divisor of its shard that is
            at most ``min(256, slots)``.
        shards: Shard shape to pass at creation, or ``None`` for unsharded.
        source: ``"static"``, ``"declared"``, ``"dense"`` (engine-written
            coordinate-like array), ``"zarr"`` (undeclared pod-written, zarr's
            whole-array chunks already one slot each) or ``"default"``
            (undeclared pod-written, one-slot default applied).
    """

    chunks: tuple[int, ...] | None
    shards: tuple[int, ...] | None
    source: LayoutSource

    @property
    def is_default(self) -> bool:
        """Whether the pod-written default was applied (the case that warns at creation)."""
        return self.source == "default"


def _source(spec: ArraySpecLike, coord_name: str | None, engine_owned: bool) -> LayoutSource:
    if not spec.time_indexed:
        return "static"
    if spec.chunks is not None:
        return "declared"
    if len(spec.shape) == 1 and (engine_owned or spec.name == coord_name):
        return "dense"
    if spec.shards is not None:
        return "declared"
    return "default"


def _divisor_chunks(shards: tuple[int, ...], time_count: int) -> tuple[int, ...]:
    # Largest divisor of the shard at most max(1, min(256, slots)): the cap
    # resolve_coord_chunks applies, lowered until it fits the shard.
    shard = int(shards[0])
    cap = max(1, min(256, int(time_count)))
    return (next(d for d in range(min(cap, shard), 0, -1) if shard % d == 0),)


def _zarr_chunks(spec: ArraySpecLike, shape: tuple[int, ...]) -> tuple[int, ...]:
    # zarr's own ``chunks="auto"`` choice, read from an array created in memory
    # through its public API: no private heuristic to track across zarr releases.
    if spec.dtype is None:
        raise TypeError(f"array {spec.name!r}: dtype is required to resolve its chunks")
    probe = zarr.create_array(
        store=MemoryStore(),
        shape=shape,
        dtype=spec.dtype,
        chunks="auto",
        shards=None,
        zarr_format=3,
    )
    return tuple(int(n) for n in probe.chunks)


def _default_chunks(spec: ArraySpecLike) -> tuple[int, ...]:
    per_slot_shape = tuple(int(dim) for dim in spec.shape[1:])
    if not per_slot_shape:
        # One slot of a rank-1 array is a scalar: nothing left to chunk.
        return (DEFAULT_TIME_CHUNK,)
    # zarr's heuristic run on the shape of ONE slot rather than the whole array.
    return (DEFAULT_TIME_CHUNK, *_zarr_chunks(spec, per_slot_shape))


def resolve_effective_layout(
    spec: ArraySpecLike,
    *,
    coord_name: str | None,
    time_count: int,
    engine_owned: bool = False,
) -> EffectiveLayout:
    """Return the chunks and shards to create ``spec``'s array with.

    Args:
        spec: The declared array spec.
        coord_name: Name of the group's time coordinate array (the plugin's
            time dimension name), or ``None`` when no array is the coordinate.
        time_count: Time extent the array is created with; sizes dense chunks and
            zarr's whole-array choice for an undeclared pod-written array.
        engine_owned: Whether the engine, not pods, writes the array.

    Returns:
        The layout to pass when creating the array. Never consulted for an
        array that already exists.
    """
    source = _source(spec, coord_name, engine_owned)
    if (
        source in ("dense", "declared")
        and spec.time_indexed
        and spec.chunks is None
        and spec.shards is not None
        and len(spec.shape) == 1
    ):
        return EffectiveLayout(
            chunks=_divisor_chunks(tuple(spec.shards), time_count),
            shards=tuple(spec.shards),
            source=source,
        )
    if source == "dense":
        return EffectiveLayout(
            chunks=resolve_coord_chunks(spec, time_count), shards=spec.shards, source=source
        )
    if source == "default":
        # zarr's choice for the whole array, what an undeclared array got before
        # this default: kept, without a warning, when it is already one slot per chunk.
        try:
            created = _zarr_chunks(spec, (int(time_count), *(int(n) for n in spec.shape[1:])))
        except ValueError:
            # zarr resolves no data type from this dtype, so it has no choice of
            # its own to keep (and could not create the array either).
            created = None
        if created is not None and created[0] == DEFAULT_TIME_CHUNK:
            return EffectiveLayout(chunks=created, shards=None, source="zarr")
        return EffectiveLayout(chunks=_default_chunks(spec), shards=None, source=source)
    return EffectiveLayout(
        chunks=None if spec.chunks is None else tuple(spec.chunks),
        shards=None if spec.shards is None else tuple(spec.shards),
        source=source,
    )


def slot_window_layout(
    spec: ArraySpecLike,
    *,
    coord_name: str | None,
    engine_owned: bool = False,
) -> EffectiveLayout | None:
    """Return the layout slot windows must align to, or ``None`` if they need not.

    Planning counterpart of :func:`resolve_effective_layout`: the same rules,
    without the time extent that only dense engine-written arrays need.

    Args:
        spec: The declared array spec.
        coord_name: Name of the group's time coordinate array, or ``None``
            when the caller does not know it. A named coordinate that
            declares no ``chunks`` does not constrain windows; one that
            declares ``chunks`` constrains them through its declared
            ``chunks`` and ``shards``. With ``None`` the coordinate is
            treated as an array pods write: a one-slot unit when it declares
            neither ``chunks`` nor ``shards``, otherwise its declared
            ``chunks`` or ``shards`` constrain windows.
        engine_owned: Whether the engine, not pods, writes the array.

    Returns:
        The declared or default layout of an array pods write, else ``None``.
    """
    if _source(spec, coord_name, engine_owned) in ("static", "dense"):
        return None
    if spec.time_indexed and spec.chunks is None and spec.shards is not None:
        # The shard is the write unit; windows never depend on inner chunks.
        return EffectiveLayout(chunks=None, shards=tuple(spec.shards), source="declared")
    # Declared layouts do not depend on the time extent. An undeclared one
    # holds one slot per chunk whatever the extent (``"zarr"`` or
    # ``"default"``); only its other dimensions do, and windows never split those.
    return resolve_effective_layout(
        spec, coord_name=coord_name, time_count=0, engine_owned=engine_owned
    )


def warn_default_layout(array_path: str, layout: EffectiveLayout) -> None:
    """Log the one warning for an array just created with the pod-written default."""
    log.warning(
        "%s: time-indexed array with no declared chunks created with chunks=%s, "
        "unsharded (one slot per chunk, so slot-disjoint writers never share one); "
        "declare chunks=... on the ZarrArraySpec to choose a layout.",
        array_path,
        layout.chunks,
    )
