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

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from firecube.core.controlplane.types import (
    MAINTENANCE_KIND,
    MAINTENANCE_OP_ARCHIVE_CREATE,
    WRITTEN_TO_TARGET,
    half_open_slot_coverage,
)
from firecube.core.zarr.planning import merge_intervals

if TYPE_CHECKING:
    from firecube.core.controlplane.manager import ChunkManager
    from firecube.core.index_resolve import ResolvedIndex

_READ_ONLY_MAINTENANCE_OPS = frozenset({MAINTENANCE_OP_ARCHIVE_CREATE})

_POSITION_ERRORS = (KeyError, ValueError, IndexError, TypeError)


class UnplaceableItemError(Exception):
    """A plugin inspect failure."""


def written_slot_coverage(
    *,
    chunk_manager: ChunkManager,
    product: str,
    plugin_name: str,
    groups: Sequence[str],
    log: logging.Logger,
) -> dict[str, list[tuple[int, int]]]:
    """Return the slot intervals whose writes are durable in the target, per group.

    Only spans recorded by ``plugin_name`` count: the active spans of complete
    runs, plus the target-durable spans of interrupted runs. Malformed span
    entries and groups outside ``groups`` are skipped.

    Returns:
        A mapping with a key for every requested group, each value its sorted,
        merged half-open slot intervals and empty when nothing is covered.
    """
    complete_run_spans = [
        chunk
        for chunk in chunk_manager.list_chunks(
            product=product, chunk_type="span", meta={"plugin": plugin_name}
        )
        if chunk.status == "active"
    ]
    target_durable_spans = _target_durable_history_spans(
        chunk_manager=chunk_manager, product=product, plugin_name=plugin_name
    )
    raw = half_open_slot_coverage((*complete_run_spans, *target_durable_spans), groups)
    log.debug(
        "resume_written_filter: complete_run_spans=%d target_durable_spans=%d",
        len(complete_run_spans),
        len(target_durable_spans),
    )
    return {group: merge_intervals(intervals) for group, intervals in raw.items()}


def _target_durable_history_spans(
    *, chunk_manager: ChunkManager, product: str, plugin_name: str
) -> list[Any]:
    """Return the active ``written_to="target"`` spans of runs that did not complete.

    Empty without a history scan when every run is completed and empty once the
    product has any data-changing maintenance run. Such a run rewrites cells
    while marking replaced only the spans it names, and cross-host times
    cannot order spans against it, so one such run disables history coverage.
    """
    if all(run.status == "complete" for run in chunk_manager.list_runs(product=product)):
        return []
    spans = chunk_manager.list_chunks(
        product=product,
        chunk_type="span",
        status="active",
        include_replaced=True,
        meta={"plugin": plugin_name, "written_to": WRITTEN_TO_TARGET},
    )
    if not spans:
        return []
    maintenance_runs = chunk_manager.list_chunks(
        product=product,
        chunk_type="run",
        include_replaced=True,
        meta={"kind": MAINTENANCE_KIND},
    )
    if any(
        (run.meta or {}).get("op") not in _READ_ONLY_MAINTENANCE_OPS for run in maintenance_runs
    ):
        return []
    return spans


def active_position_group(resolved: ResolvedIndex, slot_group: str | None) -> str | None:
    groups = resolved.groups
    if not groups:
        return None
    if slot_group is not None:
        return slot_group if slot_group in groups else None
    if len(groups) == 1:
        return groups[0]
    if _groups_share_same_axis(resolved, groups):
        return groups[0]
    return None


def _groups_share_same_axis(resolved: ResolvedIndex, groups: Sequence[str]) -> bool:
    # Identity, not equality: a wrong match drops items; a missed match only disables the skip.
    axes = [resolved._spec.groups[group] for group in groups]
    return all(axis is axes[0] for axis in axes[1:])


def _covered(intervals: Sequence[tuple[int, int]], slot: int) -> bool:
    return any(start <= slot < end for start, end in intervals)


def filter_items_by_written_slots(
    items: Sequence[Any],
    *,
    resolved: ResolvedIndex,
    coverage: Mapping[str, Sequence[tuple[int, int]]],
    active_group: str,
    required_groups: Sequence[str],
    position_for_item: Callable[[Any], Any | None],
    log: logging.Logger,
) -> tuple[list[Any], int]:
    """Drop the items whose slot every required group records as written.

    The item's coordinate comes from ``position_for_item`` (``None`` keeps
    the item), its slot from ``resolved.position`` in ``active_group``, and
    a drop needs the half-open intervals of ``coverage`` to cover that slot
    in every group of ``required_groups``. An item whose placement raises
    ``UnplaceableItemError``, ``KeyError``, ``ValueError``,
    ``IndexError`` or ``TypeError`` is kept, counted, and logged once at
    WARNING. Any other error propagates.

    Returns:
        The kept items, in input order, and the number dropped.
    """
    kept: list[Any] = []
    dropped = 0
    error_counts: Counter[str] = Counter()
    first_error: Exception | None = None
    for item in items:
        try:
            coordinate = position_for_item(item)
        except UnplaceableItemError as exc:
            cause = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
            error_counts[type(cause).__name__] += 1
            if first_error is None:
                first_error = cause
            kept.append(item)
            continue
        except _POSITION_ERRORS as exc:
            error_counts[type(exc).__name__] += 1
            if first_error is None:
                first_error = exc
            kept.append(item)
            continue
        if coordinate is None:
            kept.append(item)
            continue
        try:
            slot = int(resolved.position(active_group, coordinate))
        except _POSITION_ERRORS as exc:
            error_counts[type(exc).__name__] += 1
            if first_error is None:
                first_error = exc
            kept.append(item)
            continue
        if required_groups and all(
            _covered(coverage.get(group, ()), slot) for group in required_groups
        ):
            dropped += 1
        else:
            kept.append(item)

    if first_error is not None:
        log.warning(
            "resume_written_filter: %d item(s) kept because they could not be placed "
            "on a slot; errors: %s; first error: %r",
            sum(error_counts.values()),
            dict(error_counts),
            first_error,
        )
    log.info(
        "Resume evidence: stage=resume_written_filter group=%s required_groups=%s "
        "original_count=%s filtered_count=%s dropped_count=%s",
        active_group,
        list(required_groups),
        len(items),
        len(kept),
        dropped,
    )
    return kept, dropped
