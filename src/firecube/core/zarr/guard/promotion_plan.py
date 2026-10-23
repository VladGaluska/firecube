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

"""The object delta a staged run publishes: put, delete or leave alone.

A staged run writes a workspace copy of the product, then publishes it. Copying
every workspace file is wrong in one case: when a run rewrites a whole stored
unit (chunk, or shard when sharded) to the array's fill value, zarr deletes the
unit's object instead of writing it, so the workspace holds nothing to copy and
the target keeps the old bytes. This module turns what the run wrote and what
the workspace holds into a typed plan that also carries those deletions.

The plan is pure: objects are store keys (``/``-separated, relative to the
product root) and :func:`build_promotion_plan` performs no I/O. Only the
workspace is ever listed (:func:`list_workspace_objects`); the target is
consulted solely through the caller's ``is_unchanged`` callback.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "PlannedObject",
    "PromotionAction",
    "PromotionPlan",
    "build_promotion_plan",
    "list_workspace_objects",
    "unclaimed_data_objects",
]

_METADATA_NAME = "zarr.json"


class PromotionAction(StrEnum):
    """What promotion does with one object."""

    PUT = "put"
    DELETE = "delete"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class PlannedObject:
    """One object and the action promotion takes on it.

    Attributes:
        key: Store key relative to the product root, ``/``-separated.
        action: What promotion does.
        size: Workspace size in bytes for a put; ``0`` otherwise.
        reason: Short machine-readable cause (``written-unit``, ``deleted-unit``,
            ``metadata``, ``same-bytes``, ``unclaimed``, ``engine-managed``).
            ``engine-managed`` marks a data object of a managed time
            coordinate, left alone without comparing it with the target.
    """

    key: str
    action: PromotionAction
    size: int
    reason: str

    @property
    def is_metadata(self) -> bool:
        return self.key.rsplit("/", 1)[-1] == _METADATA_NAME


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    """Ordered object delta of one staged run."""

    objects: tuple[PlannedObject, ...]


def list_workspace_objects(root: Path) -> dict[str, int]:
    """Return ``{store key: size}`` of every regular file under ``root``."""
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def build_promotion_plan(
    *,
    workspace_objects: Mapping[str, int],
    written_units: Collection[str],
    absent_units: Collection[str],
    is_unchanged: Callable[[str], bool],
    managed_coordinates: Collection[str] = (),
) -> PromotionPlan:
    """Build the delta that publishes a staged run.

    Args:
        workspace_objects: ``{key: size}`` of the workspace, from
            :func:`list_workspace_objects`.
        written_units: Store key of the object of every stored unit the run
            wrote in every cell (what a write chokepoint reports, in stored
            geometry). The caller has already proved full coverage: a unit the
            run wrote only in part is refused before a plan exists.
        absent_units: The written units the write chokepoint reported absent
            (zarr omitted the object because every cell equals the fill value).
            Must be a subset of ``written_units``.
        is_unchanged: ``key -> bool``; whether publishing the workspace object
            would leave the target as it is (``zarr.json`` the target already
            supersedes, or byte-identical content). Asked only for objects
            that are not written units and not data objects of
            ``managed_coordinates``.
        managed_coordinates: ``group/array`` paths of the time coordinates
            the target marks as preallocated or coordinate-managed. The
            engine writes their values in the target, so the workspace copy
            is a stale seed.

    Raises:
        ValueError: ``absent_units`` is not a subset of ``written_units``, a
            written unit is neither in the workspace nor in ``absent_units``
            (no evidence either way, so the target's bytes cannot be judged),
            or a written unit lies under a managed coordinate.

    Returns:
        A plan. A written unit with a workspace object is a put. A written unit
        reported absent is a delete: zarr omits an all-fill unit, and the target
        must become fill. Nothing else is ever deleted: a missing object that no
        write of the run covers was never touched, and the target is never
        listed. A data object of a managed coordinate is unchanged, reason
        ``engine-managed``; the plan never asks ``is_unchanged`` about it. Every
        other workspace object (metadata, an unmanaged time coordinate,
        engine-owned, static, anything the run did not claim) is a put unless
        ``is_unchanged``. Data and delete entries come before metadata, so an
        interrupted publication never leaves a grown shape without its data.
    """
    written = set(written_units)
    absent = set(absent_units)
    stray = absent - written
    if stray:
        raise ValueError(f"absent units are not written units: {sorted(stray)[:5]}")
    unproved = written - workspace_objects.keys() - absent
    if unproved:
        raise ValueError(
            f"written units with no workspace object and not reported absent: {sorted(unproved)[:5]}"
        )
    managed = tuple(f"{path}/" for path in managed_coordinates)
    managed_documents = {f"{prefix}{_METADATA_NAME}" for prefix in managed}

    def _is_managed_data(key: str) -> bool:
        return bool(managed) and key.startswith(managed) and key not in managed_documents

    overlap = sorted(key for key in written if _is_managed_data(key))
    if overlap:
        raise ValueError(f"written units under an engine-managed coordinate: {overlap[:5]}")
    planned: list[PlannedObject] = []
    for key in sorted(written):
        if key in workspace_objects:
            planned.append(
                PlannedObject(key, PromotionAction.PUT, workspace_objects[key], "written-unit")
            )
        else:
            planned.append(PlannedObject(key, PromotionAction.DELETE, 0, "deleted-unit"))
    for key, size in sorted(workspace_objects.items()):
        if key in written:
            continue
        if _is_managed_data(key):
            planned.append(PlannedObject(key, PromotionAction.UNCHANGED, 0, "engine-managed"))
            continue
        is_metadata = key.rsplit("/", 1)[-1] == _METADATA_NAME
        if is_unchanged(key):
            planned.append(
                PlannedObject(
                    key, PromotionAction.UNCHANGED, 0, "metadata" if is_metadata else "same-bytes"
                )
            )
        else:
            planned.append(
                PlannedObject(
                    key, PromotionAction.PUT, size, "metadata" if is_metadata else "unclaimed"
                )
            )
    planned.sort(key=lambda obj: obj.is_metadata)  # stable: metadata last
    return PromotionPlan(tuple(planned))


def unclaimed_data_objects(plan: PromotionPlan, array_paths: Collection[str]) -> tuple[str, ...]:
    """Return the keys of unclaimed puts that are data objects of an array in *array_paths*.

    An unclaimed put changes the target although no write of the run covers
    it. For an array whose writes the run records, publishing it would copy
    bytes the run never wrote over the target.

    Args:
        plan: The plan of :func:`build_promotion_plan`.
        array_paths: ``group/array`` paths as zarr spells them (no leading,
            trailing or repeated ``/``).
    """
    prefixes = tuple(f"{path}/" for path in array_paths)
    return tuple(
        obj.key
        for obj in plan.objects
        if obj.action is PromotionAction.PUT
        and obj.reason == "unclaimed"
        and prefixes
        and obj.key.startswith(prefixes)
    )
