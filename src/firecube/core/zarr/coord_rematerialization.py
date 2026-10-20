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

"""Refusals of re-materializing an existing time coordinate, checked before any write.

The read-only half of coordinate materialization: given a store that may
already hold a coordinate array, decide whether materializing the incoming
axis over it would change what the stored values mean, and raise
``SchemaDriftError`` (``CoordinateRelabelError`` for a relabel) before anything
is written. The ``check_*`` entry points let a caller that materializes several
groups refuse before the first group is written; the materialize functions run
the same checks themselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from firecube.core.errors import ConfigurationError, SchemaDriftError
from firecube.core.index_resolve import (
    _compute_group_identity_hash as compute_group_identity_hash,
)
from firecube.core.index_spec import effective_regular_time_policy
from firecube.core.time.calendar import is_gregorian_axis
from firecube.core.time.convert import coord_to_datetime64
from firecube.core.time.encoding import (
    CoordinateEncoding,
    coordinate_encoding_for,
    coordinate_encoding_from_array,
)
from firecube.core.zarr._coord_common import (
    array_schema_mismatches,
    build_regular_coord_attrs,
    existing_array,
    values_all_nat,
)
from firecube.core.zarr._coord_lifecycle import assert_coord_markers_consistent
from firecube.core.zarr._fill import _array_is_all_fill
from firecube.core.zarr._reserved_attrs import FIRECUBE_GROUP_IDENTITY_HASH_ATTR
from firecube.core.zarr._sealing_markers import ATTR_COORD_MANAGED, ATTR_PREALLOCATED


class CoordinateRelabelError(SchemaDriftError):
    """Re-materializing a time coordinate would change what its stored values mean.

    An encoded value means an instant only together with its ``units`` and
    ``calendar``. The store is left as it was.

    Attributes:
        coord_path: The coordinate array, ``{group}/{coordinate}``.
        stored: The stored dtype, units, calendar and identity hash, as the sentence names them.
        incoming: The same facts for the incoming axis.
        differs_in: The conflicting fields: ``dtype``, ``units``, ``calendar``, the hash attr.
    """

    def __init__(
        self, coord_path: str, *, stored: str, incoming: str, differs_in: Iterable[str]
    ) -> None:
        """Build the error and its sentence.

        Args:
            coord_path: The coordinate array, ``{group}/{coordinate}``.
            stored: The stored dtype, units, calendar and identity hash.
            incoming: The same facts for the incoming axis.
            differs_in: The conflicting fields.
        """
        self.coord_path = coord_path
        self.stored = stored
        self.incoming = incoming
        self.differs_in = tuple(differs_in)
        super().__init__(
            f"coordinate array {coord_path} holds time values stored as {stored}; the "
            f"incoming axis would store {incoming}. Rewriting it would change what the "
            f"stored values mean (differs in: {', '.join(self.differs_in)}). Declare the axis "
            "the store was written with, or write the new axis to a new target."
        )


def _coord_is_established(existing: Any, stored_attrs: Mapping[str, Any]) -> bool:
    """Return whether an existing coordinate array already carries meaning.

    Established means sealed or managed (``ATTR_PREALLOCATED`` or
    ``ATTR_COORD_MANAGED``), stamped with a group identity hash, or holding
    at least one non-fill value. A bare all-fill shell is not established.
    """
    if bool(stored_attrs.get(ATTR_PREALLOCATED, False)):
        return True
    if bool(stored_attrs.get(ATTR_COORD_MANAGED, False)):
        return True
    if stored_attrs.get(FIRECUBE_GROUP_IDENTITY_HASH_ATTR) is not None:
        return True
    return not _array_is_all_fill(np.asarray(existing[:]), existing.fill_value)


def _involves_encoded_storage(existing: Any, encoding: CoordinateEncoding) -> bool:
    """Return whether *existing* or the incoming *encoding* stores encoded (calendar) time."""
    if encoding.extra_attrs:
        return True
    stored = coordinate_encoding_from_array(np.dtype(existing.dtype), dict(existing.attrs))
    return bool(stored.extra_attrs)


def _coordinate_relabel_conflicts(
    existing: Any, *, encoding: CoordinateEncoding, group_identity_hash: str
) -> dict[str, tuple[Any, Any]]:
    """Return how re-materializing *existing* under *encoding* would change its meaning.

    Applies only when the stored array or the incoming axis is encoded
    (non-Gregorian) and the stored array is established (see
    `_coord_is_established`); Gregorian-to-Gregorian re-runs and fresh
    all-fill shells return no conflicts. An encoded value is only meaningful
    together with its ``units`` and ``calendar``, so an unchanged number
    under a different epoch or calendar names a different instant.

    Returns:
        Mapping of conflicting field (``"dtype"``, ``"units"``,
        ``"calendar"``, the group identity hash attr) to ``(stored,
        incoming)``; empty when nothing conflicts. ``"dtype"`` is present
        when the storage kinds differ, in which case stored and incoming
        values cannot be compared slot by slot.
    """
    if not _involves_encoded_storage(existing, encoding):
        return {}
    stored_attrs = dict(existing.attrs)
    stored_dtype = np.dtype(existing.dtype)
    if not _coord_is_established(existing, stored_attrs):
        return {}
    conflicts: dict[str, tuple[Any, Any]] = {}
    if stored_dtype.kind != encoding.dtype.kind:
        conflicts["dtype"] = (str(stored_dtype), str(encoding.dtype))
    for key in ("units", "calendar"):
        stored_value = stored_attrs.get(key)
        incoming_value = encoding.extra_attrs.get(key)
        if stored_value != incoming_value:
            conflicts[key] = (stored_value, incoming_value)
    stored_hash = stored_attrs.get(FIRECUBE_GROUP_IDENTITY_HASH_ATTR)
    if stored_hash is not None and stored_hash != group_identity_hash:
        conflicts[FIRECUBE_GROUP_IDENTITY_HASH_ATTR] = (stored_hash, group_identity_hash)
    return conflicts


def _refuse_coordinate_relabel(
    coord_path: str,
    existing: Any,
    *,
    encoding: CoordinateEncoding,
    group_identity_hash: str,
    conflicts: Mapping[str, tuple[Any, Any]],
) -> None:
    """Raise when *conflicts* is non-empty; the store has not been touched.

    Raises:
        CoordinateRelabelError: Naming the stored and incoming encoding of
            *coord_path* and every conflicting field.
    """
    if not conflicts:
        return

    def describe(dtype: Any, attrs: Mapping[str, Any], identity_hash: Any) -> str:
        short_hash = None if identity_hash is None else str(identity_hash)[:16]
        return (
            f"dtype={np.dtype(dtype)!s} units={attrs.get('units')!r} "
            f"calendar={attrs.get('calendar')!r} identity_hash={short_hash}"
        )

    stored_attrs = dict(existing.attrs)
    stored = describe(
        existing.dtype, stored_attrs, stored_attrs.get(FIRECUBE_GROUP_IDENTITY_HASH_ATTR)
    )
    incoming = describe(encoding.dtype, encoding.extra_attrs, group_identity_hash)
    raise CoordinateRelabelError(
        coord_path, stored=stored, incoming=incoming, differs_in=tuple(conflicts)
    )


def _check_existing_irregular_coord(
    coord_path: str,
    existing: Any,
    *,
    encoding: CoordinateEncoding,
    coord_data: np.ndarray,
    group_identity_hash: str,
) -> bool:
    """Run every refusal of re-materializing an irregular coordinate; write nothing.

    Returns:
        Whether the stored values already equal *coord_data*. ``False`` means
        the stored array is entirely fill and may be filled.

    Raises:
        SchemaDriftError: See `materialize_irregular_coord_array`.
    """
    mismatches = array_schema_mismatches(
        existing=existing,
        expected_shape=(len(coord_data),),
        expected_dtype=encoding.dtype,
        expected_chunks=None,
    )
    if mismatches:
        diff = "; ".join(mismatches)
        raise SchemaDriftError(
            f"Existing coord array mismatch: '{coord_path}' has mismatches.\n"
            f"Mismatch: {diff}\n"
            "Either delete it or update the plugin's IrregularTimeAxis to match."
        )
    assert_coord_markers_consistent(dict(existing.attrs), coord_path)
    if bool(existing.attrs.get(ATTR_COORD_MANAGED, False)):
        raise SchemaDriftError(
            f"Existing coord array '{coord_path}' carries {ATTR_COORD_MANAGED}; "
            "refusing to seal engine-managed observed coordinates as an irregular axis."
        )
    relabel_conflicts = _coordinate_relabel_conflicts(
        existing, encoding=encoding, group_identity_hash=group_identity_hash
    )
    existing_values = np.asarray(existing[:])
    values_match = bool(np.array_equal(existing_values, coord_data))
    if encoding.extra_attrs:
        all_fill = bool(np.all(encoding.is_fill(existing_values)))
    else:
        all_fill = values_all_nat(existing_values)
    if not values_match and not all_fill:
        raise SchemaDriftError(
            f"Existing coord array '{coord_path}' has values that differ from the "
            "resolved IrregularTimeAxis. Delete it or align the plugin's axis values."
        )
    _refuse_coordinate_relabel(
        coord_path,
        existing,
        encoding=encoding,
        group_identity_hash=group_identity_hash,
        conflicts=relabel_conflicts,
    )
    return values_match


def check_irregular_coord_rematerialization(
    *,
    root: Any | None,
    group_name: str,
    axis: Any,
    spec: Any | None = None,
) -> None:
    """Raise what `materialize_irregular_coord_array` would refuse, writing nothing.

    Lets a caller that materializes several groups refuse before the first
    group is written. Only coordinates where the stored array or the incoming
    axis is encoded (non-Gregorian) are checked; a Gregorian-to-Gregorian
    re-run is left to `materialize_irregular_coord_array` and keeps its
    group-by-group order. Reads only the 1-D coordinate array and its attrs.

    Args:
        root: The store's root group, or ``None`` when the store does not
            exist yet. May be opened read-only.
        group_name: The group holding the coordinate.
        axis: The incoming ``IrregularTimeAxis``.
        spec: The plugin's ``ZarrArraySpec`` for the coordinate, or ``None``.

    Raises:
        ConfigurationError: If *spec* contradicts a calendar axis (see
            `coordinate_encoding_for`).
        SchemaDriftError: Exactly as `materialize_irregular_coord_array`
            would, before its first write.
    """
    coord_path = f"{group_name}/{axis.coordinate}"
    encoding = coordinate_encoding_for(axis, spec, coord_path=coord_path)
    existing = None if root is None else existing_array(root, coord_path)
    if existing is None or not _involves_encoded_storage(existing, encoding):
        return
    _check_existing_irregular_coord(
        coord_path,
        existing,
        encoding=encoding,
        coord_data=encoding.encode_values(axis.values),
        group_identity_hash=compute_group_identity_hash(axis, len(axis.values), encoding.dtype),
    )


def _plan_encoded_grid_fill(
    *,
    coord_path: str,
    existing: Any,
    encoding: CoordinateEncoding,
    values: np.ndarray,
    group_identity_hash: str,
) -> tuple[np.ndarray, int]:
    """Run every refusal of filling an existing encoded grid coordinate; write nothing.

    Storage-kind conflicts first (the values are not comparable), then
    per-slot divergence, then the remaining encoding and identity-hash
    conflicts (see `_coordinate_relabel_conflicts`).

    Returns:
        The stored values with every fill slot replaced by its nominal value,
        and how many slots that replaced.
    """
    if bool(existing.attrs.get(ATTR_COORD_MANAGED, False)):
        raise SchemaDriftError(
            f"coordinate array {coord_path} carries {ATTR_COORD_MANAGED}: calendar axes "
            "only support the grid policy, so an engine-managed observed shell here "
            "means the store was edited out of band or the schema changed underneath it."
        )
    relabel_conflicts = _coordinate_relabel_conflicts(
        existing, encoding=encoding, group_identity_hash=group_identity_hash
    )
    if "dtype" in relabel_conflicts:
        _refuse_coordinate_relabel(
            coord_path,
            existing,
            encoding=encoding,
            group_identity_hash=group_identity_hash,
            conflicts=relabel_conflicts,
        )
    stored_full = np.asarray(existing[:])
    filled = 0
    for slot_index, (stored_value, incoming_value) in enumerate(
        zip(stored_full.flat, values.flat, strict=True)
    ):
        if encoding.scalar_is_fill(stored_value):
            stored_full[slot_index] = incoming_value
            filled += 1
            continue
        if not encoding.values_equal(stored_value, incoming_value):
            raise SchemaDriftError(
                f"coordinate array {coord_path} diverged from nominal grid at slot "
                f"{slot_index}: stored {stored_value!r}, incoming {incoming_value!r}"
            )
    _refuse_coordinate_relabel(
        coord_path,
        existing,
        encoding=encoding,
        group_identity_hash=group_identity_hash,
        conflicts=relabel_conflicts,
    )
    return stored_full, filled


@dataclass(frozen=True)
class RegularCoordTarget:
    """What a regular time axis materializes to, resolved once without I/O.

    Returned by `check_regular_coord_rematerialization` and accepted by
    `materialize_regular_coord_array` so a caller that checks every group
    before writing any of them computes each group's nominal values once.

    Attributes:
        coord_path: ``{group}/{axis.coordinate}``.
        slot_count: The axis length.
        is_calendar_axis: Whether the axis stores encoded (non-Gregorian) time.
        policy: The axis's stored-values policy, ``"grid"`` or ``"observed"``.
        encoding: The coordinate's `CoordinateEncoding`.
        values: The nominal grid values, in ``encoding.dtype``.
        attrs: The coordinate attrs (see `build_regular_coord_attrs`).
        group_identity_hash: The identity hash stamped on the coordinate.
    """

    coord_path: str
    slot_count: int
    is_calendar_axis: bool
    policy: str
    encoding: CoordinateEncoding
    values: np.ndarray
    attrs: dict[str, Any]
    group_identity_hash: str


def _resolve_regular_coord_target(
    *, group_name: str, axis: Any, spec: Any | None, resolved_index: Any | None
) -> RegularCoordTarget:
    """Resolve a regular axis's `RegularCoordTarget`; see `materialize_regular_coord_array`."""
    slot_count = axis.slot_count
    if slot_count is None:
        if resolved_index is None:
            raise ValueError(
                f"Cannot materialize RegularTimeAxis '{axis.coordinate}': slot_count is None. "
                "Declare either slot_count or end_date in the plugin's index_spec."
            )
        slot_count = int(resolved_index.size(group_name))

    coord_path = f"{group_name}/{axis.coordinate}"
    is_calendar_axis = not is_gregorian_axis(axis)
    coordinate_policy = effective_regular_time_policy(axis)

    # Checked before resolving the encoding (which touches axis.units/
    # axis.encoded_units): RegularTimeAxis construction already rejects
    # mode='floor' together with calendar, but a hand-built axis-like object
    # could still reach here without those attributes, and this guard must
    # fire before anything else looks for them.
    if is_calendar_axis and coordinate_policy != "grid":
        calendar = axis.calendar
        raise ConfigurationError(
            f"calendar={calendar!r} regular axis for group {group_name!r} resolved to "
            f"policy={coordinate_policy!r}; calendar axes only support mode='exact' "
            "(policy='grid'). This should be unreachable because RegularTimeAxis "
            "construction rejects mode='floor' together with calendar; report this as a "
            "defect if you see it."
        )

    encoding = coordinate_encoding_for(axis, spec, coord_path=coord_path)
    target_dtype = encoding.dtype

    if is_calendar_axis:
        values_int = np.arange(slot_count, dtype=np.int64) * int(axis.cadence_s)
        values = encoding.encode_values(values_int)
    else:
        epoch = coord_to_datetime64(axis.epoch)
        cadence = np.timedelta64(int(axis.cadence_s * 1e9), "ns")
        values = (epoch + np.arange(slot_count, dtype=np.int64) * cadence).astype(target_dtype)

    return RegularCoordTarget(
        coord_path=coord_path,
        slot_count=int(slot_count),
        is_calendar_axis=is_calendar_axis,
        policy=coordinate_policy,
        encoding=encoding,
        values=values,
        attrs=build_regular_coord_attrs(spec, axis),
        group_identity_hash=compute_group_identity_hash(axis, int(slot_count), target_dtype),
    )


def _check_existing_regular_coord(existing: Any, target: RegularCoordTarget) -> None:
    """Run the refusals common to every existing regular coordinate; write nothing.

    Shape and lifecycle markers for every axis; the relabel refusal for a
    Gregorian axis, which refuses only an established encoded store. A
    calendar axis's remaining refusals live in `_plan_encoded_grid_fill`.
    """
    coord_path = target.coord_path
    expected_shape = (target.slot_count,)
    if tuple(existing.shape) != expected_shape:
        raise ValueError(
            f"Preallocated coord array {coord_path!r} has shape {existing.shape}, "
            f"expected {expected_shape}. Refuse to resize silently."
        )
    assert_coord_markers_consistent(dict(existing.attrs), coord_path)
    if target.is_calendar_axis:
        return
    _refuse_coordinate_relabel(
        coord_path,
        existing,
        encoding=target.encoding,
        group_identity_hash=target.group_identity_hash,
        conflicts=_coordinate_relabel_conflicts(
            existing, encoding=target.encoding, group_identity_hash=target.group_identity_hash
        ),
    )


def check_regular_coord_rematerialization(
    *,
    root: Any | None,
    group_name: str,
    axis: Any,
    spec: Any | None,
    resolved_index: Any | None = None,
) -> RegularCoordTarget:
    """Raise what `materialize_regular_coord_array` would refuse, writing nothing.

    Lets a caller that materializes several groups refuse before the first
    group is written. Only coordinates where the stored array or the incoming
    axis is encoded (non-Gregorian) are checked; a Gregorian-to-Gregorian
    re-run is left to `materialize_regular_coord_array` and keeps its
    group-by-group order. Reads only the 1-D coordinate array and its attrs,
    and never calls the plugin: observed-value discovery stays with the
    materialize call.

    Args:
        root: The store's root group, or ``None`` when the store does not
            exist yet. May be opened read-only.
        group_name: The group holding the coordinate.
        axis: The incoming ``RegularTimeAxis``.
        spec: The plugin's ``ZarrArraySpec`` for the coordinate, or ``None``.
        resolved_index: Supplies the extent when ``axis.slot_count`` is unset.

    Returns:
        The resolved `RegularCoordTarget`, to pass to
        `materialize_regular_coord_array` as *target*.

    Raises:
        ValueError: If the extent cannot be resolved, or an encoded-involved
            existing array's shape does not match.
        ConfigurationError: See `materialize_regular_coord_array`.
        SchemaDriftError: Exactly as `materialize_regular_coord_array` would,
            before its first write.
    """
    target = _resolve_regular_coord_target(
        group_name=group_name, axis=axis, spec=spec, resolved_index=resolved_index
    )
    existing = None if root is None else existing_array(root, target.coord_path)
    if existing is None or not _involves_encoded_storage(existing, target.encoding):
        return target
    _check_existing_regular_coord(existing, target)
    if target.is_calendar_axis:
        _plan_encoded_grid_fill(
            coord_path=target.coord_path,
            existing=existing,
            encoding=target.encoding,
            values=target.values,
            group_identity_hash=target.group_identity_hash,
        )
    return target
