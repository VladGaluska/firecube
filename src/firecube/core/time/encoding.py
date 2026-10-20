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

"""The storage encoding of a time coordinate array, derived from its calendar.

The one place a time axis's declared calendar is turned into an on-disk
encoding: ``datetime64`` for a Gregorian axis, an encoded ``int64``/``float64``
number with CF ``units``/``calendar`` attrs for any other calendar.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from firecube.core.errors import ConfigurationError
from firecube.core.time.calendar import is_gregorian_axis
from firecube.core.time.convert import coord_to_datetime64
from firecube.core.time.encode import encode_coordinate
from firecube.core.zarr._fill import _array_is_all_fill, _fill_mask


def _validate_calendar_spec_attrs(
    spec: Any | None, *, axis_units: str, axis_calendar: str, coord_path: str
) -> None:
    """Raise when a plugin's ``ZarrArraySpec.attrs`` contradicts its calendar axis.

    Equal ``units``/``calendar`` values in ``spec.attrs`` are fine (they are
    stripped and reinstated from the axis either way, by
    `build_regular_coord_attrs`/`build_irregular_coord_attrs`). A *different*
    value is a caller-input error: the plugin declared a coordinate array
    that disagrees with its own time axis's CF encoding.

    Raises:
        ConfigurationError: If ``spec.attrs`` declares ``units`` or
            ``calendar`` different from *axis_units*/*axis_calendar*.
    """
    if spec is None or getattr(spec, "attrs", None) is None:
        return
    spec_calendar = spec.attrs.get("calendar")
    if spec_calendar is not None and spec_calendar != axis_calendar:
        raise ConfigurationError(
            f"array spec for {coord_path!r} declares attrs['calendar']={spec_calendar!r}, "
            f"which contradicts its time axis's calendar={axis_calendar!r}. Remove the "
            "attr, or make it match the axis."
        )
    spec_units = spec.attrs.get("units")
    if spec_units is not None and spec_units != axis_units:
        raise ConfigurationError(
            f"array spec for {coord_path!r} declares attrs['units']={spec_units!r}, "
            f"which contradicts its time axis's derived units={axis_units!r}. Remove the "
            "attr, or make it match the axis."
        )


def _resolve_calendar_target_dtype(
    *,
    spec: Any | None,
    calendar: str,
    values_are_integral: bool,
    coord_path: str,
) -> np.dtype[Any]:
    """Resolve and validate the on-disk dtype for an encoded calendar coordinate.

    Defaults to ``int64`` when every value is integral, else ``float64``. A
    plugin ``ZarrArraySpec`` may declare ``int64`` or ``float64`` explicitly
    (honored); any other dtype -- including ``datetime64``, which contradicts
    the axis's calendar declaration -- is a caller-input error.

    Raises:
        ConfigurationError: If ``spec.dtype`` is ``datetime64``, is neither
            ``int64`` nor ``float64``, or is ``int64`` while the resolved
            values are not all integral.
    """
    default_dtype = np.dtype("int64") if values_are_integral else np.dtype("float64")
    spec_dtype_raw = (
        spec.dtype if spec is not None and getattr(spec, "dtype", None) is not None else None
    )
    if spec_dtype_raw is None:
        return default_dtype
    spec_dtype = np.dtype(spec_dtype_raw)
    if spec_dtype.kind == "M":
        raise ConfigurationError(
            f"array spec for {coord_path!r} declares datetime64 dtype {spec_dtype!s}, but its "
            f"time axis declares calendar={calendar!r}; a calendar coordinate is stored as an "
            "encoded number, not datetime64. Declare int64 or float64 in ZarrArraySpec.dtype, "
            "or omit dtype to use the default."
        )
    if spec_dtype not in (np.dtype("int64"), np.dtype("float64")):
        raise ConfigurationError(
            f"array spec for {coord_path!r} declares dtype {spec_dtype!s}, which is not "
            f"supported for a calendar={calendar!r} coordinate; use int64 or float64."
        )
    if spec_dtype == np.dtype("int64") and not values_are_integral:
        raise ConfigurationError(
            f"array spec for {coord_path!r} declares dtype int64, but the resolved axis "
            f"values for calendar={calendar!r} are not all integral; use float64 instead."
        )
    return spec_dtype


def _encoded_fill_value(dtype: np.dtype[Any]) -> Any:
    """Return the fill sentinel for an encoded calendar coordinate's dtype.

    ``NaN`` for a float dtype, else the ``int64`` minimum -- deliberately the
    same choice `RegionZarrWriter`/``firecube.core.controlplane.deletion``
    already use for a "clearly not real data" integer sentinel.
    """
    if dtype.kind == "f":
        return float("nan")
    return int(np.iinfo(np.int64).min)


def _scalar_is_fill(value: Any, fill_value: Any) -> bool:
    """Return whether a single stored value equals its dtype's fill sentinel."""
    return _array_is_all_fill(np.atleast_1d(np.asarray(value)), fill_value)


@dataclass(frozen=True)
class CoordinateEncoding:
    """The on-disk storage encoding for a time coordinate array.

    Firecube stores a time coordinate one of two ways: Gregorian time as
    ``datetime64``, or a non-Gregorian CF calendar as an encoded
    ``int64``/``float64`` number with ``units``/``calendar`` attrs. Every
    technical choice that distinction implies -- dtype, fill sentinel, extra
    CF attrs, and how a plugin value becomes a stored one -- is captured
    once here instead of being re-derived at each call site from
    ``axis.calendar is None`` or ``dtype.kind == "M"``.

    Built by `coordinate_encoding_for` (from a time axis, at materialization
    time) or `coordinate_encoding_from_array` (from an opened array's
    dtype/attrs, at write time); both dispatch on the same rule and produce
    an equivalent instance for the same coordinate.

    Attributes:
        dtype: The array's on-disk dtype.
        fill_value: The unfilled-slot sentinel, in *dtype*.
        extra_attrs: CF attrs this encoding requires beyond a coordinate's
            usual minimal attrs -- ``{"units": ..., "calendar": ...}`` for
            an encoded axis, ``{}`` for Gregorian.
        encode_values: Convert a sequence of plugin coordinate values to a
            dense array in *dtype*.
        encode_scalar: Convert one plugin coordinate value to a stored
            scalar in *dtype*.
        is_fill: Vectorised fill-sentinel mask over a stored array.
        scalar_is_fill: Whether one stored scalar is the fill sentinel.
        values_equal: Whether two stored scalars are equal, fill-aware
            (NaT counts as equal to NaT for Gregorian; an encoded axis's
            fill is a real, directly comparable number).
    """

    dtype: np.dtype[Any]
    fill_value: Any
    extra_attrs: dict[str, Any]
    encode_values: Callable[[Sequence[Any] | np.ndarray], np.ndarray]
    encode_scalar: Callable[[Any], Any]
    is_fill: Callable[[np.ndarray], np.ndarray]
    scalar_is_fill: Callable[[Any], bool]
    values_equal: Callable[[Any, Any], bool]


def _gregorian_encoding(dtype: np.dtype[Any]) -> CoordinateEncoding:
    """Build the Gregorian (``datetime64``) `CoordinateEncoding` for *dtype*."""
    fill_value = np.array(np.datetime64("NaT", "ns"), dtype=dtype)[()]
    return CoordinateEncoding(
        dtype=dtype,
        fill_value=fill_value,
        extra_attrs={},
        encode_values=lambda values: np.array(
            [coord_to_datetime64(value) for value in values], dtype="datetime64[ns]"
        ).astype(dtype),
        encode_scalar=lambda value: np.array(coord_to_datetime64(value), dtype=dtype)[()],
        is_fill=lambda arr: np.isnat(arr),
        scalar_is_fill=lambda value: bool(np.isnat(np.asarray(value))),
        values_equal=lambda a, b: (bool(np.isnat(a)) and bool(np.isnat(b))) or bool(a == b),
    )


def _encoded_encoding(dtype: np.dtype[Any], *, units: str, calendar: str) -> CoordinateEncoding:
    """Build the encoded (calendar) `CoordinateEncoding` for *dtype*/*units*/*calendar*."""
    fill_value = _encoded_fill_value(dtype)
    return CoordinateEncoding(
        dtype=dtype,
        fill_value=fill_value,
        extra_attrs={"units": units, "calendar": calendar},
        encode_values=lambda values: np.asarray(values, dtype=dtype),
        encode_scalar=lambda value: encode_coordinate(value, units=units, calendar=calendar),
        is_fill=lambda arr: _fill_mask(arr, fill_value),
        scalar_is_fill=lambda value: _scalar_is_fill(value, fill_value),
        values_equal=lambda a, b: bool(a == b),
    )


def _axis_values_are_integral(axis: Any) -> bool:
    """Return whether *axis*'s encoded values are all integral.

    An `IrregularTimeAxis` with ``calendar`` set stores each value
    canonicalised to a Python ``int`` or ``float`` at construction (see
    ``IrregularTimeAxis.__post_init__``); its concrete ``values`` are
    checked directly. A `RegularTimeAxis` has no ``values`` sequence -- its
    encoded values are ``n * cadence_s`` for a positive integer
    ``cadence_s``, always integral.
    """
    values = getattr(axis, "values", None)
    if values is None:
        return True
    return all(isinstance(value, int) for value in values)


def coordinate_encoding_for(
    axis: Any, spec: Any | None, *, coord_path: str | None = None
) -> CoordinateEncoding:
    """Build the storage encoding a time axis materializes to.

    Dispatches once, on `firecube.core.time.calendar.is_gregorian_axis`: a
    Gregorian axis (``axis.calendar`` unset, or normalising to a
    Gregorian-like name) stores ``datetime64``; any other calendar stores
    an encoded ``int64``/``float64`` number in the axis's CF ``units``.

    Args:
        axis: A ``RegularTimeAxis`` or ``IrregularTimeAxis``.
        spec: The plugin's ``ZarrArraySpec`` for this coordinate, or
            ``None``. A declared ``dtype`` is honored (Gregorian) or
            validated against the axis's calendar (encoded) -- see
            `_resolve_calendar_target_dtype`.
        coord_path: Store path used only to name this coordinate in
            `ConfigurationError` messages raised by dtype/attrs validation.
            Defaults to ``axis.coordinate``.

    Returns:
        The `CoordinateEncoding` this axis materializes to.

    Raises:
        ConfigurationError: See `_validate_calendar_spec_attrs` and
            `_resolve_calendar_target_dtype`.
    """
    path = coord_path if coord_path is not None else axis.coordinate

    if is_gregorian_axis(axis):
        target_dtype = (
            np.dtype(spec.dtype)
            if spec is not None and getattr(spec, "dtype", None) is not None
            else np.dtype("datetime64[ns]")
        )
        return _gregorian_encoding(target_dtype)

    calendar = axis.calendar
    units = getattr(axis, "encoded_units", None)
    if units is None:
        units = axis.units
    _validate_calendar_spec_attrs(spec, axis_units=units, axis_calendar=calendar, coord_path=path)
    target_dtype = _resolve_calendar_target_dtype(
        spec=spec,
        calendar=calendar,
        values_are_integral=_axis_values_are_integral(axis),
        coord_path=path,
    )
    return _encoded_encoding(target_dtype, units=units, calendar=calendar)


def coordinate_encoding_from_array(
    dtype: np.dtype[Any] | str, attrs: Mapping[str, Any]
) -> CoordinateEncoding:
    """Build the storage encoding an existing coordinate array already uses.

    Mirrors `coordinate_encoding_for`'s branch rule from what an opened
    array already carries, for a caller -- the region writer -- that only
    has an array, not the axis that created it: a non-``datetime64`` dtype
    whose attrs carry both ``units`` and ``calendar`` is an encoded
    (calendar) coordinate; any other combination is Gregorian.

    Args:
        dtype: The coordinate array's on-disk dtype.
        attrs: The coordinate array's attrs.

    Returns:
        The `CoordinateEncoding` matching *dtype*/*attrs*.
    """
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.kind != "M" and "units" in attrs and "calendar" in attrs:
        return _encoded_encoding(
            resolved_dtype, units=str(attrs["units"]), calendar=str(attrs["calendar"])
        )
    return _gregorian_encoding(resolved_dtype)
