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

"""Relabel refusals of ``coord_rematerialization``, through materialize and through its checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import zarr

from firecube.core.index_spec import IrregularTimeAxis, RegularTimeAxis
from firecube.core.zarr._reserved_attrs import FIRECUBE_GROUP_IDENTITY_HASH_ATTR as HASH_ATTR
from firecube.core.zarr.coord_materialization import (
    materialize_irregular_coord_array,
    materialize_regular_coord_array,
)
from firecube.core.zarr.region_writer import RegionZarrWriter

# isort: split
from firecube.core.zarr.coord_rematerialization import (
    CoordinateRelabelError,
    check_irregular_coord_rematerialization,
    check_regular_coord_rematerialization,
)


def _regular(*, moved: bool = False, calendar: str = "360_day") -> RegularTimeAxis:
    epoch = "2050-01-01T00:00:00Z" if moved else "2049-01-01T00:00:00Z"
    return RegularTimeAxis(
        coordinate="time", epoch=epoch, cadence_s=86400, slot_count=3, calendar=calendar
    )


def _irregular(*, moved: bool = False, calendar: str = "360_day") -> IrregularTimeAxis:
    units = f"days since {2050 if moved else 2049}-01-01"
    return IrregularTimeAxis(coordinate="time", values=[1, 2, 3], units=units, calendar=calendar)


_KINDS: dict[str, Any] = {
    "regular": (_regular, materialize_regular_coord_array, check_regular_coord_rematerialization),
    "irregular": (
        _irregular,
        materialize_irregular_coord_array,
        check_irregular_coord_rematerialization,
    ),
}


def _seeded(tmp_path: Path, kind: str, stale: bool = False, **stored: Any) -> tuple[Path, Any, Any]:
    store = tmp_path / "cube.zarr"
    zarr.open_group(store=str(store), mode="w", zarr_format=3)
    writer = RegionZarrWriter(str(store))
    root = writer._open_root()
    if stale:  # a consolidated snapshot of group "data" taken before its time coordinate exists
        root.require_group("data")
        zarr.consolidate_metadata(str(store), path="data")
    make, materialize, _ = _KINDS[kind]
    materialize(writer=writer, root=root, group_name="data", axis=make(**stored), spec=None)
    return store, writer, root


def _snapshot(store: Path) -> dict[str, bytes]:
    return {str(p.relative_to(store)): p.read_bytes() for p in store.rglob("*") if p.is_file()}


@pytest.mark.parametrize("stale", [False, True], ids=["live", "stale-snapshot"])
@pytest.mark.parametrize("op", ["materialize", "check"])
@pytest.mark.parametrize(
    "case",
    [
        ("regular", {}, {"moved": True}, False, "units", ("2049", "2050")),
        ("regular", {}, {"calendar": "noleap"}, False, "calendar", ("360_day", "noleap")),
        ("regular", {}, {"calendar": "proleptic_gregorian"}, False, "dtype", ("360_day", "datetime64")),
        ("regular", {"calendar": "proleptic_gregorian"}, {}, False, "dtype", ("datetime64", "360_day")),
        ("regular", {}, {}, True, HASH_ATTR, ("0" * 16, "")),
        ("irregular", {}, {"moved": True}, False, "units", ("2049", "2050")),
        ("irregular", {}, {"calendar": "noleap"}, False, "calendar", ("360_day", "noleap")),
        ("irregular", {}, {}, True, HASH_ATTR, ("0" * 16, "")),
    ],
    ids=[
        "regular-epoch", "regular-calendar", "regular-to-gregorian", "regular-to-calendar",
        "regular-stale-hash", "irregular-units", "irregular-calendar", "irregular-stale-hash",
    ],
)  # fmt: skip
def test_relabel_of_stored_times_is_refused_before_any_write(
    tmp_path: Path, stale: bool, op: str, case: tuple[Any, ...]
) -> None:
    kind, stored, incoming, tamper, differs_in, facts = case
    store, writer, root = _seeded(tmp_path, kind, stale, **stored)
    if tamper:
        root["data/time"].attrs[HASH_ATTR] = "0" * 64
        root["data/time"][:] = root["data/time"].fill_value  # marked, all fill, stale hash
    before = _snapshot(store)
    make, materialize, check = _KINDS[kind]
    read_only = zarr.open_group(store=str(store), mode="r", zarr_format=3)

    with pytest.raises(CoordinateRelabelError) as excinfo:
        if op == "materialize":
            materialize(
                writer=writer, root=root, group_name="data", axis=make(**incoming), spec=None
            )
        else:
            check(root=read_only, group_name="data", axis=make(**incoming), spec=None)

    assert excinfo.value.coord_path == "data/time"
    assert differs_in in excinfo.value.differs_in
    assert facts[0] in str(excinfo.value.stored)
    assert facts[1] in str(excinfo.value.incoming)
    assert "0" * 16 not in str(excinfo.value.incoming)  # the expected hash, not the stale one
    assert _snapshot(store) == before


@pytest.mark.parametrize("kind", ["regular", "irregular"])
def test_identical_rerun_leaves_the_store_byte_identical(tmp_path: Path, kind: str) -> None:
    store, writer, root = _seeded(tmp_path, kind)
    before = _snapshot(store)
    make, materialize, _ = _KINDS[kind]

    materialize(writer=writer, root=root, group_name="data", axis=make(), spec=None)
    _KINDS[kind][2](root=None, group_name="data", axis=make(), spec=None)

    assert _snapshot(store) == before
