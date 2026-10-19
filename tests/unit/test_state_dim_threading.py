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

import shutil
from typing import Any

import numpy as np
import pytest
import zarr

from firecube.core.config import StorageConfig
from firecube.core.zarr.state import ensure_timestamp_state_array, update_timestamp_state


def _dimension_names(array: Any) -> tuple[str, ...] | None:
    return tuple(array.metadata.dimension_names or ())


@pytest.mark.unit
def test_attach_timestamp_state_honors_custom_dim(tmp_path):
    """Setting dim='time' MUST produce a zarr state array with dimension_names=('time',)."""
    store_path = tmp_path / "custom-dim.zarr"

    ensure_timestamp_state_array(
        store_uri=str(store_path),
        array_path="G1/firecube_timestamp_state",
        length=3,
        chunk_len=2,
        dim="time",
    )

    root = zarr.open_group(store=str(store_path), mode="r")
    state = root["G1/firecube_timestamp_state"]
    assert _dimension_names(state) == ("time",)
    assert _dimension_names(state) != ("timestamp",)


@pytest.mark.unit
def test_attach_timestamp_state_backcompat_default_timestamp(tmp_path):
    """Default 'timestamp' still produces dimension_names=('timestamp',)."""
    store_path = tmp_path / "default-dim.zarr"

    ensure_timestamp_state_array(
        store_uri=str(store_path),
        array_path="G1/firecube_timestamp_state",
        length=3,
        chunk_len=2,
    )

    root = zarr.open_group(store=str(store_path), mode="r")
    state = root["G1/firecube_timestamp_state"]
    assert _dimension_names(state) == ("timestamp",)


def _snapshot_state_store(tmp_path, where: str) -> str:
    """Store whose snapshot (at the root, or at group G alone) lists G/st."""
    path = str(tmp_path / "p.zarr")
    zarr.open_group(path, mode="w", zarr_format=3).create_group("G").create_array(
        "st", shape=(2,), chunks=(2,), dtype="u1", fill_value=0, dimension_names=["timestamp"]
    )
    zarr.consolidate_metadata(path, path="G" if where == "group" else "")
    return path


SNAPSHOT_CASES = [
    pytest.param("config", "root", id="storage_config-root"),
    pytest.param("options", "group", id="storage_options-group"),
]


def _branch(kind: str) -> dict[str, Any]:
    return {"storage_config": StorageConfig(storage_type="local")} if kind == "config" else {}


@pytest.mark.unit
@pytest.mark.parametrize(("kind", "where"), SNAPSHOT_CASES)
def test_ensure_state_recreates_array_missing_despite_snapshot(tmp_path, kind, where):
    path = _snapshot_state_store(tmp_path, where)
    shutil.rmtree(tmp_path / "p.zarr" / "G" / "st")

    ensure_timestamp_state_array(
        store_uri=path, array_path="G/st", length=4, chunk_len=2, **_branch(kind)
    )

    assert zarr.open_array(path, path="G/st", mode="r").shape == (4,)


@pytest.mark.unit
@pytest.mark.parametrize(("kind", "where"), SNAPSHOT_CASES)
def test_update_state_writes_slots_grown_past_snapshot(tmp_path, kind, where):
    path = _snapshot_state_store(tmp_path, where)
    zarr.open_array(path, path="G/st", mode="r+").resize((4,))

    update_timestamp_state(
        store_uri=path, array_path="G/st", time_index_ranges=[[2, 3]], value=7, **_branch(kind)
    )

    state = zarr.open_array(path, path="G/st", mode="r")
    assert np.asarray(state[:]).tolist() == [0, 0, 7, 7]
