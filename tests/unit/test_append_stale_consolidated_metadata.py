# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import LocalStore

from firecube.core.config import StorageConfig
from firecube.core.filesystem.store_factory import create_zarr_store
from firecube.ingestor.runtime.zarr.append_services import AppendResumeService, AppendTimestampState
from firecube.ingestor.runtime.zarr.write import write_dataset_to_zarr

pytestmark = pytest.mark.unit


def _storage_config() -> StorageConfig:
    return StorageConfig(storage_type="local", storage_driver="fsspec")


def _service_kwargs(store_path: Path) -> dict[str, Any]:
    return {
        "read_source_uri": store_path.as_uri(),
        "read_storage_options": None,
        "resume_existing": True,
        "append_dim": "timestamp",
        "chunk_shape": None,
        "shard_shape": None,
        "sharding": False,
        "logger": logging.getLogger("test-v2-f12-use-consolidated-false"),
        "state_var_name": "firecube_timestamp_state",
        "storage_config": _storage_config(),
    }


def _handle(store_path: Path, mode: str = "a"):
    return create_zarr_store(
        uri=str(store_path),
        storage_config=_storage_config(),
        mode=mode,
    )


def _times(periods: int = 5) -> np.ndarray:
    offsets = np.arange(periods).astype("timedelta64[h]")
    return np.datetime64("2024-01-01T00:00:00", "ns") + offsets


def _dataset(times: np.ndarray, values: np.ndarray | None = None) -> xr.Dataset:
    data = np.arange(times.size, dtype=np.int32) if values is None else values
    return xr.Dataset(
        {
            "value": (("timestamp",), data),
            "firecube_timestamp_state": (
                ("timestamp",),
                np.ones(times.size, dtype=np.uint8),
            ),
        },
        coords={"timestamp": times},
    )


def _write_stale_consolidated_store(store_path: Path, times: np.ndarray) -> None:
    initial = xr.Dataset(
        {"value": (("timestamp",), np.arange(times.size, dtype=np.int32))},
        coords={"timestamp": times},
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 specification",
        )
        initial.to_zarr(
            str(store_path),
            group="default",
            mode="w",
            zarr_format=3,
            consolidated=False,
            safe_chunks=False,
        )
        zarr.consolidate_metadata(str(store_path))

    group = cast(
        Any,
        zarr.open_group(str(store_path), mode="r+", zarr_format=3, use_consolidated=False)[  # type: ignore[index]
            "default"
        ],
    )
    state = group.create_array(
        "firecube_timestamp_state",
        shape=times.shape,
        chunks=(max(int(times.size), 1),),
        dtype=np.uint8,
        dimension_names=("timestamp",),
    )
    state[:] = np.ones(times.shape, dtype=np.uint8)


def test_classify_with_stale_consolidated_metadata(tmp_path: Path) -> None:
    """classify must bypass stale consolidated metadata for append state."""
    store_path = tmp_path / "f12-classify-stale.zarr"
    times = _times()
    _write_stale_consolidated_store(store_path, times)

    result = AppendResumeService(**_service_kwargs(store_path)).classify_dataset(
        ds=_dataset(times),
        group="default",
    )

    assert result.mode == "region_overwrite"
    assert result.overwrite_slice == slice(0, int(times.size))
    assert result.new_values == []


def test_write_mode_r_plus_with_stale_consolidated(tmp_path: Path) -> None:
    """region writes must open r+ groups without consolidated metadata."""
    store_path = tmp_path / "f12-write-stale.zarr"
    times = _times()
    _write_stale_consolidated_store(store_path, times)

    replacement = np.arange(100, 103, dtype=np.int32)
    region = slice(1, 4)
    write_dataset_to_zarr(
        _dataset(times[region], values=replacement),
        zarr_store=_handle(store_path, mode="a"),
        group="default",
        region=region,
        time_dim="timestamp",
    )

    group = zarr.open_group(str(store_path), mode="r", zarr_format=3, use_consolidated=False)[  # type: ignore[index]
        "default"
    ]
    values = np.asarray(group["value"][:])  # type: ignore[index]
    state = np.asarray(group["firecube_timestamp_state"][:])  # type: ignore[index]

    expected = np.arange(times.size, dtype=np.int32)
    expected[region] = replacement
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(state, np.ones(times.shape, dtype=np.uint8))


def test_zarr_group_has_array_ignores_stale_consolidated_metadata(tmp_path: Path) -> None:
    """probe: the state array added after consolidation must still be found."""
    from firecube.core.controlplane.deletion import _zarr_group_has_array

    store_path = tmp_path / "f12-has-array-stale.zarr"
    _write_stale_consolidated_store(store_path, _times())

    assert _zarr_group_has_array(
        store_uri=str(store_path),
        group="default",
        array_name="firecube_timestamp_state",
        storage_config=_storage_config(),
    )
    assert not _zarr_group_has_array(
        store_uri=str(store_path),
        group="default",
        array_name="no_such_array",
        storage_config=_storage_config(),
    )


def test_region_nan_fill_ignores_stale_consolidated_metadata(tmp_path: Path) -> None:
    """fill: slots and state are filled on a store with stale consolidated metadata."""
    from firecube.core.controlplane.deletion import delete_span_via_region_nan_fill

    store_path = tmp_path / "f12-nan-fill-stale.zarr"
    times = _times()
    _write_stale_consolidated_store(store_path, times)

    delete_span_via_region_nan_fill(
        str(store_path),
        "default",
        [1, 2, 3],
        _storage_config(),
        time_dim_name="timestamp",
    )

    group = zarr.open_group(str(store_path), mode="r", zarr_format=3, use_consolidated=False)[  # type: ignore[index]
        "default"
    ]
    value_array = cast(Any, group["value"])  # type: ignore[index]
    values = np.asarray(value_array[:])
    state = np.asarray(group["firecube_timestamp_state"][:])  # type: ignore[index]

    expected = np.arange(times.size, dtype=np.int32)
    expected[1:4] = value_array.fill_value
    np.testing.assert_array_equal(values, expected)
    expected_state = np.ones(times.shape, dtype=np.uint8)
    expected_state[1:4] = 2
    np.testing.assert_array_equal(state, expected_state)


def _grown_under_snapshot(store_path: Path, shards: tuple[int, ...] | None = None) -> None:
    """Consolidate a 2-slot ``default/value``, then grow it to 4 without refreshing."""
    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    root.create_array("default/value", shape=(2,), chunks=(1,), shards=shards, dtype="i4")
    zarr.consolidate_metadata(str(store_path))
    live = zarr.open_group(str(store_path), mode="r+", zarr_format=3, use_consolidated=False)
    cast(Any, live["default/value"]).resize((4,))


def _resume_cursor(service: AppendResumeService, store: object) -> int:
    ts_state = AppendTimestampState("firecube_timestamp_state", time_dim_name="timestamp")
    service.prepare_write(
        ds=ts_state.attach(_dataset(_times(2)), append_dim="timestamp"),
        group="default",
        store=store,
        write_target_uri=None,
        arrays_for_group=None,
        ts_state=ts_state,
    )
    return service.write_cursor


def test_resume_cursor_from_store_object_reads_live_shape(tmp_path: Path) -> None:
    """A store object whose array grew under a root snapshot resumes at the live length."""
    store_path = tmp_path / "cursor-store.zarr"
    _grown_under_snapshot(store_path)
    kwargs = {**_service_kwargs(store_path), "read_source_uri": None}

    cursor = _resume_cursor(AppendResumeService(**kwargs), LocalStore(str(store_path)))

    assert cursor == 4


@pytest.mark.parametrize("shards", [None, (2,)], ids=["unsharded", "sharded"])
def test_resume_cursor_from_source_uri_reads_live_shape(
    tmp_path: Path, shards: tuple[int, ...] | None
) -> None:
    """A storage-URI source whose array grew under a root snapshot resumes at the live length."""
    store_path = tmp_path / "cursor-uri.zarr"
    _grown_under_snapshot(store_path, shards)
    kwargs = {**_service_kwargs(store_path), "sharding": shards is not None}

    cursor = _resume_cursor(AppendResumeService(**kwargs), LocalStore(str(store_path)))

    assert cursor == 4
