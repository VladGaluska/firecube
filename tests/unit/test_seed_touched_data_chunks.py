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

"""Tests for staged data write-unit seeding."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr
from zarr.core.array import Array
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

from firecube.core.storage.uri import StorageUri
from firecube.ingestor.errors import SeedingFailedError
from firecube.ingestor.runtime.zarr import staged_metadata
from firecube.ingestor.runtime.zarr.append import _compute_touched_data_chunks
from firecube.ingestor.runtime.zarr.append_services import verify_post_write_integrity
from firecube.ingestor.runtime.zarr.staged_metadata import seed_touched_data_chunks
from tests.helpers.storage import local_zarr_handle, make_local_session


def _create_pair(
    tmp: str,
    *,
    shards: tuple[int, ...] | None = None,
    chunk_key_encoding: dict[str, Any] | None = None,
) -> tuple[Path, Path, Array, Array]:
    final = Path(tmp) / "final.zarr"
    temp = Path(tmp) / "temp.zarr"
    target_root = zarr.open_group(str(final), mode="w", zarr_format=3)
    ws_root = zarr.open_group(str(temp), mode="w", zarr_format=3)
    create_kwargs: dict[str, Any] = {
        "shape": (8,),
        "dtype": "i4",
        "chunks": (2,),
    }
    if shards is not None:
        create_kwargs["shards"] = shards
    if chunk_key_encoding is not None:
        create_kwargs["chunk_key_encoding"] = chunk_key_encoding
    target = target_root.create_array("G/val", **create_kwargs)
    workspace = ws_root.create_array("G/val", **create_kwargs)
    target[:] = np.arange(8, dtype="i4")
    return final, temp, target, workspace


def _run_seed(final: Path, temp: Path, touched: list[tuple[int, ...]]) -> None:
    seed_touched_data_chunks(
        temp_store_uri=str(temp),
        final_target_uri=str(final),
        touched_chunks={"G": {"val": touched}},
        session=make_local_session(str(final)),
    )


def _count_array_writes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[slice, ...]]:
    original: Callable[..., Any] = Array.__setitem__
    calls: list[tuple[slice, ...]] = []

    def counted(self: Array, selection: Any, value: Any) -> Any:
        calls.append(selection)
        return original(self, selection, value)

    monkeypatch.setattr(Array, "__setitem__", counted)
    return calls


@pytest.mark.unit
def test_non_sharded_fresh_workspace_copies_each_touched_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, temp, target, workspace = _create_pair(str(tmp_path))
    calls = _count_array_writes(monkeypatch)

    _run_seed(final, temp, [(0,), (1,), (2,)])

    assert len(calls) == 3
    np.testing.assert_array_equal(workspace[:6], target[:6])


@pytest.mark.unit
def test_non_sharded_existing_chunk_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, temp, _target, workspace = _create_pair(str(tmp_path))
    workspace[0:2] = np.array([99, 99], dtype="i4")
    calls = _count_array_writes(monkeypatch)

    _run_seed(final, temp, [(0,)])

    assert calls == []
    np.testing.assert_array_equal(workspace[0:2], np.array([99, 99], dtype="i4"))


@pytest.mark.unit
def test_non_sharded_partial_coverage_only_copies_missing_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, temp, target, workspace = _create_pair(str(tmp_path))
    workspace[0:2] = np.array([20, 21], dtype="i4")
    workspace[4:6] = np.array([24, 25], dtype="i4")
    calls = _count_array_writes(monkeypatch)

    _run_seed(final, temp, [(0,), (1,), (2,)])

    assert len(calls) == 1
    assert calls[0] == (slice(2, 4),)
    np.testing.assert_array_equal(workspace[2:4], target[2:4])
    np.testing.assert_array_equal(workspace[0:2], np.array([20, 21], dtype="i4"))
    np.testing.assert_array_equal(workspace[4:6], np.array([24, 25], dtype="i4"))


@pytest.mark.unit
def test_sharded_touched_chunks_in_existing_shard_do_not_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, temp, _target, workspace = _create_pair(str(tmp_path), shards=(4,))
    workspace[0:4] = np.array([90, 91, 92, 93], dtype="i4")
    calls = _count_array_writes(monkeypatch)

    _run_seed(final, temp, [(0,), (1,)])

    assert calls == []
    np.testing.assert_array_equal(workspace[0:4], np.array([90, 91, 92, 93], dtype="i4"))


@pytest.mark.unit
def test_sharded_fresh_workspace_copies_each_touched_shard_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, temp, target, workspace = _create_pair(str(tmp_path), shards=(4,))
    calls = _count_array_writes(monkeypatch)

    _run_seed(final, temp, [(0,), (1,), (2,)])

    assert len(calls) == 2
    assert calls == [(slice(0, 4),), (slice(4, 8),)]
    np.testing.assert_array_equal(workspace[:], target[:])


@pytest.mark.unit
def test_copy_failure_deletes_workspace_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, temp, _target, _workspace = _create_pair(str(tmp_path))

    def fail_setitem(self: Array, selection: Any, value: Any) -> None:
        raise RuntimeError("copy failed")

    monkeypatch.setattr(Array, "__setitem__", fail_setitem)

    with pytest.raises(SeedingFailedError, match="copy failed"):
        _run_seed(final, temp, [(0,)])

    assert not temp.exists()


@pytest.mark.unit
def test_transient_read_error_during_seeding_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, temp, _target, _workspace = _create_pair(str(tmp_path))
    val_reads = 0

    def fail_get_child(parent: Any, name: str) -> Any | None:
        nonlocal val_reads
        if name.rsplit("/", 1)[-1] == "val":
            val_reads += 1
        if name.rsplit("/", 1)[-1] == "val" and val_reads == 2:
            raise PermissionError("transient read denied")
        return original_get_child(parent, name)

    original_get_child = staged_metadata._get_child
    monkeypatch.setattr(staged_metadata, "_get_child", fail_get_child)

    with pytest.raises(SeedingFailedError, match="transient read denied"):
        _run_seed(final, temp, [(0,)])

    assert not temp.exists()


@pytest.mark.unit
def test_dot_separator_chunk_key_encoding_no_reseed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, temp, _target, _workspace = _create_pair(
        str(tmp_path),
        chunk_key_encoding={"name": "default", "configuration": {"separator": "."}},
    )
    _run_seed(final, temp, [(0,), (1,)])
    calls = _count_array_writes(monkeypatch)

    _run_seed(final, temp, [(0,), (1,)])

    assert calls == []


@pytest.mark.unit
def test_write_unit_exists_bogus_store_raises() -> None:
    class BogusStore:
        pass

    class Metadata:
        chunk_key_encoding = DefaultChunkKeyEncoding(separator="/")

    class BogusArray:
        metadata = Metadata()
        path = "G/val"
        store = BogusStore()

    with pytest.raises(RuntimeError, match="does not expose async exists"):
        staged_metadata._write_unit_exists(cast(zarr.Array, BogusArray()), (0,))


@pytest.mark.unit
def test_compute_touched_data_chunks_missing_array_logs_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    temp = tmp_path / "temp.zarr"
    root = zarr.open_group(str(temp), mode="w", zarr_format=3)
    root.create_group("G")

    caplog.set_level(logging.DEBUG, logger="firecube.ingestor.runtime.zarr.append")
    result = _compute_touched_data_chunks(
        classification=None,
        write_cursor=0,
        count=1,
        append_dim="timestamp",
        zarr_store=local_zarr_handle(temp, mode="r"),
        coverage_arrays=["G/missing_val"],
        group="G",
        state_var_name="firecube_timestamp_state",
    )

    assert result == {"G": {}}
    assert any(
        record.levelno == logging.DEBUG
        and "workspace array missing" in record.message
        and record.__dict__.get("array") == "missing_val"
        for record in caplog.records
    )


@pytest.mark.unit
@pytest.mark.parametrize("snapshot_path", [None, "G"], ids=["root", "group"])
def test_cells_past_snapshot_extent_are_seeded_from_grown_target(
    tmp_path: Path, snapshot_path: str | None
) -> None:
    """Seeding sizes units from the live target, not from the snapshot the workspace copied."""
    final, temp, target, _workspace = _create_pair(str(tmp_path))
    cast(Any, zarr.open_group(str(final), mode="r+")["G"]).create_array(
        "timestamp", shape=(2,), chunks=(2,), dtype="i8"
    )
    zarr.consolidate_metadata(str(final), path=snapshot_path)
    target.resize((12,))
    target[8:12] = np.arange(100, 104, dtype="i4")
    shutil.rmtree(temp)
    session = make_local_session(str(final))
    staged_metadata.seed_staged_store_metadata(
        temp_store_uri=str(temp),
        final_target_uri=str(final),
        groups=["G"],
        strict=True,
        session=session,
        coordinate_arrays=["timestamp"],
    )

    seed_touched_data_chunks(
        temp_store_uri=str(temp),
        final_target_uri=str(final),
        touched_chunks={"G": {"val": [(4,), (5,)]}},
        session=session,
    )

    live = zarr.open_group(str(temp), mode="r", use_consolidated=False)
    np.testing.assert_array_equal(cast(Any, live["G/val"])[8:12], np.arange(100, 104, dtype="i4"))


_STATE = "firecube_timestamp_state"


def _resume_over_group_snapshot(tmp: Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Group G of the target has a snapshot at shape 8, then grows to 12; a one-slot write at 10."""
    final, temp = tmp / "final.zarr", tmp / "temp.zarr"
    uris: dict[str, Any] = {"temp_store_uri": str(temp), "final_target_uri": str(final)}
    group = zarr.open_group(str(final), mode="w", zarr_format=3).require_group("G")
    data = {"val": 100 + np.arange(12), "timestamp": np.arange(12), _STATE: np.ones(12)}
    kw: dict[str, Any] = {"shape": (8,), "chunks": (2,), "dimension_names": ("timestamp",)}
    arrays = {n: group.create_array(n, dtype="i8", **kw) for n in data}
    zarr.consolidate_metadata(str(final), path="G")
    for name, arr in arrays.items():
        arr.resize((12,))
        arr[:] = data[name]
    session = make_local_session(str(final))
    staged_metadata.seed_staged_store_metadata(
        **uris, groups=["G"], strict=True, session=session, coordinate_arrays=["timestamp"]
    )
    touched = _compute_touched_data_chunks(
        classification=None,
        write_cursor=10,
        count=1,
        append_dim="timestamp",
        zarr_store=local_zarr_handle(temp, mode="r"),
        coverage_arrays=["G/val"],
        group="G",
        state_var_name=_STATE,
    )
    seed_touched_data_chunks(**uris, touched_chunks=touched, session=session)
    return uris, session, touched


@pytest.mark.unit
def test_touched_chunks_of_grown_array_behind_group_snapshot(tmp_path: Path) -> None:
    assert _resume_over_group_snapshot(tmp_path)[2]["G"].get("val") == [(5,)]


@pytest.mark.unit
def test_group_snapshot_resume_keeps_cells_beside_new_data(tmp_path: Path) -> None:
    uris, session, touched = _resume_over_group_snapshot(tmp_path)
    ws: Any = zarr.open_group(uris["temp_store_uri"], mode="r+", use_consolidated=False)
    ws["G/val"][10] = 999

    verify_post_write_integrity(
        **uris, touched_chunks=touched, session=session, append_dim="timestamp"
    )
    session.upload_tree(
        StorageUri.from_local_path(Path(uris["temp_store_uri"])),
        StorageUri.from_local_path(Path(uris["final_target_uri"])),
        parallel_workers=4,
    )

    after: Any = zarr.open_group(uris["final_target_uri"], mode="r", use_consolidated=False)
    assert after["G/val"][10:12].tolist() == [999, 111]
