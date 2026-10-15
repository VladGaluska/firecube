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

"""Regression tests for append_time_groups() behavior.

These tests freeze the current behavior before any decomposition begins.
If any test breaks during refactoring, a regression has been introduced.

Covers: single-group append, multi-group append, resume cursor inference,
empty batch handling, coverage entry generation, timestamp-state initialization,
multires follow-up, chunk alignment warnings, shard validation.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from firecube.ingestor.runtime.zarr.append import append_time_groups
from firecube.ingestor.runtime.zarr.resume_cache import (
    ResumeCacheEntry,
    clear_resume_cache,
    get_resume_cache_entry,
    put_resume_cache_entry,
)
from tests.helpers.rows import table_row
from tests.helpers.storage import local_zarr_handle, make_local_session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(timestamps, nlat=2, nlon=3, var_name="FWI"):
    """Build a simple xr.Dataset with given timestamps."""
    ts = pd.to_datetime(list(timestamps))
    data = np.arange(len(ts) * nlat * nlon, dtype=np.float32).reshape(len(ts), nlat, nlon)
    return xr.Dataset(
        {var_name: (("timestamp", "lat", "lon"), data)},
        coords={"timestamp": ts, "lat": np.arange(nlat), "lon": np.arange(nlon)},
    )


def _write_initial_store(store_path, group, n_timestamps, nlat=2, nlon=3):
    """Write an initial zarr store with n_timestamps for pre-populating."""
    ts = pd.date_range("2024-01-01", periods=n_timestamps, freq="h")
    ds = xr.Dataset(
        {
            "FWI": (
                ("timestamp", "lat", "lon"),
                np.zeros((n_timestamps, nlat, nlon), dtype=np.float32),
            )
        },
        coords={"timestamp": ts, "lat": np.arange(nlat), "lon": np.arange(nlon)},
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 specification",
        )
        ds.to_zarr(
            str(store_path),
            group=group,
            mode="w",
            zarr_format=3,
            safe_chunks=False,
            align_chunks=True,
        )
    return ts


def _dataset_factory(var_name="FWI", nlat=2, nlon=3):
    """Return a dataset_for_batch callable."""

    def _build(group: str, batch_ts):
        return _make_dataset(batch_ts, nlat=nlat, nlon=nlon, var_name=var_name)

    return _build


def _append(store: str, group_to_timestamps: dict[str, list[Any]], **kwargs: Any) -> dict[str, Any]:
    """Run ``append_time_groups`` against a fresh local session on ``store``."""
    kwargs.setdefault("dataset_for_batch", _dataset_factory())
    kwargs.setdefault("batch_size", 10)
    return append_time_groups(
        store=store,
        zarr_store=local_zarr_handle(store),
        session=make_local_session(store),
        group_to_timestamps=group_to_timestamps,
        **kwargs,
    )


def _fwi(g: str) -> list[str]:
    return [f"{g}/FWI"]


@pytest.fixture(autouse=True)
def _clear_resume_cache():
    """Ensure resume cache is empty between tests."""
    clear_resume_cache()
    yield
    clear_resume_cache()


@pytest.mark.unit
def test_fresh_append_writes_and_coverage(subtests: Any, tmp_path: Path) -> None:
    with table_row(subtests, "single_group_basic_append", tmp_path) as (workdir, _):
        store = str(workdir / "single.zarr")
        ts = pd.date_range("2024-01-01", periods=3, freq="h")
        metrics = _append(store, {"G1": list(ts)})
        assert metrics["batch_processing"]["timestamps_written"] == 3
        ds = xr.open_zarr(store, group="G1", consolidated=False)
        assert ds.sizes["timestamp"] == 3

    with table_row(subtests, "single_group_coverage_entry", tmp_path) as (workdir, _):
        store = str(workdir / "cov.zarr")
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        metrics = _append(store, {"G1": list(ts)}, arrays_for_group=_fwi)
        assert len(metrics["coverage"]) == 1
        cov = metrics["coverage"][0]
        assert cov["group"] == "G1"
        assert cov["arrays"] == ["G1/FWI"]
        assert cov["time_index_ranges"] == [[0, 4]]
        assert cov["state_array"] == "G1/firecube_timestamp_state"

    with table_row(subtests, "multi_group_append", tmp_path) as (workdir, _):
        store = str(workdir / "multi.zarr")
        ts1 = pd.date_range("2024-01-01", periods=2, freq="h")
        ts2 = pd.date_range("2024-02-01", periods=3, freq="h")
        ts3 = pd.date_range("2024-03-01", periods=4, freq="h")
        metrics = _append(store, {"A": list(ts1), "B": list(ts2), "C": list(ts3)})
        assert metrics["batch_processing"]["timestamps_written"] == 2 + 3 + 4
        for grp, expected in [("A", 2), ("B", 3), ("C", 4)]:
            ds = xr.open_zarr(store, group=grp, consolidated=False)
            assert ds.sizes["timestamp"] == expected, f"Group {grp} mismatch"

    with table_row(subtests, "multi_group_independent_coverage", tmp_path) as (workdir, _):
        store = str(workdir / "multi_cov.zarr")
        ts1 = pd.date_range("2024-01-01", periods=2, freq="h")
        ts2 = pd.date_range("2024-02-01", periods=3, freq="h")
        metrics = _append(store, {"X": list(ts1), "Y": list(ts2)}, arrays_for_group=_fwi)
        cov = metrics["coverage"]
        assert len(cov) == 2
        assert {c["group"] for c in cov} == {"X", "Y"}
        for entry in cov:
            assert len(entry["time_index_ranges"]) >= 1

    with table_row(subtests, "coverage_entry_time_index_ranges_single_batch", tmp_path) as (
        workdir,
        _,
    ):
        store = str(workdir / "cov_contig.zarr")
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        metrics = _append(store, {"G1": list(ts)}, arrays_for_group=_fwi, batch_size=100)
        assert metrics["coverage"][0]["time_index_ranges"] == [[0, 4]]

    with table_row(subtests, "coverage_entry_multiple_batches", tmp_path) as (workdir, _):
        store = str(workdir / "cov_multi_batch.zarr")
        ts = pd.date_range("2024-01-01", periods=6, freq="h")
        metrics = _append(store, {"G1": list(ts)}, arrays_for_group=_fwi, batch_size=2)
        ranges = metrics["coverage"][0]["time_index_ranges"]
        assert ranges == [[0, 1], [2, 3], [4, 5]]

    with table_row(subtests, "coverage_entry_time_bounds", tmp_path) as (workdir, _):
        store = str(workdir / "cov_bounds.zarr")
        ts = pd.date_range("2024-06-15T08:00", periods=4, freq="h")
        metrics = _append(store, {"G1": list(ts)}, arrays_for_group=_fwi)
        cov = metrics["coverage"][0]
        assert cov["time_min"] is not None
        assert cov["time_max"] is not None
        assert pd.Timestamp(cov["time_min"]) == ts[0]
        assert pd.Timestamp(cov["time_max"]) == ts[-1]

    with table_row(subtests, "timestamp_state_initialized", tmp_path) as (workdir, _):
        store = str(workdir / "state.zarr")
        ts = pd.date_range("2024-01-01", periods=3, freq="h")
        _append(store, {"G1": list(ts)})
        ds = xr.open_zarr(store, group="G1", consolidated=False)
        assert "firecube_timestamp_state" in ds
        assert np.all(ds["firecube_timestamp_state"].values == 1)

    with table_row(subtests, "metrics_structure", tmp_path) as (workdir, _):
        store = str(workdir / "metrics.zarr")
        ts = pd.date_range("2024-01-01", periods=3, freq="h")
        metrics = _append(store, {"G1": list(ts)})
        assert "duration_s" in metrics
        assert isinstance(metrics["duration_s"], float)
        assert "timestamps_per_group" in metrics
        assert metrics["timestamps_per_group"]["G1"] == 3
        assert "batch_processing" in metrics
        bp = metrics["batch_processing"]
        assert bp["batch_size"] == 10
        assert bp["batches_attempted"] >= 1
        assert bp["batches_written"] >= 1
        assert bp["timestamps_requested"] == 3
        assert bp["timestamps_written"] == 3

    with table_row(subtests, "resume_cache_updated_after_write", tmp_path) as (workdir, _):
        store = str(workdir / "cache_update.zarr")
        ts = pd.date_range("2024-01-01", periods=4, freq="h")
        _append(store, {"G1": list(ts)})
        entry = get_resume_cache_entry((store, "G1", "timestamp"))
        assert entry is not None
        assert entry.cursor == 4


@pytest.mark.unit
def test_empty_and_skipped_batches_write_nothing(subtests: Any, tmp_path: Path) -> None:
    with table_row(subtests, "empty_batch_no_writes", tmp_path) as (workdir, _):
        store = str(workdir / "empty.zarr")
        metrics = _append(store, {"G1": []})
        assert metrics["batch_processing"]["timestamps_written"] == 0
        assert metrics["batch_processing"]["batches_written"] == 0
        assert "coverage" not in metrics

    with table_row(subtests, "empty_group_no_writes", tmp_path) as (workdir, _):
        store = str(workdir / "empty_group.zarr")
        ts = pd.date_range("2024-01-01", periods=2, freq="h")
        metrics = _append(store, {"has_data": list(ts), "empty": []}, arrays_for_group=_fwi)
        cov_groups = [c["group"] for c in metrics.get("coverage", [])]
        assert "has_data" in cov_groups
        assert "empty" not in cov_groups

    with table_row(subtests, "dataset_none_skipped", tmp_path) as (workdir, _):
        store = str(workdir / "none_ds.zarr")
        ts = pd.date_range("2024-01-01", periods=3, freq="h")
        metrics = _append(store, {"G1": list(ts)}, dataset_for_batch=lambda group, batch_ts: None)
        assert metrics["batch_processing"]["timestamps_written"] == 0
        assert "coverage" not in metrics


@pytest.mark.unit
def test_resume_append_transitions(subtests: Any, tmp_path: Path) -> None:
    with table_row(subtests, "resume_cursor_inference", tmp_path) as (workdir, _):
        store = str(workdir / "resume_cursor.zarr")
        _write_initial_store(store, "G1", n_timestamps=5)
        new_ts = pd.date_range("2024-01-01T05:00:00", periods=3, freq="h")
        metrics = _append(store, {"G1": list(new_ts)}, arrays_for_group=_fwi, resume_existing=True)
        assert metrics["coverage"][0]["time_index_ranges"] == [[5, 7]]
        ds = xr.open_zarr(store, group="G1", consolidated=False)
        assert ds.sizes["timestamp"] == 8
        assert float(ds["FWI"].isel(timestamp=0, lat=0, lon=0).values) == 0.0

    with table_row(subtests, "resume_cache_used_when_available", tmp_path) as (workdir, _):
        store = str(workdir / "cache_hit.zarr")
        group = "G1"
        _write_initial_store(store, group, n_timestamps=5)
        cache_key = (store, group, "timestamp")
        put_resume_cache_entry(
            cache_key, ResumeCacheEntry(cursor=5, chunk_len=2, state_initialized=False)
        )
        new_ts = pd.date_range("2024-01-01T05:00:00", periods=2, freq="h")
        with patch(
            "firecube.ingestor.runtime.zarr.append._read_existing_group_meta",
            wraps=lambda *a, **kw: (True, ["timestamp", "lat", "lon"], [0, 2, 3], [99, 2, 3], None),
        ):
            metrics = _append(
                store, {group: list(new_ts)}, arrays_for_group=_fwi, resume_existing=True
            )
        assert metrics["coverage"][0]["time_index_ranges"] == [[5, 6]]
        assert zarr.open_array(store, path=f"{group}/FWI", mode="r").shape == (7, 2, 3)
        assert zarr.open_array(store, path=f"{group}/timestamp", mode="r").shape == (7,)
        resume_entry = get_resume_cache_entry(cache_key)
        assert resume_entry is not None
        assert resume_entry.cursor == 7

    with table_row(subtests, "overlap_state1_silent_filter_no_raise", tmp_path) as (workdir, _):
        # Overlapping state=1 timestamps under resume_existing are silently skipped (T25/E3).
        store = str(workdir / "overlap.zarr")
        _write_initial_store(store, "G1", n_timestamps=4)
        overlap_ts = pd.date_range("2024-01-01T02:00:00", periods=2, freq="h")
        metrics = _append(
            store, {"G1": list(overlap_ts)}, arrays_for_group=_fwi, resume_existing=True
        )
        assert metrics["timestamps_skipped"] == 2
        assert metrics["batch_processing"]["timestamps_skipped"] == 2
        assert metrics["batch_processing"]["timestamps_written"] == 0

    with table_row(subtests, "timestamp_state_on_resume_legacy_store", tmp_path) as (workdir, _):
        # Resuming into a legacy store (no state array) backfills the state array.
        store = str(workdir / "legacy_state.zarr")
        _write_initial_store(store, "G1", n_timestamps=3)
        new_ts = pd.date_range("2024-01-01T03:00:00", periods=2, freq="h")
        _append(store, {"G1": list(new_ts)}, arrays_for_group=_fwi, resume_existing=True)
        ds = xr.open_zarr(store, group="G1", consolidated=False)
        assert "firecube_timestamp_state" in ds
        assert ds.sizes["timestamp"] == 5

    with table_row(subtests, "resume_append_tolerates_configured_oversized_chunks", tmp_path) as (
        workdir,
        _,
    ):
        # The first write preserves the configured chunk shape even when it exceeds the
        # initial data extent; a later resume must validate that stored chunk metadata.
        store = str(workdir / "clamped_chunk.zarr")
        chunk_shape = {"timestamp": 24, "lat": 2, "lon": 1000}
        first = pd.date_range("2024-01-01", periods=2, freq="h")
        _append(store, {"G1": list(first)}, chunk_shape=chunk_shape)

        stored = zarr.open_array(store, path="G1/FWI", mode="r")
        assert tuple(stored.chunks) == (24, 2, 1000)

        # A fresh run has no in-process resume cache, so the resume path reads stored
        # metadata and validates chunks (this is the bug's trigger).
        clear_resume_cache()

        later = pd.date_range("2024-01-01T02:00:00", periods=1, freq="h")
        metrics = _append(store, {"G1": list(later)}, chunk_shape=chunk_shape, resume_existing=True)
        assert metrics["batch_processing"]["timestamps_written"] == 1
        ds = xr.open_zarr(store, group="G1", consolidated=False)
        assert ds.sizes["timestamp"] == 3


@pytest.mark.unit
def test_chunk_alignment_reporting(subtests: Any, tmp_path: Path) -> None:
    chunk_shape = {"timestamp": 2, "lat": 2, "lon": 3}
    with table_row(subtests, "alignment_warning_logged", tmp_path) as (workdir, _):
        store = str(workdir / "unaligned.zarr")
        ts = pd.date_range("2024-01-01", periods=3, freq="h")
        logger = logging.getLogger("test.alignment")
        with patch.object(logger, "warning") as mock_warn:
            _append(store, {"G1": list(ts)}, chunk_shape=chunk_shape, logger=logger)
        assert mock_warn.call_count >= 1
        warn_msg = mock_warn.call_args_list[0][0][0]
        assert "unaligned" in warn_msg.lower()

    with table_row(subtests, "aligned_no_warning", tmp_path) as (workdir, _):
        store = str(workdir / "aligned.zarr")
        ts = pd.date_range("2024-01-01", periods=4, freq="h")
        logger = logging.getLogger("test.aligned")
        with patch.object(logger, "warning") as mock_warn:
            _append(store, {"G1": list(ts)}, chunk_shape=chunk_shape, batch_size=2, logger=logger)
        for call in mock_warn.call_args_list:
            if call[0]:
                assert "unaligned" not in call[0][0].lower()

    with table_row(subtests, "coverage_aligned_field", tmp_path) as (workdir, _):
        store = str(workdir / "aligned_field.zarr")
        ts = pd.date_range("2024-01-01", periods=4, freq="h")
        metrics = _append(
            store,
            {"G1": list(ts)},
            arrays_for_group=_fwi,
            chunk_shape=chunk_shape,
            batch_size=2,
        )
        assert metrics["coverage"][0]["aligned"] is True

        store2 = str(workdir / "unaligned_field.zarr")
        ts2 = pd.date_range("2024-01-01", periods=3, freq="h")
        metrics2 = _append(
            store2, {"G1": list(ts2)}, arrays_for_group=_fwi, chunk_shape=chunk_shape
        )
        assert metrics2["coverage"][0]["aligned"] is False


@pytest.mark.unit
def test_shard_validation(subtests: Any, tmp_path: Path) -> None:
    ts1 = pd.date_range("2024-01-01", periods=2, freq="h")
    ts2 = pd.date_range("2024-01-01T02:00:00", periods=2, freq="h")

    with table_row(subtests, "shard_validation_runs", tmp_path) as (workdir, _):
        store = str(workdir / "shard_ok.zarr")
        shard = {"timestamp": 2, "lat": 2, "lon": 3}
        _append(store, {"G1": list(ts1)}, shard_shape=shard)
        metrics = _append(store, {"G1": list(ts2)}, shard_shape=shard)
        assert metrics["batch_processing"]["timestamps_written"] == 2

    with table_row(subtests, "shard_validation_mismatch_raises", tmp_path) as (workdir, _):
        store = str(workdir / "shard_bad.zarr")
        big = _dataset_factory(nlat=20, nlon=20)
        _append(
            store,
            {"G1": list(ts1)},
            dataset_for_batch=big,
            shard_shape={"timestamp": 2, "lat": 20, "lon": 20},
        )
        with pytest.raises(ValueError, match="shard_shape"):
            _append(
                store,
                {"G1": list(ts2)},
                dataset_for_batch=big,
                shard_shape={"timestamp": 2, "lat": 10, "lon": 10},
            )
