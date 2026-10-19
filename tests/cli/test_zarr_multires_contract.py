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

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from firecube.cli.main import cli


def _invoke_multires(args: list[str]):
    return CliRunner().invoke(cli, ["zarr", "multires", *args])


def test_coherence_mismatch_rejected() -> None:
    result = _invoke_multires(
        [
            "--target",
            "file:///tmp/x.zarr",
            "--product-name",
            "pn",
            "--storage-type",
            "s3",
            "--storage-driver",
            "fsspec",
            "--resolutions",
            "2",
        ]
    )

    assert result.exit_code == 2
    assert "incompatible" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("had_snapshot", [False, True], ids=["creates_none", "refreshes_existing"])
def test_multires_snapshot_never_created_and_refreshed_when_present(
    tmp_path: Path, had_snapshot: bool
) -> None:
    store = tmp_path / "x.zarr"
    ds = xr.Dataset(
        {"FWI": (["lat", "lon"], np.ones((2, 2), dtype="f4"))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    ds.to_zarr(store, zarr_format=3, consolidated=had_snapshot)

    flags = ["--storage-type", "local", "--storage-driver", "fsspec", "--resolutions", "45"]
    result = _invoke_multires(["--target", store.as_uri(), "--product-name", "pn", *flags])

    assert result.exit_code == 0, result.output
    snapshot = json.loads((store / "zarr.json").read_text()).get("consolidated_metadata")
    if had_snapshot:
        assert {"multires/45.0deg", "multires/45.0deg/FWI"} <= set(snapshot["metadata"])
    else:
        assert snapshot is None


def test_case_insensitive_flags_accepted(tmp_path: Path) -> None:
    result = _invoke_multires(
        [
            "--target",
            (tmp_path / "x.zarr").as_uri(),
            "--product-name",
            "pn",
            "--storage-type",
            "LOCAL",
            "--storage-driver",
            "FSSPEC",
            "--resolutions",
            "2",
        ]
    )

    assert "Invalid value" not in result.output
