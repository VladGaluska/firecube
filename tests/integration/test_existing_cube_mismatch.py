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

import pytest


@pytest.mark.integration
def test_single_group_mismatch_pinpoints_group(tmp_path):
    """Existing 'timestamp' cube + plugin declares 'time' raises exact guidance."""
    from firecube.ingestor.errors import ConfigurationError
    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility
    from tests.fixtures.cf_dataset_fixtures import make_legacy_timestamp_dataset

    target = str(tmp_path / "legacy.zarr")
    ds = make_legacy_timestamp_dataset()
    ds.to_zarr(target, mode="w", zarr_format=3, consolidated=False)

    with pytest.raises(ConfigurationError) as exc:
        verify_dim_compatibility(target, "time", group_paths=["."], storage_config=None)

    assert str(exc.value) == (
        f"Existing cube at {target} group '.' uses time dimension 'timestamp' "
        "but plugin declared 'time'.\n"
        "Refusing to append. Rebuild the cube with the declared time dimension "
        "or migrate the existing cube before appending."
    )


@pytest.mark.integration
def test_multi_group_pinpoints_offending_group(tmp_path):
    """Cube with 2 groups, one wrong dim: error names the offending group."""
    from firecube.ingestor.errors import ConfigurationError
    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility
    from tests.fixtures.cf_dataset_fixtures import (
        make_cf_compliant_dataset,
        make_legacy_timestamp_dataset,
    )

    target = str(tmp_path / "multi.zarr")
    ok_ds = make_cf_compliant_dataset(time_dim="time")
    bad_ds = make_legacy_timestamp_dataset()
    ok_ds.to_zarr(target, group="OK_GROUP", mode="w", zarr_format=3, consolidated=False)
    bad_ds.to_zarr(target, group="BAD_GROUP", mode="a", zarr_format=3, consolidated=False)

    with pytest.raises(ConfigurationError) as exc:
        verify_dim_compatibility(
            target,
            "time",
            group_paths=["OK_GROUP", "BAD_GROUP"],
            storage_config=None,
        )

    msg = str(exc.value)
    assert "BAD_GROUP" in msg
    assert "OK_GROUP" not in msg
    assert "Refusing to append" in msg


@pytest.mark.integration
def test_new_cube_no_error(tmp_path):
    """Target does not exist: no error."""
    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility

    target = str(tmp_path / "does_not_exist.zarr")
    verify_dim_compatibility(target, "time", group_paths=["."], storage_config=None)


@pytest.mark.integration
def test_matching_cube_no_error(tmp_path):
    """Existing 'time' cube + plugin declares 'time': no error."""
    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility
    from tests.fixtures.cf_dataset_fixtures import make_cf_compliant_dataset

    target = str(tmp_path / "matching.zarr")
    ds = make_cf_compliant_dataset(time_dim="time")
    ds.to_zarr(target, mode="w", zarr_format=3, consolidated=False)

    verify_dim_compatibility(target, "time", group_paths=["."], storage_config=None)


@pytest.mark.integration
def test_backcompat_existing_timestamp_with_default_declaration(tmp_path):
    """Existing 'timestamp' cube + plugin declares 'timestamp': no error."""
    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility
    from tests.fixtures.cf_dataset_fixtures import make_legacy_timestamp_dataset

    target = str(tmp_path / "ts.zarr")
    ds = make_legacy_timestamp_dataset()
    ds.to_zarr(target, mode="w", zarr_format=3, consolidated=False)

    verify_dim_compatibility(target, "timestamp", group_paths=["."], storage_config=None)


@pytest.mark.integration
def test_auxiliary_coordinates_do_not_create_ambiguous_time_dim(tmp_path):
    """2-D auxiliary coordinates are non-temporal and must not be treated as time axes."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility

    target = str(tmp_path / "native_test_product.zarr")
    ds = xr.Dataset(
        data_vars={
            "FWI": (
                ("timestamp", "lat", "lon"),
                np.zeros((3, 2, 3), dtype=np.float32),
                {"coordinates": "msg_lat msg_lon"},
            )
        },
        coords={
            "timestamp": pd.date_range("2025-01-01", periods=3),
            "lat": np.array([10.0, 20.0], dtype=np.float32),
            "lon": np.array([30.0, 40.0, 50.0], dtype=np.float32),
            "msg_lat": (("lat", "lon"), np.zeros((2, 3), dtype=np.float32)),
            "msg_lon": (("lat", "lon"), np.zeros((2, 3), dtype=np.float32)),
        },
    )
    ds.to_zarr(target, group="F024", mode="w", zarr_format=3, consolidated=False)

    verify_dim_compatibility(target, "timestamp", group_paths=["F024"], storage_config=None)


@pytest.mark.integration
def test_unknown_data_array_time_dimension_does_not_block_verification(tmp_path):
    """Data arrays whose dim names are NOT in {time, timestamp} are treated as
    static spatial arrays and skipped by the time-dim consistency check.

    Contract note: prior to the static-array support fix this test was named
    ``test_unknown_data_array_time_dimension_fails_closed`` and asserted that
    arrays with unknown time-like dims (e.g. ``valid_time``) raised a
    ``ConfigurationError`` with message ``"cannot determine time dimension"``.
    That contract was inverted to support static lat/lon arrays
    (dims ``['ny', 'nx']``) that legitimately have no time dim. The trade-off:
    typos in time-dim names (e.g. ``valid_time`` instead of ``time``) are no
    longer surfaced by this validator; users must rely on plugin-level schema
    declaration to catch such mistakes.
    """
    import numpy as np
    import xarray as xr

    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility

    target = str(tmp_path / "unknown_dim.zarr")
    ds = xr.Dataset(
        {"temperature": (("valid_time", "lat"), np.zeros((2, 3), dtype=np.float32))},
        coords={"valid_time": [0.0, 1.0], "lat": [10.0, 20.0, 30.0]},
    )
    ds.to_zarr(target, mode="w", zarr_format=3, consolidated=False)

    # No raise — array has no time-like dim, treated as static and skipped.
    verify_dim_compatibility(target, "time", group_paths=["."], storage_config=None)


@pytest.mark.integration
def test_ambiguous_existing_cube_raises(tmp_path):
    """A data array with both 'time' and 'timestamp' dims is ambiguous."""
    import numpy as np
    import xarray as xr

    from firecube.ingestor.errors import ConfigurationError
    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility

    target = str(tmp_path / "ambiguous.zarr")
    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "timestamp", "lat"],
                np.zeros((2, 2, 3), dtype=np.float32),
            )
        },
        coords={"time": [0, 1], "timestamp": [0, 1], "lat": [10, 20, 30]},
    )
    ds.to_zarr(target, mode="w", zarr_format=3, consolidated=False)

    with pytest.raises(ConfigurationError, match="contains both 'time' and 'timestamp'"):
        verify_dim_compatibility(target, "time", group_paths=["."], storage_config=None)


@pytest.mark.integration
def test_whole_cube_ambiguity_detected(tmp_path):
    """Same group has two arrays with different time dim names: ambiguous-state error."""
    import numpy as np
    import xarray as xr

    from firecube.ingestor.errors import ConfigurationError
    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility

    target = str(tmp_path / "amb.zarr")
    ds1 = xr.Dataset(
        {"a": (("time", "x"), np.zeros((2, 3)))},
        coords={"time": [0.0, 1.0]},
    )
    ds1.to_zarr(target, mode="w", zarr_format=3, consolidated=False)
    ds2 = xr.Dataset(
        {"b": (("timestamp", "x"), np.zeros((2, 3)))},
        coords={"timestamp": [0.0, 1.0]},
    )
    ds2.to_zarr(target, mode="a", zarr_format=3, consolidated=False)

    with pytest.raises(ConfigurationError) as exc:
        verify_dim_compatibility(target, "time", group_paths=["."], storage_config=None)
    msg = str(exc.value).lower()
    assert any(tok in msg for tok in ["ambiguous", "both", "conflicting"])


@pytest.mark.integration
def test_dim_check_lists_only_the_group_top_level(tmp_path, monkeypatch):
    """The check reads one level of the group: its listings must not grow with chunk objects."""
    import numpy as np
    import xarray as xr

    from firecube.ingestor.runtime.zarr import existing_cube_check

    target = tmp_path / "many_chunks.zarr"
    ds = xr.Dataset(
        {name: (("time", "x"), np.zeros((2, 3))) for name in ("a", "b", "c")},
        coords={"time": [0.0, 1.0]},
    )
    ds.to_zarr(str(target), mode="w", zarr_format=3, consolidated=False)
    for name in ("a", "b", "c"):
        for i in range(500):  # chunk objects the check has no reason to list
            chunk = target / name / "c" / str(i) / "0"
            chunk.parent.mkdir(parents=True, exist_ok=True)
            chunk.write_bytes(b"x")

    listed = {"entries": 0}

    class _CountingFs:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, attr):
            member = getattr(self._inner, attr)
            if attr not in ("find", "ls", "walk", "glob"):
                return member

            def counted(*args, **kwargs):
                result = list(member(*args, **kwargs))
                listed["entries"] += len(result)
                return result

            return counted

    real_create = existing_cube_check.create_filesystem
    monkeypatch.setattr(
        existing_cube_check, "create_filesystem", lambda binding: _CountingFs(real_create(binding))
    )

    outcome = existing_cube_check.verify_dim_compatibility(
        str(target), "time", group_paths=["."], storage_config=None
    )

    assert outcome is None  # a matching cube passes
    arrays = 4  # a, b, c and the time coordinate
    assert listed["entries"] < 12, (
        f"dim check listed {listed['entries']} entries for {arrays} arrays"
    )


@pytest.mark.integration
@pytest.mark.s3
@pytest.mark.parametrize("change", ["created_after_listing", "deleted_after_listing"])
def test_dim_check_reads_arrays_changed_after_listing(change, monkeypatch, tmp_path):
    """An array another writer creates or deletes after this process listed the group counts."""
    import boto3
    import s3fs
    from moto.server import ThreadedMotoServer

    from firecube.core.filesystem import create_filesystem
    from firecube.core.storage.uri import StorageUri
    from firecube.ingestor.errors import ConfigurationError
    from firecube.ingestor.runtime.zarr.existing_cube_check import verify_dim_compatibility
    from tests.helpers.storage import make_test_binding

    server = ThreadedMotoServer(port=0)
    server.start()
    try:
        host, port = server.get_host_and_port()
        for name, value in {
            "AWS_ENDPOINT_URL": f"http://{host}:{port}",
            "AWS_ACCESS_KEY_ID": "t",
            "AWS_SECRET_ACCESS_KEY": "t",
            "AWS_DEFAULT_REGION": "us-east-1",
        }.items():
            monkeypatch.setenv(name, value)
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="bkt")
        s3.put_object(Bucket="bkt", Key="cube.zarr/g/zarr.json", Body=b'{"node_type": "group"}')
        s3.put_object(Bucket="bkt", Key="cube.zarr/g/a/c/0/0", Body=b"x")
        array = b'{"node_type": "array", "dimension_names": ["timestamp", "x"]}'
        if change == "deleted_after_listing":
            s3.put_object(Bucket="bkt", Key="cube.zarr/g/a/zarr.json", Body=array)
        s3fs.S3FileSystem.clear_instance_cache()
        binding = make_test_binding(tmp_path, product="cube.zarr", protocol="s3", authority="bkt")
        create_filesystem(binding).find(
            StorageUri(protocol="s3", authority="bkt", path="/cube.zarr/g")
        )
        if change == "created_after_listing":
            s3.put_object(Bucket="bkt", Key="cube.zarr/g/a/zarr.json", Body=array)
            with pytest.raises(ConfigurationError, match="timestamp"):
                verify_dim_compatibility("s3://bkt/cube.zarr", "time", ["g"], binding)
        else:
            s3.delete_object(Bucket="bkt", Key="cube.zarr/g/a/zarr.json")
            assert verify_dim_compatibility("s3://bkt/cube.zarr", "time", ["g"], binding) is None
    finally:
        server.stop()
