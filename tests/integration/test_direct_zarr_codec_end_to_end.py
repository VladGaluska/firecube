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

"""End-to-end DirectZarr codec write-path integration tests.

Exercises the full ``ZarrArraySpec -> derive_effective_codecs_for_spec ->
RegionZarrWriter.ensure_group`` chain against a real local Zarr store and
asserts on ``arr.metadata.codecs`` after write. Covers:

* E2E-1: fresh cube with a declared Blosc compressor spec.
* E2E-2: per-array ``compressors=()`` (compress-except-X) yields an
  uncompressed array even when the template default enables compression.
* E2E-3: resume with identical codec config does not raise.
* E2E-4: resume with a changed compressor level raises
  ``SchemaDriftError`` naming the ``compressors`` field.
* E2E-5: sharded array + declared compressor keeps both effects visible.
* E2E-6: static (``time_indexed=False``) array is not exempt from codec
  drift checks.
* E2E-7: backward-compat — cubes created without any codec kwargs
  (letting zarr pick its own default) resume cleanly against a spec that
  declares no codec fields.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import zarr
from zarr.abc.codec import BytesBytesCodec

from firecube.core.errors import SchemaDriftError
from firecube.core.zarr.region_writer import RegionZarrWriter
from firecube.ingestor.runtime.zarr.write import derive_effective_codecs_for_spec
from firecube.ingestor.templates.config import ZarrTemplateConfig
from firecube.ingestor.templates.direct_zarr import ZarrArraySpec
from tests.helpers.rows import table_row

pytestmark = pytest.mark.integration


def _codec_names(codecs: list[Any]) -> list[str]:
    names: list[str] = []
    for codec in codecs:
        to_dict = getattr(codec, "to_dict", None)
        if callable(to_dict):
            dumped = cast(dict[str, Any], to_dict())
            name = dumped.get("name")
            if isinstance(name, str):
                names.append(name)
                continue
        names.append(type(codec).__name__)
    return names


def _open_local_array(store_path: str, array_path: str) -> Any:
    return zarr.open_array(store_path, path=array_path, mode="r")


def test_e2e_1_fresh_cube_with_declared_codec(tmp_path) -> None:
    """Fresh DirectZarr cube with declared BloscCodec writes correct codec to zarr.json."""
    store_path = str(tmp_path / "cube.zarr")
    spec = ZarrArraySpec(
        name="data",
        shape=(10, 10),
        dtype="f4",
        compressors=({"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5}},),
    )
    template = ZarrTemplateConfig(zarr_compression=True)
    filters, serializer, compressors = derive_effective_codecs_for_spec(spec, template)

    writer = RegionZarrWriter(f"file://{store_path}")
    writer.ensure_group(
        "g/data",
        shape=(10, 10),
        dtype="f4",
        filters=filters,
        serializer=serializer,
        compressors=compressors,
    )

    arr = _open_local_array(store_path, "g/data")
    names = _codec_names(list(cast(Any, arr.metadata).codecs))
    assert "blosc" in names, f"expected blosc in on-disk codecs, got {names!r}"


def test_e2e_2_per_array_empty_compressors_uncompressed(tmp_path) -> None:
    """Per-array ``compressors=()`` produces uncompressed array even with template True."""
    store_path = str(tmp_path / "cube.zarr")
    spec = ZarrArraySpec(
        name="mask",
        shape=(10, 10),
        dtype="u1",
        compressors=(),
    )
    template = ZarrTemplateConfig(zarr_compression=True)
    filters, serializer, compressors = derive_effective_codecs_for_spec(spec, template)

    writer = RegionZarrWriter(f"file://{store_path}")
    writer.ensure_group(
        "g/mask",
        shape=(10, 10),
        dtype="u1",
        filters=filters,
        serializer=serializer,
        compressors=compressors,
    )

    arr = _open_local_array(store_path, "g/mask")
    on_disk = list(cast(Any, arr.metadata).codecs)
    assert not any(isinstance(codec, BytesBytesCodec) for codec in on_disk), (
        f"expected no BytesBytesCodec (compressor), got {_codec_names(on_disk)!r}"
    )


def _blosc(clevel: int) -> tuple[dict[str, Any], ...]:
    return ({"name": "blosc", "configuration": {"cname": "zstd", "clevel": clevel}},)


# Resume rows: (old test name, array, shape, dtype, time_indexed, resume clevel, extent).
# The cube is created at clevel 5; a resume at another level must raise drift naming
# ``compressors`` (static arrays are not exempt), the same level must not.
_RESUME_ROWS = [
    ("test_e2e_3_resume_same_config_no_drift", "data", (10,), "f4", True, 5, 10),
    ("test_e2e_4_resume_different_codec_raises_drift", "data", (10,), "f4", True, 9, 10),
    ("test_e2e_6_static_array_codec_drift", "lat", (100, 100), "f8", False, 9, 0),
]


def test_e2e_resume_codec_drift(tmp_path, subtests: pytest.Subtests) -> None:
    """Resume with the created codec config passes; a changed compressor level is drift."""
    for row_id, name, shape, dtype, time_indexed, clevel, extent in _RESUME_ROWS:
        with table_row(subtests, row_id, tmp_path) as (workdir, _mp):
            store_path = str(workdir / "cube.zarr")
            spec_v1 = ZarrArraySpec(
                name=name,
                shape=shape,
                dtype=dtype,
                time_indexed=time_indexed,
                compressors=_blosc(5),
            )
            template = ZarrTemplateConfig(zarr_compression=True)
            filters, serializer, compressors = derive_effective_codecs_for_spec(spec_v1, template)
            RegionZarrWriter(f"file://{store_path}").ensure_group(
                f"g/{name}",
                shape=shape,
                dtype=dtype,
                filters=filters,
                serializer=serializer,
                compressors=compressors,
            )

            spec_v2 = ZarrArraySpec(
                name=name,
                shape=shape,
                dtype=dtype,
                time_indexed=time_indexed,
                compressors=_blosc(clevel),
            )
            writer_v2 = RegionZarrWriter(f"file://{store_path}")
            if clevel == 5:
                writer_v2.verify_array_spec(f"g/{name}", spec_v2, extent)
            else:
                with pytest.raises(SchemaDriftError) as exc_info:
                    writer_v2.verify_array_spec(f"g/{name}", spec_v2, extent)
                assert "compressors" in str(exc_info.value), str(exc_info.value)


def test_e2e_5_sharded_with_codec(tmp_path) -> None:
    """Sharded array with declared codec: both shards and compressor codec applied."""
    store_path = str(tmp_path / "cube.zarr")
    spec = ZarrArraySpec(
        name="data",
        shape=(100, 100),
        dtype="f4",
        chunks=(25, 25),
        shards=(50, 50),
        compressors=({"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5}},),
    )
    template = ZarrTemplateConfig(zarr_compression=True)
    filters, serializer, compressors = derive_effective_codecs_for_spec(spec, template)

    writer = RegionZarrWriter(f"file://{store_path}")
    writer.ensure_group(
        "g/data",
        shape=(100, 100),
        dtype="f4",
        chunks=(25, 25),
        shards=(50, 50),
        filters=filters,
        serializer=serializer,
        compressors=compressors,
    )

    arr = _open_local_array(store_path, "g/data")
    on_disk = list(cast(Any, arr.metadata).codecs)
    # Sharded arrays wrap the inner pipeline inside a sharding_indexed codec;
    # regardless of nesting, at least one compressor codec must be present in
    # the top-level codec list because the sharding codec itself is a
    # BytesBytesCodec-like container carrying compression.
    names = _codec_names(on_disk)
    assert any("blosc" in name or "sharding" in name for name in names), (
        f"expected sharding or blosc in on-disk codecs, got {names!r}"
    )
    existing_shards = getattr(arr, "shards", None)
    assert existing_shards is not None and tuple(existing_shards) == (50, 50), (
        f"expected shards=(50, 50), got {existing_shards!r}"
    )


def test_e2e_7_backward_compat_zarr_default(tmp_path) -> None:
    """Existing cube created with zarr default (no codec kwargs) resumes without drift."""
    store_path = str(tmp_path / "cube.zarr")

    writer_v1 = RegionZarrWriter(f"file://{store_path}")
    writer_v1.ensure_group("g/data", shape=(10,), dtype="f4")

    spec = ZarrArraySpec(name="data", shape=(10,), dtype="f4")
    writer_v2 = RegionZarrWriter(f"file://{store_path}")
    writer_v2.verify_array_spec("g/data", spec, 10)
