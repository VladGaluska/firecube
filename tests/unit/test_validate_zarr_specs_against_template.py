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

from typing import Any

import pytest

from firecube.core.errors import ConfigurationError
from firecube.ingestor.templates.config import (
    ZarrTemplateConfig,
    validate_zarr_specs_against_template,
)
from firecube.ingestor.templates.direct_zarr import ZarrArraySpec

_RAISES: tuple[tuple[str, str, Any], ...] = (
    (
        "test_compression_false_with_per_array_compressors_raises",
        "compressors",
        ({"name": "blosc"},),
    ),
    (
        "test_compression_false_with_per_array_filters_raises",
        "filters",
        ({"name": "bitround", "configuration": {"keepbits": 8}},),
    ),
    ("test_compression_false_with_per_array_serializer_raises", "serializer", {"name": "bytes"}),
    ("test_compression_false_with_empty_compressors_raises", "compressors", ()),
)
_SUCCEEDS: tuple[tuple[str, bool, dict[str, Any]], ...] = (
    ("test_compression_false_without_any_codec_fields_succeeds", False, {}),
    ("test_compression_true_with_empty_compressors_succeeds", True, {"compressors": ()}),
    (
        "test_compression_true_with_per_array_compressors_succeeds",
        True,
        {"compressors": ({"name": "blosc"},)},
    ),
    ("test_compression_true_without_any_codec_fields_succeeds", True, {}),
)


@pytest.mark.unit
def test_compression_false_with_per_array_codec_raises(subtests: pytest.Subtests) -> None:
    template = ZarrTemplateConfig(zarr_compression=False)
    for row_id, field, value in _RAISES:
        with subtests.test(msg=row_id):
            spec = ZarrArraySpec(name="counts", shape=(10, 10), dtype="i4", **{field: value})
            with pytest.raises(ConfigurationError) as excinfo:
                validate_zarr_specs_against_template([spec], template)
            message = str(excinfo.value)
            assert "'counts'" in message
            assert field in message
            assert "zarr_compression=False" in message


@pytest.mark.unit
def test_compatible_codec_declarations_succeed(subtests: pytest.Subtests) -> None:
    for row_id, compression, codecs in _SUCCEEDS:
        with subtests.test(msg=row_id):
            spec = ZarrArraySpec(name="counts", shape=(10, 10), dtype="i4", **codecs)
            template = ZarrTemplateConfig(zarr_compression=compression)
            validate_zarr_specs_against_template([spec], template)


@pytest.mark.unit
def test_multiple_specs_only_offender_name_in_error() -> None:
    clean_spec = ZarrArraySpec(name="latitude", shape=(10,), dtype="f4")
    offender = ZarrArraySpec(
        name="radiance",
        shape=(10, 10),
        dtype="f4",
        compressors=({"name": "blosc"},),
    )
    other_clean = ZarrArraySpec(name="longitude", shape=(10,), dtype="f4")
    template = ZarrTemplateConfig(zarr_compression=False)
    with pytest.raises(ConfigurationError) as excinfo:
        validate_zarr_specs_against_template(
            [clean_spec, offender, other_clean],
            template,
        )
    message = str(excinfo.value)
    assert "'radiance'" in message
    assert "'latitude'" not in message
    assert "'longitude'" not in message
