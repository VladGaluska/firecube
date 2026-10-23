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

"""zarr_codecs=[...] requires zarr_compression=True; False+codec is a contradiction."""

from __future__ import annotations

import pytest

from firecube.core.errors import ConfigurationError
from firecube.ingestor.templates.config import ZarrTemplateConfig
from tests.helpers.rows import table_row

pytestmark = pytest.mark.unit


def test_false_and_codecs_raises() -> None:
    with pytest.raises(
        ConfigurationError, match="zarr_compression=False conflicts with zarr_codecs"
    ):
        ZarrTemplateConfig(
            zarr_compression=False, zarr_codecs=[{"name": "blosc", "configuration": {}}]
        )


def test_error_names_both_fields() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        ZarrTemplateConfig(
            zarr_compression=False, zarr_codecs=[{"name": "blosc", "configuration": {}}]
        )

    msg = str(exc_info.value)
    assert "zarr_compression" in msg
    assert "zarr_codecs" in msg


def test_compatible_combinations_accepted(subtests, tmp_path) -> None:
    for row_id, compression, codecs in [
        ("test_true_and_codecs_accepted", True, [{"name": "blosc", "configuration": {}}]),
        ("test_true_and_none_accepted", True, None),
        ("test_false_and_none_accepted", False, None),
    ]:
        with table_row(subtests, row_id, tmp_path):
            cfg = ZarrTemplateConfig(zarr_compression=compression, zarr_codecs=codecs)
            assert cfg.zarr_compression is compression
            assert cfg.zarr_codecs == codecs
