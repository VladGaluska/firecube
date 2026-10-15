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

import importlib

import pytest

ProductResolver = importlib.import_module("firecube.core.product.resolver").ProductResolver


def test_resolve_s3_uri_returns_product_identity() -> None:
    identity = ProductResolver.resolve(
        "s3://bucket/2026/TEST_PRODUCT.zarr", "zarr", product_name="TEST_PRODUCT"
    )

    assert identity.product_name == "TEST_PRODUCT"
    assert identity.product_uri.to_str() == "s3://bucket/2026/TEST_PRODUCT.zarr"
    assert identity.control_root_uri.to_str() == "s3://bucket/2026/TEST_PRODUCT.zarr/.firecube"


def test_resolve_file_uri_returns_product_identity() -> None:
    identity = ProductResolver.resolve("file:///abs/path/cube.zarr", "zarr", product_name="cube")

    assert identity.product_name == "cube"
    assert identity.product_uri.to_str() == "file:///abs/path/cube.zarr"
    assert identity.product_uri.authority is None


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        pytest.param("TEST_PRODUCT.zarr", "full URI", id="bare_name_rejected_with_full_uri_hint"),
        pytest.param("/abs/path", "file://", id="absolute_path_mentions_file_uri"),
        pytest.param("./relative", "rejected", id="relative_path_mentions_rejected"),
        pytest.param("ftp://host/path", "unsupported", id="invalid_uri_propagates_parse_error"),
    ],
)
def test_resolve_rejects_non_uri_inputs(uri: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ProductResolver.resolve(uri, "zarr", product_name="p")
