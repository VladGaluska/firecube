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

from firecube.core.credentials import Credentials  # pyright: ignore[reportMissingImports]
from firecube.core.storage.uri import StorageUri

ProductIdentity = importlib.import_module("firecube.core.product.identity").ProductIdentity
StorageBinding = importlib.import_module("firecube.core.storage.binding").StorageBinding
StorageDriverConfig = importlib.import_module(
    "firecube.core.storage.driver_config"
).StorageDriverConfig
StorageCacheKey = importlib.import_module("firecube.core.storage.cache_key").StorageCacheKey


def _binding(
    *,
    access_key: str | None = "ak",
    secret_key: str | None = "sk",
    endpoint_url: str | None = "https://s3.example.com",
):
    identity = ProductIdentity.from_uri(
        StorageUri.parse("s3://bucket/data/x.zarr"), "zarr", product_name="x"
    )
    driver = StorageDriverConfig(
        driver="obstore",
        endpoint_url=endpoint_url,
        credentials=None
        if access_key is None and secret_key is None
        else Credentials(access_key=access_key, secret_key=secret_key),
        region="eu-west-1",
    )
    return StorageBinding(identity=identity, driver=driver)


def test_same_fields_produce_same_key() -> None:
    left = _binding()
    right = _binding()

    assert StorageCacheKey.from_binding(left) == StorageCacheKey.from_binding(right)


_DIFFERING_FIELD_ROWS = (
    ("test_different_access_key_changes_key", {"access_key": "ak-1"}, {"access_key": "ak-2"}),
    ("test_different_secret_key_changes_key", {"secret_key": "sk-1"}, {"secret_key": "sk-2"}),
    (
        "test_different_endpoint_changes_key",
        {"endpoint_url": "https://one.example.com"},
        {"endpoint_url": "https://two.example.com"},
    ),
)


def test_different_field_changes_key(subtests) -> None:
    for row_id, left, right in _DIFFERING_FIELD_ROWS:
        with subtests.test(msg=row_id):
            assert StorageCacheKey.from_binding(_binding(**left)) != StorageCacheKey.from_binding(
                _binding(**right)
            )


def test_anonymous_vs_explicit_credentials_change_key() -> None:
    anonymous = StorageCacheKey.from_binding(_binding(access_key=None, secret_key=None))
    explicit = StorageCacheKey.from_binding(_binding())

    assert anonymous != explicit


def test_cache_key_is_hashable() -> None:
    key = StorageCacheKey.from_binding(_binding())

    assert key in {key}
    assert isinstance(hash(key), int)


def test_repr_is_safe_and_contains_fingerprint_not_raw_credentials() -> None:
    key = StorageCacheKey.from_binding(_binding())
    text = repr(key)

    assert "ak" not in text
    assert "sk" not in text
    assert key.credential_fingerprint in text
