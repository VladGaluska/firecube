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

import click
import pytest

from firecube.cli._uri_policy import (
    SCHEME_TO_STORAGE_TYPE,
    apply_smart_default,
    parse_and_validate_uri,
    resolve_storage_type,
    validate_uri_storage_coherence,
)
from firecube.core.storage.session import _storage_type_for_uri
from firecube.core.storage.uri import StorageUri


def test_scheme_mapping_matches_storage_session_for_file() -> None:
    uri = StorageUri.parse("file:///tmp/example.zarr")

    assert SCHEME_TO_STORAGE_TYPE["file"] == _storage_type_for_uri(uri)


def test_scheme_mapping_matches_storage_session_for_s3() -> None:
    uri = StorageUri.parse("s3://bucket/example.zarr")

    assert SCHEME_TO_STORAGE_TYPE["s3"] == _storage_type_for_uri(uri)


@pytest.mark.contract
@pytest.mark.parametrize(
    "raw",
    ["file:///abs/p.zarr", "file:///tmp/a b.zarr", "file:///tmp/a%20b.zarr", "s3://b/p", "s3://b"],
)
def test_parse_and_validate_uri_returns_the_path_literally(raw: str) -> None:
    assert parse_and_validate_uri(raw, param_hint="--target").to_str() == raw


_BARE = r"^URI scheme required \(file:// or s3://\)\."
_REFUSED = {
    "URI must not (be empty|contain control)": ["", "   ", "file:///tmp/a\tb.zarr"],
    _BARE + " Did you mean file://": ["/abs/path.zarr", "./rel/path.zarr", "rel/p.zarr", "x.zarr"],
    _BARE + " Did you mean file:///tmp/a b.zarr\\?$": ["/tmp/a b.zarr"],
    _BARE + "$": ["file:/tmp/a.zarr", "~firecube_no_such_user_17/a.zarr"],
    "not supported": [f"{s}://x" for s in ("gs", "GS", "abfs", "https", "memory", "ftp")],
    "must be lowercase .* Did you mean (file|s3)://": ["FILE:///a", "S3://b/x", "File:///x"],
    r"must not name a host \(got '(h|localhost)'\)": ["file://h/p", "file://localhost/a"],
    "must name a path, not the root directory": ["file://", "file:///"],
    "s3:// URI must name a bucket": ["s3:///missing-bucket"],
    r"must not contain '\?' or '#'": ["file:///a?x=1", "file:///a?", "file:///a#", "s3://b/x#y"],
}


@pytest.mark.contract
@pytest.mark.parametrize(("raw", "msg"), [(r, m) for m, rs in _REFUSED.items() for r in rs])
def test_parse_and_validate_uri_refuses_each_rule_naming_the_option(raw: str, msg: str) -> None:
    with pytest.raises(click.UsageError, match=msg) as refused:
        parse_and_validate_uri(raw, param_hint="--target")
    assert refused.value.format_message().startswith("Invalid value for '--target': ")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("file:///abs/path.zarr", "local"),
        ("s3://bucket/path.zarr", "s3"),
    ],
)
def test_resolve_storage_type_returns_scheme_mapping(raw: str, expected: str) -> None:
    assert resolve_storage_type(parse_and_validate_uri(raw, param_hint="--target")) == expected


@pytest.mark.parametrize(
    ("raw", "storage_type"),
    [
        ("file:///abs/path.zarr", "local"),
        ("s3://bucket/path.zarr", "s3"),
    ],
)
def test_validate_uri_storage_coherence_accepts_matches(raw: str, storage_type: str) -> None:
    validate_uri_storage_coherence(parse_and_validate_uri(raw, param_hint="--target"), storage_type)


@pytest.mark.parametrize(
    ("raw", "storage_type", "scheme"),
    [
        ("file:///abs/path.zarr", "s3", "file"),
        ("s3://bucket/path.zarr", "local", "s3"),
    ],
)
def test_validate_uri_storage_coherence_rejects_mismatches(
    raw: str, storage_type: str, scheme: str
) -> None:
    with pytest.raises(click.UsageError, match=f"incompatible with URI scheme '{scheme}'"):
        validate_uri_storage_coherence(
            parse_and_validate_uri(raw, param_hint="--target"), storage_type
        )


def test_apply_smart_default_infers_local_from_file_uri() -> None:
    uri = parse_and_validate_uri("file:///abs/path.zarr", param_hint="--target")
    assert apply_smart_default(uri, None) == "local"


def test_apply_smart_default_infers_s3_from_uri_scheme() -> None:
    uri = parse_and_validate_uri("s3://bucket/path.zarr", param_hint="--target")
    assert apply_smart_default(uri, None) == "s3"


def test_apply_smart_default_accepts_explicit_local() -> None:
    uri = parse_and_validate_uri("file:///abs/path.zarr", param_hint="--target")

    assert apply_smart_default(uri, "local") == "local"


def test_apply_smart_default_validates_after_resolution() -> None:
    uri = parse_and_validate_uri("file:///abs/path.zarr", param_hint="--target")

    with pytest.raises(click.UsageError, match="incompatible with URI scheme 'file'"):
        apply_smart_default(uri, "s3")


def test_apply_smart_default_accepts_explicit_s3_storage_type() -> None:
    uri = parse_and_validate_uri("s3://bucket/path.zarr", param_hint="--target")

    assert apply_smart_default(uri, "s3") == "s3"
