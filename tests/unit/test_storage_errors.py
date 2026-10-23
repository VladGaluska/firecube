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
from typing import Any

import pytest

from firecube.core.errors import StorageError
from tests.helpers.storage import make_test_session

storage_session_module = importlib.import_module("firecube.core.storage.session")
StorageUri = importlib.import_module("firecube.core.storage.uri").StorageUri


def _assert_cause_is_write_failure(error: StorageError) -> None:
    cause = error.__cause__
    assert isinstance(cause, OSError), f"wrong cause: {cause!r}"
    assert str(cause) == "write failed"


_UPLOAD_ROWS: tuple[tuple[str, dict[str, Any], str | None], ...] = (
    # (row id, make_test_session kwargs, target)
    ("local_session_upload_raises_typed_error", {}, None),
    (
        "s3_session_upload_raises_typed_error",
        {"protocol": "s3", "authority": "bucket"},
        "s3://bucket/product",
    ),
)


@pytest.mark.unit
def test_session_upload_raises_typed_error(subtests, monkeypatch, temp_workspace):
    class _BrokenFs:
        def open(self, *args, **kwargs):
            _ = (args, kwargs)
            raise OSError("write failed")

        def capabilities(self) -> set[type]:
            return set()

    def _fake_create_filesystem(_binding):
        return _BrokenFs()

    monkeypatch.setattr(storage_session_module, "create_filesystem", _fake_create_filesystem)

    source_file = temp_workspace / "sample.bin"
    source_file.write_bytes(b"abc")

    for row_id, session_kwargs, target in _UPLOAD_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            session = make_test_session(temp_workspace, **session_kwargs)
            destination = (
                StorageUri.parse(target)
                if target
                else StorageUri.from_local_path(temp_workspace / "output.zarr")
            )

            with pytest.raises(StorageError) as excinfo:
                session.upload_tree(
                    StorageUri.from_local_path(source_file), destination, parallel_workers=1
                )

            _assert_cause_is_write_failure(excinfo.value)
