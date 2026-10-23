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

from firecube.ingestor.runtime.parallel_run_id import derive_pod_run_id

pytestmark = pytest.mark.unit


_ROWS = (
    (
        "test_slot_group_with_slash_encoded",
        "multires/0.5deg",
        "base__group=multires%2F0.5deg__slot=0-100",
    ),
    ("test_slot_group_with_equals_encoded", "key=value", "base__group=key%3Dvalue__slot=0-100"),
    ("test_slot_group_with_unicode_encoded", "ñoño", "base__group=%C3%B1o%C3%B1o__slot=0-100"),
    ("test_slot_group_simple_unchanged", "group_a", "base__group=group_a__slot=0-100"),
    ("test_slot_group_none_format_unchanged", None, "base__slot=0-100"),
)


def test_slot_group_encoding(subtests: pytest.Subtests) -> None:
    for row_id, group, expected in _ROWS:
        with subtests.test(msg=row_id):
            assert derive_pod_run_id("base", 0, 100, group) == expected


def test_encoded_run_id_is_path_segment_safe() -> None:
    run_id = derive_pod_run_id("base", 0, 100, "grp/sub?#")

    assert "/" not in run_id
    assert "?" not in run_id
    assert "#" not in run_id
    assert "grp%2Fsub%3F%23" in run_id
