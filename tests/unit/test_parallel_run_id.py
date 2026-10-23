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

from firecube.ingestor.runtime.parallel_run_id import derive_pod_run_id

_POD_RUN_ID_ROWS: tuple[tuple[str, tuple[Any, ...], str], ...] = (
    # (row id, derive_pod_run_id args, expected run id)
    ("single_pod_returns_base_unchanged", ("run-abc", None, None), "run-abc"),
    ("single_pod_partial_none_returns_base_unchanged", ("run-abc", 0, None), "run-abc"),
    ("single_pod_partial_end_only_returns_base", ("run-abc", None, 100), "run-abc"),
    ("parallel_adds_slot_signature", ("run-abc", 100, 200), "run-abc__slot=100-200"),
    ("zero_start_is_distinct_from_none", ("base", 0, 100), "base__slot=0-100"),
    ("slot_group_none_preserves_phase3_format", ("run", 0, 100, None), "run__slot=0-100"),
    ("slot_group_set_includes_group", ("run", 0, 100, "group_a"), "run__group=group_a__slot=0-100"),
    ("slot_group_set_no_range_returns_base", ("run", None, None, "group_a"), "run"),
    ("different_ranges_produce_different_ids", ("base", 100, 200), "base__slot=100-200"),
    (
        "different_groups_same_range_distinct_run_ids",
        ("run", 0, 100, "A"),
        "run__group=A__slot=0-100",
    ),
    (
        "different_groups_same_range_distinct_run_ids_b",
        ("run", 0, 100, "B"),
        "run__group=B__slot=0-100",
    ),
)


def test_derive_pod_run_id(subtests):
    for row_id, args, expected in _POD_RUN_ID_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            assert derive_pod_run_id(*args) == expected
