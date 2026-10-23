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

from firecube.ingestor.errors import ConfigurationError
from firecube.ingestor.types.planned_range import validate_chunk_alignment
from tests.helpers.rows import table_row

pytestmark = pytest.mark.unit


def test_validate_chunk_alignment_passes(subtests, tmp_path) -> None:
    for row_id, start, end, chunks, global_expected in [
        ("test_aligned_range_passes", 0, 100, {"g": [(100,)]}, None),
        ("test_misaligned_end_terminal_passes", 900, 950, {"g": [(100,)]}, {"g": 950}),
    ]:
        with table_row(subtests, row_id, tmp_path):
            validate_chunk_alignment(start, end, chunks, global_expected=global_expected)


def test_validate_chunk_alignment_fails(subtests, tmp_path) -> None:
    for row_id, start, end, chunks, global_expected in [
        ("test_misaligned_start_fails", 50, 200, {"g": [(100,)]}, {"g": 300}),
        ("test_misaligned_end_non_terminal_fails", 100, 250, {"g": [(100,)]}, {"g": 500}),
        (
            "test_misaligned_end_terminal_without_global_expected_fails",
            900,
            950,
            {"g": [(100,)]},
            None,
        ),
        (
            "test_multiple_groups_one_terminal_one_not",
            100,
            150,
            {"a": [(50,)], "b": [(100,)]},
            {"a": 150, "b": 500},
        ),
    ]:
        with table_row(subtests, row_id, tmp_path), pytest.raises(ConfigurationError):
            validate_chunk_alignment(start, end, chunks, global_expected=global_expected)
