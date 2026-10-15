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

import pytest

from firecube.ingestor.runtime.resume_guard import _ranges_overlap_inclusive_vs_halfopen

_OVERLAP_ROWS = (
    # (row id, inclusive range, half-open range, overlaps)
    ("back_to_back_disjoint", (0, 9), (10, 20), False),
    ("boundary_overlap_inclusive_end_meets_halfopen_start", (0, 10), (10, 20), True),
    ("complete_overlap", (5, 15), (0, 20), True),
    ("completely_disjoint_high", (100, 200), (0, 50), False),
)


@pytest.mark.unit
def test_ranges_overlap_inclusive_vs_halfopen(subtests):
    for row_id, inclusive, half_open, expected in _OVERLAP_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            assert _ranges_overlap_inclusive_vs_halfopen(inclusive, half_open) is expected
