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

"""Integer axis resolver contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from firecube.core import index_resolve
from firecube.core.index_spec import IntegerAxis
from tests.helpers.rows import table_row


def _resolver(size: int = 8):
    return index_resolve.IntegerResolver(axis=IntegerAxis(slot_count=size))


def test_resolver_for_integer_axis_returns_integer_resolver() -> None:
    resolver = index_resolve._resolver_for(IntegerAxis(slot_count=8))

    assert isinstance(resolver, index_resolve.IntegerResolver)


def test_integer_resolver_accepts_in_range_integers(subtests, tmp_path: Path) -> None:
    rows = [
        ("position_maps_integer_coordinate_to_same_slot", "position", 5, 5),
        ("position_accepts_numpy_integer", "position", np.int64(7), 7),
        ("coordinate_maps_integer_index_to_same_coordinate", "coordinate", 3, 3),
    ]
    for row_id, method, arg, expected in rows:
        with table_row(subtests, row_id, tmp_path):
            assert getattr(_resolver(), method)(arg) == expected


def test_integer_resolver_rejects_out_of_contract_values(subtests, tmp_path: Path) -> None:
    range_error = (IndexError, r"0.*8")
    bool_error = (TypeError, "bool is explicitly rejected")
    rows = [
        ("position_rejects_negative_coordinate", "position", -1, *range_error),
        (
            "position_rejects_coordinate_at_size",
            "position",
            8,
            IndexError,
            "coordinate 8 out of range",
        ),
        ("position_rejects_bool_coordinate", "position", True, *bool_error),
        (
            "position_rejects_non_integral_coordinate",
            "position",
            1.5,
            TypeError,
            "coordinate must be an integral type",
        ),
        ("coordinate_rejects_negative_index", "coordinate", -1, *range_error),
        ("coordinate_rejects_bool_index", "coordinate", False, *bool_error),
    ]
    for row_id, method, arg, exc, match in rows:
        with table_row(subtests, row_id, tmp_path), pytest.raises(exc, match=match):
            getattr(_resolver(), method)(arg)
