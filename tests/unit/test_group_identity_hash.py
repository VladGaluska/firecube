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

from typing import Literal

import numpy as np
import pytest

from firecube.core.index_resolve import _compute_group_identity_hash
from firecube.core.index_spec import IntegerAxis, IrregularTimeAxis, RegularTimeAxis

pytestmark = pytest.mark.unit


def _make_regular(
    *,
    coordinate: str = "timestamp",
    epoch: str = "2024-01-01T00:00:00Z",
    cadence_s: int = 600,
    mode: Literal["exact", "floor"] = "exact",
    slot_count: int | None = 100,
    end_date: str | None = None,
) -> RegularTimeAxis:
    return RegularTimeAxis(
        coordinate=coordinate,
        epoch=epoch,
        cadence_s=cadence_s,
        mode=mode,
        slot_count=slot_count,
        end_date=end_date,
    )


def test_regular_axis_hash_is_deterministic() -> None:
    axis = _make_regular()
    a = _compute_group_identity_hash(axis, 100, "datetime64[ns]")
    b = _compute_group_identity_hash(axis, 100, "datetime64[ns]")
    assert a == b
    assert len(a) == 64


NS = "datetime64[ns]"
HASH_DIFFERENCES = (
    (
        "test_regular_axis_hash_differs_when_slot_count_differs",
        (_make_regular(), 100, NS),
        (_make_regular(), 101, NS),
    ),
    (
        "test_regular_axis_hash_differs_when_cadence_differs",
        (_make_regular(cadence_s=600), 100, NS),
        (_make_regular(cadence_s=900), 100, NS),
    ),
    (
        "test_regular_axis_hash_differs_when_epoch_differs",
        (_make_regular(epoch="2024-01-01T00:00:00Z"), 100, NS),
        (_make_regular(epoch="2025-01-01T00:00:00Z"), 100, NS),
    ),
    (
        "test_regular_axis_hash_differs_when_mode_differs",
        (_make_regular(mode="exact"), 100, NS),
        (_make_regular(mode="floor"), 100, NS),
    ),
    (
        "test_regular_axis_hash_differs_when_dtype_differs",
        (_make_regular(), 100, NS),
        (_make_regular(), 100, "datetime64[s]"),
    ),
)


def test_regular_axis_hash_differs_per_identity_input(subtests) -> None:
    for row_id, left, right in HASH_DIFFERENCES:
        with subtests.test(msg=row_id):
            assert _compute_group_identity_hash(*left) != _compute_group_identity_hash(*right)


def test_regular_axis_hash_accepts_numpy_dtype_and_string_equivalently() -> None:
    axis = _make_regular()
    from_string = _compute_group_identity_hash(axis, 100, "datetime64[ns]")
    from_np_dtype = _compute_group_identity_hash(axis, 100, np.dtype("datetime64[ns]"))
    assert from_string == from_np_dtype


def test_integer_axis_hash_deterministic() -> None:
    axis = IntegerAxis(slot_count=42)
    a = _compute_group_identity_hash(axis, 42, "int64")
    b = _compute_group_identity_hash(axis, 42, "int64")
    assert a == b
    assert len(a) == 64


def test_integer_axis_hash_differs_by_size() -> None:
    axis = IntegerAxis(slot_count=42)
    assert _compute_group_identity_hash(axis, 42, "int64") != (
        _compute_group_identity_hash(axis, 43, "int64")
    )


def test_irregular_axis_hash_deterministic() -> None:
    axis = IrregularTimeAxis(
        coordinate="timestamp",
        values=("2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z"),
    )
    a = _compute_group_identity_hash(axis, 2, "datetime64[ns]")
    b = _compute_group_identity_hash(axis, 2, "datetime64[ns]")
    assert a == b
    assert len(a) == 64


def test_regular_and_integer_axes_produce_different_hashes() -> None:
    reg = _make_regular()
    integer = IntegerAxis(slot_count=100)
    assert _compute_group_identity_hash(reg, 100, "datetime64[ns]") != (
        _compute_group_identity_hash(integer, 100, "datetime64[ns]")
    )


def test_unsupported_axis_type_raises_not_implemented() -> None:
    class _Fake:
        pass

    with pytest.raises(NotImplementedError, match="No group identity hash"):
        _compute_group_identity_hash(_Fake(), 10, "float32")  # type: ignore[arg-type]


def test_regular_axis_end_date_and_slot_count_yield_same_hash_at_same_size() -> None:
    a = _make_regular(slot_count=100)
    b = _make_regular(slot_count=None, end_date="2024-01-01T16:40:00Z")
    assert _compute_group_identity_hash(a, 100, "datetime64[ns]") == (
        _compute_group_identity_hash(b, 100, "datetime64[ns]")
    )


def test_gregorian_axis_hashes_keep_their_pinned_persisted_values() -> None:
    # Persisted-format contract: a shift here invalidates every existing store.
    # Migrate, do not regenerate the literals.
    regular = _make_regular()
    assert _compute_group_identity_hash(regular, 100, "datetime64[ns]") == (
        "04e7861a55f053efca8b8fab188aec5a0cfd63ca130c844407d0d880237e775a"
    )
    numeric = IrregularTimeAxis(coordinate="time", values=(1, 2, 3))
    assert _compute_group_identity_hash(numeric, 3, "int64") == (
        "3efcac228fc838b66355fec2aebae37971fa710f80fff08410c58b9b30b861f9"
    )
    iso = IrregularTimeAxis(
        coordinate="timestamp", values=["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]
    )
    assert _compute_group_identity_hash(iso, 2, "datetime64[ns]") == (
        "6e0646f7147f7bd8985f5f96e93b16a699535e34429ccf4f66fafef37e359ded"
    )


def test_calendar_enters_the_hash_of_a_non_gregorian_axis() -> None:
    def hash_for(**kwargs: object) -> str:
        axis = RegularTimeAxis(
            coordinate="time",
            epoch="2049-01-01T00:00:00Z",
            cadence_s=86400,
            slot_count=90,
            **kwargs,  # type: ignore[arg-type]
        )
        return _compute_group_identity_hash(axis, 90, "int64")

    assert hash_for(calendar="360_day") != hash_for()
    assert hash_for(calendar="360_day") != hash_for(calendar="noleap")
    assert hash_for(calendar="360_day") == (
        "105bacab35c891da511a2b5841869dff9b66c7f3a13fee6ff2bc7a485b2672f1"
    )
    assert hash_for(calendar="standard") == hash_for()
