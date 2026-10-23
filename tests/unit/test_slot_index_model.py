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

"""Unit tests for the SlotIndexModel / SlotAxis foundation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from firecube.core.errors import (
    FirecubeError,
    SlotIndexModelClaimTimeoutError,
    SlotIndexModelConflictError,
    SlotIndexModelError,
    SlotIndexUnmanagedStoreError,
)
from firecube.core.slot_index import (
    SLOT_INDEX_MODEL_ATTR,
    SLOT_INDEX_MODEL_IDENTITY_HASH_ATTR,
    SlotAxis,
    SlotIndexModel,
    epoch_s_to_iso,
    iso_to_epoch_s,
    normalize_epoch_iso,
)

GOLDEN_EPOCH_ISO = "2026-01-01T00:00:00Z"
GOLDEN_EPOCH_SECONDS = 1767225600
GOLDEN_CANONICAL_BYTES = (
    b'{"epoch":"2026-01-01T00:00:00Z","groups":{"g1":{"cadence_s":300,'
    b'"mode":"exact"}},"name":"opera_v1","schema_version":"v1","time_unit":null}'
)


def _golden_model() -> SlotIndexModel:
    return SlotIndexModel(
        name="opera_v1",
        epoch=GOLDEN_EPOCH_ISO,
        groups={"g1": SlotAxis(cadence_s=300, mode="exact")},
    )


_G = {"g": SlotAxis(300, "exact")}
_REJECTION_ROWS: tuple[tuple[str, Callable[[], object], type[Exception], str | None], ...] = (
    (
        "slot_axis_rejects_zero_cadence",
        lambda: SlotAxis(cadence_s=0, mode="exact"),
        ValueError,
        "cadence_s",
    ),
    ("slot_axis_rejects_negative_cadence", lambda: SlotAxis(-1, "exact"), ValueError, "cadence_s"),
    (
        "slot_axis_rejects_invalid_mode",
        lambda: SlotAxis(300, cast(Any, "invalid")),
        ValueError,
        "mode",
    ),
    ("slot_axis_rejects_bool_cadence", lambda: SlotAxis(True, "exact"), TypeError, None),
    ("slot_axis_rejects_bool_cadence_false", lambda: SlotAxis(False, "exact"), TypeError, None),
    ("slot_axis_rejects_float_cadence", lambda: SlotAxis(cast(int, 1.5), "exact"), TypeError, None),
    (
        "slot_axis_rejects_string_cadence",
        lambda: SlotAxis(cast(Any, "1"), "exact"),
        TypeError,
        None,
    ),
    (
        "slot_index_model_rejects_empty_name",
        lambda: SlotIndexModel("", GOLDEN_EPOCH_ISO, dict(_G)),
        ValueError,
        "name",
    ),
    (
        "slot_index_model_rejects_empty_epoch",
        lambda: SlotIndexModel("x", "", dict(_G)),
        ValueError,
        "epoch",
    ),
    (
        "slot_index_model_rejects_empty_groups",
        lambda: SlotIndexModel("x", GOLDEN_EPOCH_ISO, {}),
        ValueError,
        "groups",
    ),
    (
        "slot_index_model_rejects_empty_group_key",
        lambda: SlotIndexModel("x", GOLDEN_EPOCH_ISO, {"": SlotAxis(300, "exact")}),
        ValueError,
        "group keys",
    ),
    (
        "normalize_epoch_iso_rejects_non_utc_offset",
        lambda: normalize_epoch_iso("2026-01-01T00:00:00+05:00"),
        ValueError,
        "UTC",
    ),
    (
        "iso_to_epoch_s_rejects_non_utc_offset",
        lambda: iso_to_epoch_s("2026-01-01T00:00:00-07:00"),
        ValueError,
        "UTC",
    ),
)


def test_invalid_inputs_are_rejected(subtests) -> None:
    for row_id, build, exc, match in _REJECTION_ROWS:
        with subtests.test(msg=f"test_{row_id}"), pytest.raises(exc, match=match):
            build()


_CONVERSION_ROWS = (
    ("iso_to_epoch_s_golden", lambda: iso_to_epoch_s(GOLDEN_EPOCH_ISO), GOLDEN_EPOCH_SECONDS),
    ("epoch_s_to_iso_golden", lambda: epoch_s_to_iso(GOLDEN_EPOCH_SECONDS), GOLDEN_EPOCH_ISO),
    (
        "normalize_epoch_iso_accepts_explicit_utc_offset",
        lambda: normalize_epoch_iso("2026-01-01T00:00:00+00:00"),
        GOLDEN_EPOCH_ISO,
    ),
    (
        "normalize_epoch_iso_round_trip_z",
        lambda: normalize_epoch_iso(GOLDEN_EPOCH_ISO),
        GOLDEN_EPOCH_ISO,
    ),
)


def test_epoch_conversions(subtests) -> None:
    for row_id, convert, expected in _CONVERSION_ROWS:
        with subtests.test(msg=f"test_{row_id}"):
            assert convert() == expected


def test_slot_axis_accepts_numpy_int64_cadence() -> None:
    axis = SlotAxis(cadence_s=cast(int, np.int64(300)), mode="exact")
    assert axis.cadence_s == 300
    assert axis.mode == "exact"


def test_canonical_bytes_golden() -> None:
    assert _golden_model().canonical_bytes() == GOLDEN_CANONICAL_BYTES


def test_canonical_bytes_group_order_independent() -> None:
    a = SlotIndexModel(
        name="x",
        epoch=GOLDEN_EPOCH_ISO,
        groups={"b": SlotAxis(600, "floor"), "a": SlotAxis(300, "exact")},
    )
    b = SlotIndexModel(
        name="x",
        epoch=GOLDEN_EPOCH_ISO,
        groups={"a": SlotAxis(300, "exact"), "b": SlotAxis(600, "floor")},
    )
    assert a.canonical_bytes() == b.canonical_bytes()
    assert a.identity_hash == b.identity_hash


def test_identity_hash_is_sha256_hex_length() -> None:
    assert len(_golden_model().identity_hash) == 64


def test_identity_hash_not_inside_canonical_bytes() -> None:
    model = _golden_model()
    assert model.identity_hash.encode("ascii") not in model.canonical_bytes()


def test_time_unit_none_serialised_as_json_null() -> None:
    model = _golden_model()
    assert b'"time_unit":null' in model.canonical_bytes()


def test_z_and_plus_zero_epoch_produce_distinct_hashes() -> None:
    z_model = SlotIndexModel(
        name="x", epoch="2026-01-01T00:00:00Z", groups={"g": SlotAxis(300, "exact")}
    )
    offset_model = SlotIndexModel(
        name="x",
        epoch="2026-01-01T00:00:00+00:00",
        groups={"g": SlotAxis(300, "exact")},
    )
    assert z_model.identity_hash != offset_model.identity_hash


def test_root_attr_constants_are_stable() -> None:
    assert SLOT_INDEX_MODEL_ATTR == "firecube_slot_index_model"
    assert SLOT_INDEX_MODEL_IDENTITY_HASH_ATTR == "firecube_slot_index_model_identity_hash"


def test_exception_hierarchy() -> None:
    assert issubclass(SlotIndexModelError, FirecubeError)
    assert issubclass(SlotIndexModelConflictError, SlotIndexModelError)
    assert issubclass(SlotIndexUnmanagedStoreError, SlotIndexModelError)
    assert issubclass(SlotIndexModelClaimTimeoutError, SlotIndexModelError)
