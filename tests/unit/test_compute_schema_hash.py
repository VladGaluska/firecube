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

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from firecube.ingestor.templates.direct_zarr import (
    ZarrArraySpec,
    ZarrGroupSpec,
    _compute_schema_hash,
)
from tests.helpers.rows import table_row

pytestmark = pytest.mark.unit


def _schema(
    *,
    dtype: Any = np.float32,
    chunks: tuple[int, ...] | None = (1, 4, 5),
    fill_value: Any = 0.0,
) -> list[ZarrGroupSpec]:
    return [
        ZarrGroupSpec(
            group="data",
            arrays=[
                ZarrArraySpec(
                    name="values",
                    shape=(1, 4, 5),
                    dtype=dtype,
                    chunks=chunks,
                    fill_value=fill_value,
                )
            ],
        )
    ]


def test_stable_across_calls() -> None:
    schema = _schema()

    assert _compute_schema_hash(schema, {"data": 10}) == _compute_schema_hash(
        schema,
        {"data": 10},
    )


def test_sensitive_to_dtype() -> None:
    assert _compute_schema_hash(_schema(dtype=np.float32), {"data": 10}) != _compute_schema_hash(
        _schema(dtype=np.float64),
        {"data": 10},
    )


def test_sensitive_to_chunks() -> None:
    assert _compute_schema_hash(_schema(chunks=(1, 4, 5)), {"data": 10}) != _compute_schema_hash(
        _schema(chunks=(2, 4, 5)),
        {"data": 10},
    )


def test_attrs_affect_hash() -> None:
    base = ZarrGroupSpec(
        group="data",
        arrays=[ZarrArraySpec(name="values", shape=(1, 4, 5), dtype=np.float32, chunks=(1, 4, 5))],
    )
    with_attrs = ZarrGroupSpec(
        group="data",
        arrays=[
            ZarrArraySpec(
                name="values",
                shape=(1, 4, 5),
                dtype=np.float32,
                chunks=(1, 4, 5),
                attrs={"ignored": True},
            )
        ],
    )
    object.__setattr__(with_attrs.arrays[0], "codecs", ["ignored"])

    assert _compute_schema_hash([base], {"data": 10}) != _compute_schema_hash(
        [with_attrs],
        {"data": 10},
    )


def test_codecs_do_not_affect_hash() -> None:
    base = ZarrGroupSpec(
        group="data",
        arrays=[ZarrArraySpec(name="values", shape=(1, 4, 5), dtype=np.float32, chunks=(1, 4, 5))],
    )
    with_codecs = ZarrGroupSpec(
        group="data",
        arrays=[ZarrArraySpec(name="values", shape=(1, 4, 5), dtype=np.float32, chunks=(1, 4, 5))],
    )
    object.__setattr__(with_codecs.arrays[0], "codecs", ["zstd"])

    assert _compute_schema_hash([base], {"data": 10}) == _compute_schema_hash(
        [with_codecs],
        {"data": 10},
    )


def test_normalizes_nan_fill_value() -> None:
    assert _compute_schema_hash(
        _schema(fill_value=float("nan")), {"data": 10}
    ) == _compute_schema_hash(
        _schema(fill_value=np.float32("nan")),
        {"data": 10},
    )


def test_declared_field_changes_hash(subtests, tmp_path: Path) -> None:
    rows = {
        "shards_changes_hash": {"shards": (5, 4)},
        "attrs_changes_hash": {"attrs": {"units": "K"}},
        "dimension_names_changes_hash": {"dimension_names": ("time", "y")},
        "time_indexed_changes_hash": {"time_indexed": False},
    }
    for row_id, fields in rows.items():
        with table_row(subtests, row_id, tmp_path):
            spec_a = ZarrGroupSpec(
                group="g", arrays=[ZarrArraySpec(name="d", shape=(10, 4), dtype="float32")]
            )
            spec_b = ZarrGroupSpec(
                group="g",
                arrays=[ZarrArraySpec(name="d", shape=(10, 4), dtype="float32", **fields)],
            )
            assert _compute_schema_hash([spec_a], {"g": 10}) != _compute_schema_hash(
                [spec_b], {"g": 10}
            )


def test_attrs_order_insensitive() -> None:
    """attrs dict key order does not change hash."""
    spec_a = ZarrGroupSpec(
        group="g",
        arrays=[
            ZarrArraySpec(
                name="d",
                shape=(10, 4),
                dtype="float32",
                attrs={"units": "K", "calendar": "standard"},
            )
        ],
    )
    spec_b = ZarrGroupSpec(
        group="g",
        arrays=[
            ZarrArraySpec(
                name="d",
                shape=(10, 4),
                dtype="float32",
                attrs={"calendar": "standard", "units": "K"},
            )
        ],
    )
    assert _compute_schema_hash([spec_a], {"g": 10}) == _compute_schema_hash([spec_b], {"g": 10})
