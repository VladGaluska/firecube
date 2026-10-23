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

"""Per-batch schema claim is skipped in slot-range (parallel) mode.

In slot-range mode the global schema is created/verified once per pod at startup
(``_verify_schema_at_pod_startup``). Re-running schema setup per batch under the
exclusive ``zarr_region:<group>:schema`` claim would make concurrent pods on the
same group race on idempotent work. So ``write_groups`` skips that claim when a
``slot_range`` is set, while single-pod runs keep the §24 claim-then-ensure path.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest

from firecube.ingestor.runtime.zarr.strategies.indexed_region import IndexedRegionStrategy
from firecube.ingestor.templates.direct_zarr import WriteIntent, ZarrArraySpec, ZarrGroupSpec
from tests.helpers.rows import table_row

pytestmark = pytest.mark.unit


class _Writer:
    def ensure_group(self, group: str, **kwargs) -> None:
        _ = (group, kwargs)

    def set_group_attrs(self, group: str, attrs) -> None:
        _ = (group, attrs)


def _patch_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "firecube.ingestor.runtime.zarr.strategies.indexed_region.RegionZarrWriter",
        lambda *args, **kwargs: _Writer(),
    )
    monkeypatch.setattr(
        IndexedRegionStrategy,
        "_dispatch_intent",
        staticmethod(lambda writer, intent: None),
    )


def _schema() -> list[ZarrGroupSpec]:
    return [
        ZarrGroupSpec(
            group="data",
            arrays=[ZarrArraySpec(name="values", shape=(1,), dtype=object)],
        )
    ]


def _intent() -> WriteIntent:
    return WriteIntent(group="data", array="values", ts_index=0, data=None, y_slice=slice(0, 1))


def _run(monkeypatch, *, slot_range) -> list[str]:
    _patch_io(monkeypatch)
    group_claims: list[str] = []
    IndexedRegionStrategy(store_uri="/tmp/x.zarr", schema=_schema()).write_groups(
        group_to_intents={"data": [_intent()]},
        schema=_schema(),
        claim_for_group=lambda g: group_claims.append(g) or contextlib.nullcontext(),
        claim_for_slot=lambda g, ts: contextlib.nullcontext(),
        slot_range=slot_range,
    )
    return group_claims


def test_indexed_region_schema_claim_modes(
    subtests: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with table_row(
        subtests, "slot_range_mode_skips_per_batch_schema_claim", tmp_path, monkeypatch
    ) as (_, mp):
        assert mp is not None
        assert _run(mp, slot_range=(0, 100)) == []

    with table_row(subtests, "single_pod_still_takes_schema_claim", tmp_path, monkeypatch) as (
        _,
        mp,
    ):
        assert mp is not None
        assert _run(mp, slot_range=None) == ["data"]
