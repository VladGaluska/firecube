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

"""Deterministic payload-lifetime guard for lazy ``WriteIntent`` payloads.

A callable payload must be resolved at dispatch and released once written:
the write path may not keep resolved payloads across intents or after the
call returns. Lifetime is observed with ``weakref`` and ``gc``; there is no
RSS or timing involved.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import zarr

from firecube.ingestor.runtime.zarr.strategies.indexed_region import IndexedRegionStrategy
from firecube.ingestor.templates.direct_zarr import WriteIntent

_GROUP = "G"
_SLOTS = 6


def test_resolved_payloads_are_released_after_each_write(tmp_path: Path) -> None:
    """Every payload is dead before the next one resolves, and after the call.

    The intents stay alive in the caller throughout, so only the write path
    can be retaining a payload if one survives.
    """
    target = tmp_path / "product.zarr"
    grp = zarr.open_group(str(target), mode="a", zarr_format=3).require_group(_GROUP)
    grp.create_array(
        name="data", shape=(_SLOTS, 4, 4), dtype="float32", fill_value=-1.0, chunks=(1, 4, 4)
    )

    refs: list[weakref.ref] = []
    alive_when_resolving: list[tuple[int, int]] = []

    def _thunk(slot: int):
        def _resolve() -> np.ndarray:
            gc.collect()
            alive_when_resolving.append((slot, sum(r() is not None for r in refs)))
            payload = np.full((4, 4), float(slot + 1), dtype=np.float32)
            refs.append(weakref.ref(payload))
            return payload

        return _resolve

    intents = [
        WriteIntent(
            group=_GROUP,
            array="data",
            ts_index=slot,
            data=_thunk(slot),
            kind="region",
            y_slice=slice(0, 4),
        )
        for slot in range(_SLOTS)
    ]
    assert alive_when_resolving == [], "payloads must stay lazy until dispatch"

    IndexedRegionStrategy(store_uri=str(target)).write_groups(group_to_intents={_GROUP: intents})
    gc.collect()

    written = zarr.open_array(str(target / _GROUP / "data"), mode="r")
    for slot in range(_SLOTS):
        np.testing.assert_array_equal(written[slot], np.full((4, 4), slot + 1, dtype=np.float32))
    assert alive_when_resolving == [(slot, 0) for slot in range(_SLOTS)], (
        "a previous payload was still referenced when the next one resolved"
    )
    assert [r() for r in refs] == [None] * _SLOTS, "a resolved payload outlived the write call"
