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

import itertools
from typing import Any

from firecube.core.controlplane import ChunkManager, SpanCoverage

_SPAN_SEQUENCE = itertools.count()


def record_run_with_span(
    manager: ChunkManager,
    *,
    product: str,
    plugin: str,
    run_id: str,
    group: str = "data",
    ranges: list[Any],
    terminal: str | None,
    span_meta: dict[str, Any] | None = None,
) -> None:
    """
    `terminal` is the run's final status (`"complete"`, `"failed"`), or
    `None` for a run that never reached the final. An interrupted run's buffered
    span events reach the WAL when the process closes its writers, which
    `None` stands for and a run killed doesn't have them. `span_meta` is
    merged over the recorder's own keys, so a test can add or override
    `written_to`.
    """
    output_path = f"{manager.base_uri.rstrip('/')}/{product}"
    # list_chunks collapses the active spans of one group with equal bounds to
    # the latest run's, so each recorded span gets bounds of its own.
    sequence = next(_SPAN_SEQUENCE)
    time_min = f"2026-01-01T00:{sequence:02d}:00Z"
    time_max = f"2026-01-01T01:{sequence:02d}:00Z"
    manager.record_run_started(
        product=product,
        run_id=run_id,
        output_path=output_path,
        output_format="zarr",
        size=0,
        meta={"plugin": plugin},
    )
    manager.record_span(
        product=product,
        run_id=run_id,
        batch_id="batch-001",
        group=group,
        status="active",
        coverage=SpanCoverage(
            group=group,
            arrays=[f"{group}/values"],
            time_index_ranges=ranges,
            time_min=time_min,
            time_max=time_max,
        ),
        meta={
            "plugin": plugin,
            "group": group,
            "time_min": time_min,
            "time_max": time_max,
            **(span_meta or {}),
        },
    )
    if terminal is None:
        manager.close()
    else:
        manager.record_run_terminal(
            product=product,
            run_id=run_id,
            output_path=output_path,
            output_format="zarr",
            size=1,
            meta={"plugin": plugin},
            status=terminal,
        )
