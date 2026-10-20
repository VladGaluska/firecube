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

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import click
import pytest

from firecube.cli._errors import render
from firecube.core.errors import ConfigurationError, FirecubeError, Message, Param
from firecube.ingestor.contracts.interfaces import PipelineHost
from firecube.ingestor.runtime import engine
from firecube.ingestor.runtime.engine import PipelineExecutor, PipelineFailedBatchesError
from firecube.ingestor.types.context import (
    IngestContext,
    OutputPaths,
    PipelineBatch,
    PipelineResult,
    PipelineRunState,
    PluginContext,
    RuntimeIngestContext,
)


def _runtime_ctx(tmp_path: Path) -> RuntimeIngestContext:
    return RuntimeIngestContext.from_ingest_context(
        IngestContext(source=str(tmp_path), target="fallback.zarr", output_format="zarr"),
        run_id="run-001",
        temp_root=tmp_path,
        materializer=lambda p: Path(p),
    )


def _batch(batch_id: str) -> PipelineBatch:
    return PipelineBatch(
        batch_id=batch_id,
        data_path=Path("."),
        items=[],
        size_bytes=0,
        files_count=0,
    )


def _state(*results: PipelineResult) -> PipelineRunState:
    return PipelineRunState(
        product="dummy",
        pipeline_workers=1,
        batch_size=1,
        batches=tuple(result.batch for result in results),
        ingestion_start_time=0.0,
        batch_creation_duration=0.0,
        processing_start_time=0.0,
        results=results,
    )


def _host() -> Any:
    host = MagicMock(spec=PipelineHost)
    host._aggregate_metrics.return_value = {}
    host.name = "dummy"
    host._chunk_manager = None
    return host


def test_finalize_raises_when_any_batch_failed(tmp_path: Path) -> None:
    success_batch = _batch("b1")
    failed_batch = _batch("b2")
    state = _state(
        PipelineResult(
            batch=success_batch, outputs=OutputPaths(primary="s3://bucket/product"), success=True
        ),
        PipelineResult(
            batch=failed_batch, outputs=OutputPaths(primary=Path("")), success=False, error="boom"
        ),
    )

    with pytest.raises(PipelineFailedBatchesError, match="boom"):
        PipelineExecutor().finalize(_runtime_ctx(tmp_path), state, _host())


def test_finalize_passes_when_all_batches_succeed(tmp_path: Path) -> None:
    first_batch = _batch("b1")
    second_batch = _batch("b2")
    state = _state(
        PipelineResult(
            batch=first_batch, outputs=OutputPaths(primary="s3://bucket/product"), success=True
        ),
        PipelineResult(
            batch=second_batch, outputs=OutputPaths(primary="local-product.zarr"), success=True
        ),
    )

    result = PipelineExecutor().finalize(_runtime_ctx(tmp_path), state, _host())

    assert result.output_path == "s3://bucket/product"


@pytest.mark.contract
@pytest.mark.parametrize("exc", [ConfigurationError("bad"), ValueError("bad")])
def test_batch_result_keeps_firecube_exception_only(tmp_path: Path, exc: Exception) -> None:
    host = _host()
    host._process_batch.side_effect = exc
    result = engine._process_batch_timed(host, _batch("b1"), PluginContext(_runtime_ctx(tmp_path)))
    assert not result.success and result.error == str(exc)
    assert result.failure is (exc if isinstance(exc, FirecubeError) else None)


@pytest.mark.contract
def test_failed_batches_error_text_renders_flags_only_under_click() -> None:
    cause = ConfigurationError(Message("bad {p}", p=Param("time_dim")))
    failed = [PipelineResult(batch=_batch("b1"), success=False, error=str(cause))]
    failed[0].failure = cause
    err = PipelineFailedBatchesError(run_id="r1", failed=failed, not_attempted=2)
    assert isinstance(err, FirecubeError) and isinstance(err, RuntimeError)
    assert (err.run_id, list(err.failed), err.not_attempted) == ("r1", failed, 2)
    assert "--" not in str(err) and "bad time_dim" in str(err)
    assert render(
        err, click.Context(click.Command("ingest", params=[click.Option(["--time-dim"])]))
    ) == (
        "Pipeline run 'r1' had 1 failed batch(es): bad --time-dim. 2 later batch(es) were not "
        "attempted: this host stops at the first failed batch so no batch appends past a gap. "
        "Run recorded as status=failed. Recovery: resolve the batch error above before retrying. "
        "For compatible Zarr inputs, --option resume_existing=true skips present timestamps "
        "and refills failed slots; --option force_reingest=true overwrites existing timestamps. "
        "Neither option can insert absent timestamps before the stored maximum or bypass schema validation."
    )
