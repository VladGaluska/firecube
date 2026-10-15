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

import pytest

from firecube.ingestor.errors import ConfigurationError
from firecube.ingestor.runtime.base import BaseIngestor
from firecube.ingestor.types.context import (
    OutputPaths,
    PipelineBatch,
    PipelineResult,
    PipelineRunState,
    PluginContext,
    RuntimeIngestContext,
)
from tests.helpers.rows import table_row


class _WrongPluginToRuntimeIngestor(BaseIngestor):
    """Fixture: intentionally wrong — plugin-facing hook typed as RuntimeIngestContext."""

    PRODUCT_NAME = "boundary_plugin_to_runtime_test"
    name = "boundary_plugin_to_runtime_test"

    def discover_source_files(self, ctx: RuntimeIngestContext):  # pyright: ignore[reportIncompatibleMethodOverride]
        _ = ctx
        return []

    def _process_batch(self, batch: PipelineBatch, ctx: PluginContext) -> PipelineResult:
        return PipelineResult(batch=batch, outputs=OutputPaths(primary=Path("")), success=True)

    def _aggregate_metrics(self, ctx: RuntimeIngestContext, state: PipelineRunState):
        _ = (ctx, state)
        return {}


class _WrongRuntimeToPluginIngestor(BaseIngestor):
    """Fixture: intentionally wrong — runtime-finalization hook typed as PluginContext."""

    PRODUCT_NAME = "boundary_runtime_to_plugin_test"
    name = "boundary_runtime_to_plugin_test"

    def _process_batch(self, batch: PipelineBatch, ctx: PluginContext) -> PipelineResult:
        return PipelineResult(batch=batch, outputs=OutputPaths(primary=Path("")), success=True)

    def _aggregate_metrics(self, ctx: PluginContext, state: PipelineRunState):  # pyright: ignore[reportIncompatibleMethodOverride]
        _ = (ctx, state)
        return {}


@pytest.mark.unit
def test_context_hook_annotation_validation(subtests: Any, tmp_path: Path) -> None:
    with table_row(subtests, "plugin_hook_annotated_runtime_context_is_rejected", tmp_path):
        ingestor = _WrongPluginToRuntimeIngestor(name="boundary_plugin_to_runtime_test")  # pyright: ignore[reportAbstractUsage]
        with pytest.raises(
            ConfigurationError,
            match=r"must use PluginContext, not RuntimeIngestContext",
        ):
            ingestor._validate_context_hook_signatures()

    with table_row(subtests, "runtime_hook_annotated_plugin_context_is_rejected", tmp_path):
        ingestor = _WrongRuntimeToPluginIngestor(name="boundary_runtime_to_plugin_test")  # pyright: ignore[reportAbstractUsage]
        with pytest.raises(
            ConfigurationError,
            match=r"must use RuntimeIngestContext, not PluginContext",
        ):
            ingestor._validate_context_hook_signatures()
