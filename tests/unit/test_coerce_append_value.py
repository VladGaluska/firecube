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

"""Tests for _coerce_append_value 1970 bug fix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from firecube.ingestor.runtime.zarr.append import _coerce_append_value
from tests.helpers.rows import table_row


def test_coerce_append_value_rows(subtests: Any, tmp_path: Path) -> None:
    with table_row(subtests, "datetime64_passthrough", tmp_path):
        # datetime64 value converts correctly without attrs.
        val = np.datetime64("2023-12-01", "s")
        result = _coerce_append_value(val)
        assert isinstance(result, pd.Timestamp)
        assert str(result).startswith("2023-12-01")

    with table_row(subtests, "iso_string_passthrough", tmp_path):
        # ISO string converts correctly without attrs.
        result = _coerce_append_value("2023-12-01T00:00:00")
        assert isinstance(result, pd.Timestamp)
        assert str(result).startswith("2023-12-01")

    with table_row(subtests, "numeric_without_cf_units_returns_raw", tmp_path):
        result = _coerce_append_value(1.0)
        assert result == 1.0

        result_int = _coerce_append_value(42)
        assert result_int == 42

    with table_row(subtests, "numeric_with_cf_units_decodes_correctly", tmp_path):
        # Numeric value WITH CF units decodes to correct timestamp.
        attrs = {"units": "days since 2000-01-01", "calendar": "standard"}
        result = _coerce_append_value(0.0, attrs)
        assert isinstance(result, pd.Timestamp)
        assert str(result).startswith("2000-01-01")
        assert "1970" not in str(result)
