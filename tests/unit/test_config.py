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

from firecube.core.config import build_storage_config
from tests.helpers.rows import table_row


def test_storage_type_precedence(subtests: Any, tmp_path: Path) -> None:
    cfg = {"storage": {"type": "local"}}
    # (row id, environment, CLI overrides, expected storage type)
    rows: tuple[tuple[str, dict[str, str], dict[str, Any], str], ...] = (
        ("test_storage_type_precedence_env_over_config", {"FIRECUBE_STORAGE_TYPE": "s3"}, {}, "s3"),
        (
            "test_storage_type_precedence_cli_over_env_over_config",
            {"FIRECUBE_STORAGE_TYPE": "s3"},
            {"storage_type": "local"},
            "local",
        ),
        ("test_storage_type_falls_back_to_config_when_no_env_or_cli", {}, {}, "local"),
    )
    for row_id, env, overrides, expected in rows:
        with table_row(subtests, row_id, tmp_path):
            assert build_storage_config(cfg, env, overrides).storage_type == expected
