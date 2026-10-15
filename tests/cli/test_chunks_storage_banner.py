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

from click.testing import CliRunner

from firecube.cli.main import cli
from tests.helpers.rows import table_row

_CONFIG_TEMPLATE = """\
[storage]
type = "local"
target_path = "{target_path}"
"""


def _write_config(tmp_path: Path, *, target_path: Path) -> Path:
    config_file = tmp_path / "firecube-test.toml"
    config_file.write_text(
        _CONFIG_TEMPLATE.format(target_path=target_path),
        encoding="utf-8",
    )
    return config_file


def _stub_list_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        "firecube.core.controlplane.manager.ChunkManager.list_chunks",
        lambda self, **kwargs: [],
    )


_ROWS = [
    (
        "test_chunks_banner_shown_with_config_source",
        "config",
        [],
        True,
        ["[chunks] storage: type=local", "(from config)"],
        [],
    ),
    (
        "test_chunks_banner_suppressed_with_quiet",
        "config",
        ["--quiet"],
        True,
        [],
        ["[chunks] storage:"],
    ),
    (
        "test_chunks_banner_shown_with_env_source",
        "env",
        [],
        True,
        ["[chunks] storage: type=s3", "(from env)"],
        [],
    ),
    (
        "test_chunks_no_banner_when_neither_config_nor_env",
        "none",
        [],
        False,
        ["Chunk operations require a full product URI or a [storage] configuration"],
        ["[chunks] storage:"],
    ),
]


def test_chunks_storage_banner(subtests: Any, monkeypatch, tmp_path: Path) -> None:
    for row_id, source, extra_args, exit_ok, present, absent in _ROWS:
        with table_row(subtests, row_id, tmp_path, monkeypatch) as (workdir, mp):
            assert mp is not None
            config_file = _write_config(workdir, target_path=workdir / "product.zarr")
            mp.delenv("FIRECUBE_STORAGE_TYPE", raising=False)
            mp.delenv("FIRECUBE_BUCKET", raising=False)
            mp.setenv(
                "FIRECUBE_CONFIG",
                str(config_file if source == "config" else workdir / "missing.toml"),
            )
            if source == "env":
                mp.setenv("FIRECUBE_STORAGE_TYPE", "s3")
                mp.setenv("FIRECUBE_BUCKET", "test")
            _stub_list_chunks(mp)

            result = CliRunner().invoke(
                cli,
                [
                    "chunks",
                    *extra_args,
                    "--workspace",
                    str(workdir / "ws"),
                    "list",
                    "--format",
                    "json",
                ],
            )

            assert (result.exit_code == 0) is exit_ok, result.stderr
            for text in present:
                assert text in result.stderr
            for text in absent:
                assert text not in result.stderr
