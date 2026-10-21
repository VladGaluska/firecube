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


def test_cli_completion_scripts(subtests: Any, tmp_path: Path) -> None:
    with table_row(subtests, "completion_bash_script_to_stdout", tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "bash"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "_FIRECUBE_COMPLETE" in result.output
        assert "complete -o" in result.output
        assert "firecube" in result.output

    with table_row(subtests, "completion_zsh_script_to_file", tmp_path):
        runner = CliRunner()
        target = tmp_path / "completion" / "_firecube"
        result = runner.invoke(
            cli,
            ["completion", "zsh", "--output", str(target)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert target.exists()
        content = target.read_text()
        assert "_FIRECUBE_COMPLETE" in content
