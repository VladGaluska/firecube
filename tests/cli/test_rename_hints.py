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

"""Tests for CLI rename-hint layer."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from firecube.cli.main import cli


def test_renamed_flags_show_replacement_hint(subtests) -> None:
    rows = {
        "ingest_source_shows_input_data_hint": (
            ["ingest", "cli_test_plugin", "--source", "/tmp/data"],
            "--source",
            "--input-data (-i)",
            "strict-URI refactor",
        ),
        "archive_create_target_shows_archive_hint": (
            ["archive", "create", "--target", "/tmp/x.tgm", "--source", "file:///tmp/p.zarr"],
            "--target",
            "--archive (-a)",
            "tgm artifact output",
        ),
        "archive_restore_source_shows_archive_hint": (
            ["archive", "restore", "--source", "/tmp/x.tgm", "--target", "file:///tmp/p.zarr"],
            "--source",
            "--archive (-a)",
            "tgm artifact path",
        ),
    }
    for row_id, (args, flag, replacement, reason) in rows.items():
        with subtests.test(msg=row_id):
            result = CliRunner().invoke(cli, args)
            assert result.exit_code == 2
            for expected in ("No such option", flag, "Hint:", replacement, reason):
                assert expected in result.output


def test_unknown_flag_without_registry_entry_has_no_hint() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "cli_test_plugin", "--definitely-not-a-flag", "x"])
    assert result.exit_code == 2
    assert "Hint:" not in result.output


def test_chunks_product_still_uses_click_native_suggestion() -> None:
    """--product -> --product-name is auto-suggested by Click; we don't override."""
    runner = CliRunner()
    result = runner.invoke(cli, ["chunks", "list", "--product", "x"])
    assert result.exit_code == 2
    output = result.output
    assert "No such option" in output
    assert "--product" in output
    assert "Did you mean" in output
    assert "--product-name" in output
    assert "Hint:" not in output


def test_valid_flags_on_archive_commands_get_no_hint(subtests) -> None:
    """--target is valid on archive restore; --source is valid on archive create."""
    rows = {
        "old_target_on_archive_restore_no_hint": (
            ["restore", "--archive", "file:///tmp/x.tgm", "--target", "file:///tmp/p.zarr"],
            "[dry-run] Would restore:",
        ),
        "old_source_on_archive_create_no_hint": (
            ["create", "--source", "file:///tmp/p.zarr", "--archive", "file:///tmp/x.tgm"],
            "[dry-run] Would archive:",
        ),
    }
    for row_id, (args, message) in rows.items():
        with subtests.test(msg=row_id):
            result = CliRunner().invoke(cli, ["archive", *args, "--dry-run"])
            assert result.exit_code == 0, result.output
            assert message in result.output
            assert "Hint:" not in result.output


def test_hint_fires_under_real_prog_name() -> None:
    """Regression: hint must fire when invoked as the ``firecube`` binary, not only CliRunner.

    Catches the ``build_command_path`` bug where filtering by hardcoded ``"cli"``
    info_name silently breaks production (where info_name is ``"firecube"``)
    while tests pass.
    """
    with pytest.raises(click.NoSuchOption) as excinfo:
        cli(
            args=["ingest", "cli_test_plugin", "--source", "/tmp"],
            prog_name="firecube",
            standalone_mode=False,
        )
    assert "Hint:" in excinfo.value.message
    assert "--input-data (-i)" in excinfo.value.message
