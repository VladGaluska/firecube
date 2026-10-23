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

from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from click.testing import CliRunner

from firecube.cli._typed_options import display_format_option, output_path_option
from tests.helpers.rows import table_row

# (row id, option decorator, CLI args, exit code 0?, substrings of the output).
# The command echoes every parameter as ``name=value``.
_ROWS: tuple[
    tuple[str, Callable[[Callable[..., Any]], Any], list[str], bool, tuple[str, ...]], ...
] = (
    (
        "test_display_format_option_default_table",
        display_format_option(default="table"),
        [],
        True,
        ("output_format=table",),
    ),
    (
        "test_display_format_option_json_default",
        display_format_option(default="json"),
        [],
        True,
        ("output_format=json",),
    ),
    (
        "test_display_format_option_help_shows_choices",
        display_format_option(),
        ["--help"],
        True,
        ("--format", "table", "json", "csv"),
    ),
    (
        "test_display_format_option_explicit_choice",
        display_format_option(),
        ["-f", "csv"],
        True,
        ("output_format=csv",),
    ),
    (
        "test_display_format_option_rejects_invalid_choice",
        display_format_option(),
        ["-f", "yaml"],
        False,
        ("Invalid value for '-f' / '--format'", "'yaml' is not one of 'table', 'json', 'csv'"),
    ),
    (
        "test_output_path_option_required",
        output_path_option(required=True),
        [],
        False,
        ("Missing option '-o' / '--output'",),
    ),
    (
        "test_output_path_option_accepts_value",
        output_path_option(required=True),
        ["-o", "/tmp/foo.bin"],
        True,
        ("output_path=/tmp/foo.bin",),
    ),
    (
        "test_output_path_option_optional_when_not_required",
        output_path_option(required=False),
        [],
        True,
        ("output_path=None",),
    ),
    (
        "test_output_path_option_help_marker",
        output_path_option(required=True),
        ["--help"],
        True,
        ("--output",),
    ),
    (
        "test_output_path_option_custom_help_text",
        output_path_option(required=False, help_text="custom destination help"),
        ["--help"],
        True,
        ("custom destination help",),
    ),
)


def test_output_option_contract(subtests: Any, tmp_path: Path) -> None:
    for row_id, option, args, succeeds, expected in _ROWS:
        with table_row(subtests, row_id, tmp_path):

            @click.command()
            @option
            def cmd(**params: object) -> None:
                for name, value in params.items():
                    click.echo(f"{name}={value}")

            result = CliRunner().invoke(cmd, args)

            assert (result.exit_code == 0) is succeeds, result.output
            for part in expected:
                assert part in result.output
