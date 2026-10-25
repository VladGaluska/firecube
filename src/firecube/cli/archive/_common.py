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

"""Helpers shared by the ``firecube archive`` commands."""

# pyright: reportMissingImports=false, reportCallIssue=false

from __future__ import annotations

import sys
from pathlib import Path

import click

from firecube.cli._uri_policy import parse_and_validate_uri


def _require_overwrite_confirmation(*, overwrite: bool, yes_i_really_mean_it: bool) -> None:
    """Enforce --yes-i-really-mean-it (or interactive confirm) for --overwrite."""
    if not overwrite or yes_i_really_mean_it:
        return
    if not sys.stdin.isatty():
        raise click.UsageError(
            "OverwriteWithoutConfirmation: --overwrite requires --yes-i-really-mean-it in non-TTY."
        )
    click.confirm("Overwrite existing data?", abort=True)


def _is_zip_archive(archive: str) -> bool:
    """Return True when the archive URI names a zip archive rather than a .tgm."""
    return archive.rstrip("/").lower().endswith(".zip")


def _local_archive(archive: str) -> str:
    """Admit an ``--archive`` value and return the literal path of the local file it names.

    Raises:
        click.BadParameter: The value fails the URI rule.
        click.ClickException: The archive is remote, or no file exists at its path.
    """
    archive_uri = parse_and_validate_uri(archive, param_hint="--archive")
    if archive_uri.is_remote():
        suffix = ".zip" if _is_zip_archive(archive) else ".tgm"
        raise click.ClickException(f"Remote {suffix} artifacts not yet supported")
    if not Path(archive_uri.path).is_file():
        raise click.ClickException(f"Archive not found: {archive}")
    return archive_uri.path
