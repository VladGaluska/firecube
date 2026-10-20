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
"""Run the ``firecube`` command that a refusal message tells the operator to type."""

import re
import shlex
import warnings

import click
from click.testing import CliRunner

from firecube.cli.main import cli

_OPTIONS = r'(?: (?:"[^"]*"|\S*[^\s.,;]))*'  # every token up to a line or sentence end


def run_hint(message: str, command: str, *extra_args: str) -> None:
    match = re.search(re.escape(command) + _OPTIONS, message)
    assert match, f"no {command!r} in message: {message!r}"
    result = CliRunner().invoke(cli, shlex.split(match.group(0))[1:] + list(extra_args))
    assert result.exit_code == 0, f"{match.group(0)!r} exited {result.exit_code}: {result.output}"


def parse_hint(message: str, command: str, *, samples: dict[str, str] | None = None) -> str:
    """Parse the printed ``command`` against the real CLI, running no callback; Click's error propagates."""
    start = message.find(command)
    assert start >= 0, f"no {command!r} in message: {message!r}"
    text = message[start:].split("`" if message[start - 1 : start] == "`" else "\n")[0]
    text = re.sub(r"<([\w -]+)>", lambda m: (samples or {}).get(m.group(1), "x"), text)
    cmd: click.Command = cli
    path: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # Click 8.4 deprecates protected_args
        ctx = cmd.make_context("firecube", shlex.split(text.rstrip(".,;"))[1:])
        while isinstance(cmd, click.Group) and ctx.protected_args:
            name, found, rest = cmd.resolve_command(ctx, [*ctx.protected_args, *ctx.args])
            assert found is not None and name is not None, f"no command in {text!r}"
            cmd = found
            path.append(name)
            ctx = cmd.make_context(name, rest, parent=ctx)
    return " ".join(path)
