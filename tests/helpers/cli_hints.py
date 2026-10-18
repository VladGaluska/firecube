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

from click.testing import CliRunner

from firecube.cli.main import cli

_OPTIONS = r'(?: (?:"[^"]*"|\S*[^\s.,;]))*'  # every token up to a line or sentence end


def run_hint(message: str, command: str, *extra_args: str) -> None:
    match = re.search(re.escape(command) + _OPTIONS, message)
    assert match, f"no {command!r} in message: {message!r}"
    result = CliRunner().invoke(cli, shlex.split(match.group(0))[1:] + list(extra_args))
    assert result.exit_code == 0, f"{match.group(0)!r} exited {result.exit_code}: {result.output}"
