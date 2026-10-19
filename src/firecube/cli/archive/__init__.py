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

"""Archive CLI commands for Zip (.zip) and Tensogram (.tgm) archival.

Archive subcommands distinguish two URI categories:

* **Source/target Zarr products** are resolved via `ProductResolver` and feed
  a `StorageSession` so that driver, endpoint, and credential resolution
  happen at the CLI boundary.
* **`.tgm` and `.zip` artifact files** are *not* products. They are external archive
  files and use raw `StorageUri` only —
  product resolvers MUST NOT be used for them. ``info``/``validate``/``list``
  operate on artifacts only and never construct a session.
"""

# pyright: reportMissingImports=false, reportCallIssue=false

from __future__ import annotations

import click

from firecube.cli.archive._create import create
from firecube.cli.archive._inspect import info, list_cmd, validate
from firecube.cli.archive._restore import restore
from firecube.core import observability


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
def archive(ctx: click.Context) -> None:
    """create and manage Zip and Tensogram archives

    Archives compress all array groups from a Zarr store into a single portable
    file using configurable codecs. Use archive create to produce an archive
    and archive restore to extract it back to Zarr.
    """
    observability.init_observability("firecube-archive")
    ctx.ensure_object(dict)


archive.add_command(create)
archive.add_command(restore)
archive.add_command(info)
archive.add_command(validate)
archive.add_command(list_cmd)

__all__ = ["archive"]
