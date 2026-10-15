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

"""Per-row state for table-driven tests built on pytest's ``subtests`` fixture.

A merged test loops over rows and wraps each one in ``table_row``::

    for row_id, args, expected in ROWS:
        with table_row(subtests, row_id, tmp_path, monkeypatch) as (workdir, mp):
            mp.setenv("X", args)
            assert run(workdir) == expected

Rules:

* ``row_id`` is the old test name or parameter id, so the old node id stays
  greppable in the new source. Ids must be unique within one test (a repeated
  id fails that row with ``FileExistsError``; other rows still run).
* Every row builds its own state inside the ``with`` block, in its own
  ``workdir``. Do not share mutable objects, files or module state across
  rows; build them per row, or keep them immutable.
* Patch through the yielded ``mp``, not the outer ``monkeypatch`` fixture: only
  ``mp`` is undone at the end of the row. Without a ``monkeypatch`` argument
  ``mp`` is ``None``.
* A failing row is reported under its own id and does not stop later rows
  (except under ``-x``/``--maxfail``). Fixtures are still function-scoped and
  shared by the rows; ``table_row`` only isolates the directory and the patches.
* NOT reset per row: the outer ``monkeypatch`` fixture and anything patched
  through it (including environment variables), the process working directory,
  module-level globals, and caches or registries (``lru_cache``, plugin
  registries). Patch through ``mp`` and clear such state inside the row.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import pytest


class Row(NamedTuple):
    workdir: Path
    mp: pytest.MonkeyPatch | None


@contextmanager
def table_row(
    subtests: Any,
    row_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> Iterator[Row]:
    """Run one table row as a subtest with a fresh directory and fresh patches.

    The directory is ``<=40 safe chars of row_id>-<10 hex of sha256(row_id)>``:
    distinct ids never collide and the name length is bounded. An exact repeat
    of an id finds its directory taken; that failure is raised at the end of
    that row (the body still runs, on a directory that does not exist anew) and
    is reported under the row's id.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", row_id).strip("_")[:40] or "row"
    digest = hashlib.sha256(row_id.encode("utf-8", "surrogatepass")).hexdigest()[:10]
    workdir = tmp_path / f"{slug}-{digest}"
    with subtests.test(msg=row_id):
        setup_error: OSError | None = None
        try:
            workdir.mkdir()
        except OSError as exc:
            setup_error = exc
        if monkeypatch is None:
            yield Row(workdir, None)
        else:
            with monkeypatch.context() as mp:
                yield Row(workdir, mp)
        if setup_error is not None:
            raise setup_error
