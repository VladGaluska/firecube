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

"""Process-wide locks on stored write units, and the store identity they are keyed by."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from zarr.storage import LocalStore, MemoryStore, StorePath

from firecube.core.uris import is_remote_target, local_path_from_target, storage_uri_from_target


def _store_identity(store_uri: str, store: object | None) -> str:
    """The backing store of a writer, spelled one way for the in-process unit locks.

    A local directory is its real path, however it is addressed: a supplied
    ``LocalStore``, a plain, relative or ``file://`` path, a trailing slash,
    a symlink. Any other store is its normalised URI.
    """
    if isinstance(store, LocalStore):
        return str(Path(store.root).resolve())
    target = store if isinstance(store, str) else store_uri
    if is_remote_target(target):
        return storage_uri_from_target(target).to_str()
    return str(local_path_from_target(target).resolve())


_PATH_PROBE_STORE = MemoryStore()


def canonical_array_path(path: str) -> str:
    """The path zarr stores ``path`` under, spelled one way for locks and ledgers.

    Zarr drops leading, trailing and repeated slashes and reads a backslash as a
    slash, so ``vec``, ``vec/``, ``/vec`` and ``data//vec`` address one node.
    The public ``StorePath`` applies that rule, so the lock table never
    restates it.

    Raises:
        ValueError: A ``.`` or ``..`` segment, which zarr refuses too.
    """
    return StorePath(_PATH_PROBE_STORE, path).path


LockKey = tuple[str, str, tuple[int, ...]]
"""``(store identity, canonical array path, unit index)``; unit ``()`` is the whole array."""


@dataclass
class _LockEntry:
    """One unit's mutex and the number of threads holding or waiting for it."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


class _UnitLocks:
    """Process-wide locks on stored write units, keyed by (store, array, unit index).

    A partial write reads a whole stored object, merges into it and writes it
    back; two threads doing that to one object lose the first write. Locks are
    taken in sorted key order (no deadlock between writers touching several
    units) and an entry is dropped once no thread holds or waits for it, so the
    table only ever holds units in use.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[LockKey, _LockEntry] = {}

    @contextlib.contextmanager
    def hold(self, keys: Iterable[LockKey]) -> Iterator[None]:
        held: list[LockKey] = []
        try:
            for key in sorted(set(keys)):
                with self._guard:
                    entry = self._entries.setdefault(key, _LockEntry())
                    entry.users += 1
                try:
                    entry.lock.acquire()
                except BaseException:
                    self._drop(key)
                    raise
                held.append(key)
            yield
        finally:
            for key in reversed(held):
                self._entries[key].lock.release()
                self._drop(key)

    def _drop(self, key: LockKey) -> None:
        with self._guard:
            entry = self._entries[key]
            entry.users -= 1
            if entry.users == 0:
                del self._entries[key]


_UNIT_LOCKS = _UnitLocks()
