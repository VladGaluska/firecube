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

import asyncio
import threading
from dataclasses import replace
from pathlib import Path

CONTENDER_TIMEOUT_S = 30.0


def local_path(store):
    return Path(str(store).removeprefix("file://")).resolve()


class ContendedUnit:
    def __init__(self, store, chunk_key, unit):
        self.chunk_path = local_path(store) / chunk_key
        self.lock_key = (local_path(store), chunk_key.rsplit("/c/", 1)[0], unit)
        self.arrived = threading.Event()
        self._reads = 0
        self._counter = threading.Lock()

    def after_read(self):
        with self._counter:
            self._reads += 1
            first = self._reads == 1
        if not first:
            self.arrived.set()
            return
        if not self.arrived.wait(CONTENDER_TIMEOUT_S):
            raise AssertionError(
                f"no second writer reached {self.chunk_path} while the first held it"
            )


def force_contended_units(monkeypatch, *units):
    from zarr.storage import LocalStore

    gates = [ContendedUnit(*unit) for unit in units]
    by_path = {gate.chunk_path: gate for gate in gates}
    original = LocalStore.get

    async def get(self, key, *args, **kwargs):
        data = await original(self, key, *args, **kwargs)
        gate = by_path.get((Path(self.root) / key).resolve())
        if gate is not None:
            await asyncio.to_thread(gate.after_read)
        return data

    monkeypatch.setattr(LocalStore, "get", get)
    _observe_lock_contention(monkeypatch, {gate.lock_key: gate.arrived for gate in gates})
    return gates


def force_contended_static(monkeypatch, store, array_path):
    from firecube.core.zarr.guard._unit_locks import _store_identity
    from firecube.core.zarr.region_writer import RegionZarrWriter

    root = local_path(store)
    held = threading.Event()
    first_payload = []
    arrived = threading.Event()
    first_done = threading.Event()
    counter = threading.Lock()
    calls = [0]
    original = RegionZarrWriter.write_static

    def write_static(self, group, array_name, data):
        writer_store = local_path(_store_identity(self._store_uri, self._store))
        if f"{group}/{array_name}" != array_path or writer_store != root:
            return original(self, group, array_name, data)
        with counter:
            calls[0] += 1
            first = calls[0] == 1
        if not first:
            arrived.set()
            if not first_done.wait(CONTENDER_TIMEOUT_S):
                raise AssertionError(f"the held writer of {array_path} never wrote")
            return original(self, group, array_name, data)
        first_payload.append(data.copy())
        held.set()
        if not arrived.wait(CONTENDER_TIMEOUT_S):
            raise AssertionError(
                f"no second writer reached {root / array_path} while the first held it"
            )
        try:
            return original(self, group, array_name, data)
        finally:
            first_done.set()

    monkeypatch.setattr(RegionZarrWriter, "write_static", write_static)
    _observe_lock_contention(monkeypatch, {(root, array_path, ()): arrived})
    return held, first_payload


def _observe_lock_contention(monkeypatch, arrived):
    from firecube.core.zarr.guard import _unit_locks

    class _ObservedLock:
        def __init__(self, lock, event):
            self._lock, self._event = (lock, event)

        def acquire(self, blocking=True, timeout=-1):
            if self._lock.acquire(blocking=False):
                return True
            self._event.set()
            return self._lock.acquire(blocking, timeout)

        def release(self):
            self._lock.release()

    class _ObservedEntries(dict):
        def setdefault(self, key, default=None):
            store, array_path, unit = key
            event = arrived.get((local_path(str(store)), array_path, tuple(unit)))
            if key not in self and event is not None:
                assert default is not None
                default = replace(default, lock=_ObservedLock(default.lock, event))
            return super().setdefault(key, default)

    monkeypatch.setattr(_unit_locks._UNIT_LOCKS, "_entries", _ObservedEntries())
