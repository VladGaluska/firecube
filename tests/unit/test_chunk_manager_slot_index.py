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

"""Unit tests for ChunkManager.ensure_slot_index_model and companions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import zarr
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from firecube.core.controlplane import ChunkManager
from firecube.core.controlplane.types import (
    SLOT_INDEX_CURRENT_FILENAME,
    SLOT_INDEX_DIRNAME,
    ResolvedIndexRecord,
    SlotIndexModelRecord,
    WriteDomain,
    canonical_index_bytes,
)
from firecube.core.errors import ManifestError, SlotIndexModelConflictError
from firecube.core.filesystem import StorageFilesystem
from firecube.core.product.identity import ProductIdentity
from firecube.core.slot_index import (
    SLOT_INDEX_MODEL_ATTR,
    SLOT_INDEX_MODEL_IDENTITY_HASH_ATTR,
    SlotAxis,
    SlotIndexModel,
)
from firecube.core.storage.binding import StorageBinding
from firecube.core.storage.driver_config import StorageDriverConfig
from firecube.core.storage.uri import StorageUri
from firecube.core.zarr.guard.metadata_authority import _NODE_LOCKS


def _make_manager(tmp_path: Path) -> ChunkManager:
    product_uri = StorageUri.from_local_path(tmp_path / "__firecube_controlplane__")
    binding = StorageBinding(
        identity=ProductIdentity.from_uri(product_uri, "zarr", product_name="control_product"),
        driver=StorageDriverConfig(),
    )
    return ChunkManager(binding=binding, workspace=tmp_path)


def _model(name: str = "opera_v1") -> SlotIndexModel:
    return SlotIndexModel(
        name=name,
        epoch="2026-01-01T00:00:00Z",
        groups={"g1": SlotAxis(cadence_s=300, mode="exact")},
    )


def _control_root(tmp_path: Path, product: str) -> Path:
    return tmp_path / product / ".firecube"


def _slot_index_current(tmp_path: Path, product: str) -> Path:
    return _control_root(tmp_path, product) / SLOT_INDEX_DIRNAME / SLOT_INDEX_CURRENT_FILENAME


def _claim_file(tmp_path: Path, product: str) -> Path:
    domain = WriteDomain(product=product, category="slot_index_model", name="current")
    return _control_root(tmp_path, product) / "claims" / domain.claim_name


def _read_zarr_attrs_hash(tmp_path: Path, product: str) -> str | None:
    try:
        store = LocalStore(str(tmp_path / product))
        root = zarr.open_group(store=store, mode="r", zarr_format=3)
    except Exception:
        return None
    value = root.attrs.get(SLOT_INDEX_MODEL_IDENTITY_HASH_ATTR)
    return None if value is None else str(value)


def _start_run(cm: ChunkManager, tmp_path: Path, product: str, run_id: str) -> None:
    cm.record_run_started(
        product=product,
        run_id=run_id,
        output_path=str(tmp_path / product),
        output_format="zarr",
        size=0,
        meta={"plugin": "test"},
    )


@pytest.mark.unit
def test_fresh_store_records_model(tmp_path):
    cm = _make_manager(tmp_path)
    _start_run(cm, tmp_path, "prod1", "run-1")
    model = _model()

    record = cm.ensure_slot_index_model(product="prod1", model=model, run_id="run-1")

    assert record.identity_hash == model.identity_hash
    assert record.model.name == model.name
    cp_file = _slot_index_current(tmp_path, "prod1")
    assert cp_file.is_file()
    on_disk = SlotIndexModelRecord.from_json_bytes(cp_file.read_bytes())
    assert on_disk.identity_hash == model.identity_hash
    assert _read_zarr_attrs_hash(tmp_path, "prod1") == model.identity_hash


@pytest.mark.unit
def test_get_slot_index_model_returns_none_for_nonexistent(tmp_path):
    cm = _make_manager(tmp_path)
    assert cm.get_slot_index_model(product="nonexistent") is None


@pytest.mark.unit
def test_get_slot_index_model_propagates_non_missing_read_errors(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    cm = _make_manager(tmp_path)

    class _FakePath:
        def __init__(self, parts: tuple[str, ...] = ()) -> None:
            self.parts = parts

        def join(self, *segments: str) -> _FakePath:
            return _FakePath((*self.parts, *segments))

    class _FailingFilesystem:
        def open(self, path: _FakePath, mode: str):
            _ = (path, mode)
            raise PermissionError("permission denied while reading current.json")

    monkeypatch.setattr(
        cm.repo,
        "get_control_root_uri",
        lambda _product: "file:///control/prod1/.firecube",
    )
    monkeypatch.setattr(
        cm.repo,
        "_get_fs",
        lambda _uri: (_FailingFilesystem(), _FakePath()),
    )

    with pytest.raises(PermissionError, match="permission denied"):
        cm.get_slot_index_model(product="prod1")


@pytest.mark.unit
def test_slot_index_current_record_is_not_published_via_truncating_write(tmp_path):
    cm = _make_manager(tmp_path)
    _start_run(cm, tmp_path, "prod1", "run-1")
    model = _model()
    real_fs, real_root = cm.repo._get_fs(cm.repo.get_control_root_uri("prod1"))
    read_during_write_errors: list[BaseException] = []

    class _ProxyFilesystem:
        def __init__(self, wrapped: StorageFilesystem, on_current_json_open: Callable[[], None]):
            self._wrapped = wrapped
            self._on_current_json_open = on_current_json_open

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

        def open(self, uri: StorageUri, mode: str = "rb") -> Any:
            handle = self._wrapped.open(uri, mode)
            if mode == "wb" and uri.path.endswith(
                f"/{SLOT_INDEX_DIRNAME}/{SLOT_INDEX_CURRENT_FILENAME}"
            ):
                self._on_current_json_open()
            return handle

    def read_while_current_json_is_truncated() -> None:
        try:
            cm.get_slot_index_model(product="prod1")
        except ManifestError as exc:
            read_during_write_errors.append(exc)

    proxy_fs = _ProxyFilesystem(real_fs, read_while_current_json_is_truncated)
    cm.repo._filesystem = cast(StorageFilesystem, proxy_fs)

    cm.ensure_slot_index_model(product="prod1", model=model, run_id="run-1")

    assert real_root.to_str().endswith("/prod1/.firecube")
    assert read_during_write_errors == []


@pytest.mark.unit
def test_mirror_attrs_for_remote_product_uses_driver_aware_zarr_store(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    cm = _make_manager(tmp_path)
    model = _model()
    record = SlotIndexModelRecord(
        model=model,
        identity_hash=model.identity_hash,
        schema_version="v1",
        recorded_at="2026-01-01T00:00:00Z",
        recorded_by_run_id="run-1",
    )
    remote_store = MemoryStore()  # no root group yet: the mirror creates it
    sentinel_storage_options = {"client_kwargs": {"endpoint_url": "https://example.invalid"}}
    create_calls: list[dict[str, object]] = []
    from_url_calls: list[tuple[str, object, bool]] = []

    class _StoreHandle:
        """Shaped like the fsspec driver's handle: a URI string plus options."""

        def zarr_kwargs(self) -> dict[str, object]:
            return {
                "store": "s3://bucket/prod1.zarr",
                "storage_options": sentinel_storage_options,
            }

    def fake_create_zarr_store(*, uri: str, storage_config: object, mode: str):
        create_calls.append({"uri": uri, "storage_config": storage_config, "mode": mode})
        return _StoreHandle()

    def fail_create_obstore_store(*args: object, **kwargs: object):
        _ = (args, kwargs)
        raise AssertionError(
            "slot-index root attrs must use create_zarr_store, not create_obstore_store directly"
        )

    monkeypatch.setattr(
        ChunkManager,
        "get_product_root",
        lambda self, product: "s3://bucket/prod1.zarr",
    )
    monkeypatch.setattr(
        "firecube.core.filesystem.store_factory.create_zarr_store",
        fake_create_zarr_store,
    )
    monkeypatch.setattr(
        "firecube.core.filesystem.store_factory.create_obstore_store",
        fail_create_obstore_store,
    )

    def fake_from_url(
        url: str, storage_options: object = None, read_only: bool = False
    ) -> MemoryStore:
        from_url_calls.append((url, storage_options, read_only))
        return remote_store

    monkeypatch.setattr(FsspecStore, "from_url", fake_from_url)

    cm._mirror_attrs("prod1", record)

    assert create_calls == [
        {
            "uri": "s3://bucket/prod1.zarr",
            "storage_config": cm.storage_config,
            "mode": "a",
        }
    ]
    assert from_url_calls == [("s3://bucket/prod1.zarr", sentinel_storage_options, False)]
    stored = zarr.open_group(store=remote_store, mode="r", zarr_format=3).attrs
    assert stored[SLOT_INDEX_MODEL_ATTR] == model.canonical_bytes().decode("utf-8")
    assert stored[SLOT_INDEX_MODEL_IDENTITY_HASH_ATTR] == model.identity_hash


@pytest.mark.unit
def test_root_creation_by_a_paused_mirror_does_not_erase_a_completed_mirror(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Two mirrors meet an absent root; the first pauses as it creates the root.

    The second completes its whole mirror in that pause. Creating the root must
    not replace the document the second one wrote, or the second's attrs vanish.
    """
    cm = _make_manager(tmp_path)
    model = _model()
    legacy = SlotIndexModelRecord(
        model=model,
        identity_hash=model.identity_hash,
        schema_version="v1",
        recorded_at="2026-01-01T00:00:00Z",
        recorded_by_run_id="run-1",
    )
    index = {"schema_version": "v1", "name": "idx", "groups": {}}
    resolved = ResolvedIndexRecord(
        recorded_at="2026-01-01T00:00:00+00:00",
        recorded_by_run_id="run-2",
        identity_hash=hashlib.sha256(canonical_index_bytes(index)).hexdigest(),
        index=index,
    )
    paused = False
    second_done = threading.Event()
    second_errors: list[BaseException] = []

    def run_second_mirror() -> None:
        try:
            cm._mirror_resolved_index_attrs("prod1", resolved)
        except BaseException as exc:  # reported by the test thread
            second_errors.append(exc)
        finally:
            second_done.set()

    second = threading.Thread(target=run_second_mirror)

    def wait_until_second_settled(timeout_s: float = 30.0) -> bool:
        """Wait until the second mirror finished, or is queued behind the first on a node lock."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with _NODE_LOCKS._guard:
                queued = any(entry.users >= 2 for entry in _NODE_LOCKS._entries.values())
            if second_done.is_set() or queued:
                return True
            time.sleep(0.005)
        return False

    def pause_at_root_creation(original: Any) -> Any:
        async def put(self: LocalStore, key: str, *args: Any, **kwargs: Any) -> Any:
            nonlocal paused
            if key == "zarr.json" and not paused:
                paused = True
                second.start()
                # Guarded creation holds the root's node lock, so the second
                # mirror queues behind it; unguarded creation lets it run to
                # the end. Resume only once one of the two is observed.
                assert await asyncio.to_thread(wait_until_second_settled)
            return await original(self, key, *args, **kwargs)

        return put

    monkeypatch.setattr(LocalStore, "set", pause_at_root_creation(LocalStore.set))
    monkeypatch.setattr(
        LocalStore, "set_if_not_exists", pause_at_root_creation(LocalStore.set_if_not_exists)
    )

    cm._mirror_attrs("prod1", legacy)

    second.join(timeout=30)
    assert paused
    assert second_errors == []
    stored = zarr.open_group(store=LocalStore(str(tmp_path / "prod1")), mode="r", zarr_format=3)
    assert stored.attrs[SLOT_INDEX_MODEL_IDENTITY_HASH_ATTR] == model.identity_hash
    assert stored.attrs["firecube_resolved_index_identity_hash"] == resolved.identity_hash


@pytest.mark.unit
def test_read_slot_index_attrs_hash_for_remote_product_uses_driver_aware_zarr_store(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    cm = _make_manager(tmp_path)
    expected_hash = _model().identity_hash
    sentinel_store = object()
    sentinel_storage_options = {"client_kwargs": {"endpoint_url": "https://example.invalid"}}
    create_calls: list[dict[str, object]] = []
    open_calls: list[dict[str, object]] = []

    class _Root:
        def __init__(self) -> None:
            self.attrs = {SLOT_INDEX_MODEL_IDENTITY_HASH_ATTR: expected_hash}

    class _StoreHandle:
        def zarr_kwargs(self) -> dict[str, object]:
            return {
                "store": sentinel_store,
                "storage_options": sentinel_storage_options,
            }

    def fake_create_zarr_store(*, uri: str, storage_config: object, mode: str):
        create_calls.append({"uri": uri, "storage_config": storage_config, "mode": mode})
        return _StoreHandle()

    def fail_create_obstore_store(*args: object, **kwargs: object):
        _ = (args, kwargs)
        raise AssertionError(
            "slot-index root attrs must use create_zarr_store, not create_obstore_store directly"
        )

    def fake_open_group(**kwargs: object):
        open_calls.append(dict(kwargs))
        return _Root()

    monkeypatch.setattr(
        ChunkManager,
        "get_product_root",
        lambda self, product: "s3://bucket/prod1.zarr",
    )
    monkeypatch.setattr(
        "firecube.core.filesystem.store_factory.create_zarr_store",
        fake_create_zarr_store,
    )
    monkeypatch.setattr(
        "firecube.core.filesystem.store_factory.create_obstore_store",
        fail_create_obstore_store,
    )
    monkeypatch.setattr(zarr, "open_group", fake_open_group)

    result = cm.read_slot_index_attrs_hash(product="prod1")

    assert create_calls == [
        {"uri": "s3://bucket/prod1.zarr", "storage_config": cm.storage_config, "mode": "r"}
    ]
    assert open_calls == [
        {
            "store": sentinel_store,
            "storage_options": sentinel_storage_options,
            "mode": "r",
            "zarr_format": 3,
            "use_consolidated": False,
        }
    ]
    assert result == expected_hash


@pytest.mark.unit
def test_second_call_same_model_verified(tmp_path):
    cm = _make_manager(tmp_path)
    _start_run(cm, tmp_path, "prod1", "run-1")
    model = _model()

    first = cm.ensure_slot_index_model(product="prod1", model=model, run_id="run-1")
    on_disk_first = SlotIndexModelRecord.from_json_bytes(
        _slot_index_current(tmp_path, "prod1").read_bytes()
    )
    second = cm.ensure_slot_index_model(product="prod1", model=model, run_id="run-1")
    on_disk_second = SlotIndexModelRecord.from_json_bytes(
        _slot_index_current(tmp_path, "prod1").read_bytes()
    )

    assert first.identity_hash == second.identity_hash == model.identity_hash
    assert on_disk_first.recorded_at == on_disk_second.recorded_at
    assert on_disk_first.recorded_by_run_id == on_disk_second.recorded_by_run_id


@pytest.mark.unit
def test_different_model_raises_conflict(tmp_path):
    cm = _make_manager(tmp_path)
    _start_run(cm, tmp_path, "prod1", "run-1")
    cm.ensure_slot_index_model(product="prod1", model=_model(name="opera_v1"), run_id="run-1")
    _start_run(cm, tmp_path, "prod1", "run-2")
    other = _model(name="other_v1")
    assert other.identity_hash != _model(name="opera_v1").identity_hash

    with pytest.raises(SlotIndexModelConflictError):
        cm.ensure_slot_index_model(product="prod1", model=other, run_id="run-2")


@pytest.mark.unit
def test_empty_run_id_raises(tmp_path):
    cm = _make_manager(tmp_path)
    with pytest.raises(ValueError):
        cm.ensure_slot_index_model(product="prod1", model=_model(), run_id="")


@pytest.mark.unit
def test_cp_only_state_triggers_row1_remirror(tmp_path):
    """Backward-compat: CP-only match with attrs absent → re-mirror + accept.

    Pre-seeds current.json (CP present, matches model.identity_hash) but leaves
    the zarr root WITHOUT the identity-hash attr, and pins a fake winner claim
    so acquire_claim raises ClaimConflictError. The loser must execute Row 1
    of the unified 5-row convergence policy: re-mirror attrs from the
    authoritative CP record, then return the CP record (no timeout).
    """
    cm = _make_manager(tmp_path)
    _start_run(cm, tmp_path, "prod1", "winner")
    model = _model()
    winner_record = SlotIndexModelRecord(
        model=model,
        identity_hash=model.identity_hash,
        schema_version="v1",
        recorded_at="2026-01-01T00:00:00+00:00",
        recorded_by_run_id="winner",
    )
    cp_file = _slot_index_current(tmp_path, "prod1")
    cp_file.parent.mkdir(parents=True, exist_ok=True)
    cp_file.write_bytes(winner_record.to_json_bytes())
    claim_file = _claim_file(tmp_path, "prod1")
    claim_file.parent.mkdir(parents=True, exist_ok=True)
    claim_file.write_text(
        json.dumps(
            {
                "product": "prod1",
                "domain": "prod1:slot_index_model:current",
                "owner_id": "winner",
                "claim_path": str(claim_file),
                "acquired_at": 9999999999.0,
                "last_heartbeat_at": 9999999999.0,
                "heartbeat_interval_s": 30,
                "stale_threshold_s": 120,
            }
        )
    )
    assert _read_zarr_attrs_hash(tmp_path, "prod1") is None

    returned = cm.ensure_slot_index_model(
        product="prod1",
        model=model,
        run_id="loser",
        max_retries=2,
        initial_backoff_s=0.01,
    )

    assert returned.identity_hash == model.identity_hash
    assert _read_zarr_attrs_hash(tmp_path, "prod1") == model.identity_hash, (
        "loser must re-mirror the identity hash to the zarr root attrs (Row 1)"
    )
