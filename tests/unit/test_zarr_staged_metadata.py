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

"""Tests for staged metadata seeding."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from firecube.core.storage.session import StorageSession
from firecube.ingestor.runtime.zarr.staged_metadata import (
    StagedMetadataError,
    _is_coordinate_chunk,
    seed_staged_store_metadata,
)
from tests.helpers.storage import assert_no_fsspec_bypass, make_local_session


def _make_zarr_store(base: Path, group: str, array: str, shape: list[int]) -> None:
    """Create a minimal Zarr V3 store with zarr.json at group/array."""
    arr_path = base / group / array
    arr_path.mkdir(parents=True, exist_ok=True)
    meta = {
        "node_type": "array",
        "shape": shape,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [min(shape[0], 40), *shape[1:]]},
        },
        "dimension_names": ["timestamp"] + [f"d{i}" for i in range(len(shape) - 1)],
        "data_type": "float32",
        "fill_value": None,
        "chunk_key_encoding": {"name": "default", "separator": "/"},
        "codecs": [],
    }
    (arr_path / "zarr.json").write_text(json.dumps(meta))


def _deny_write_bytes_under(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make Path.write_bytes fail for deterministic staged-write failure tests."""
    original_write_bytes = Path.write_bytes

    def guarded_write_bytes(self: Path, data: bytes) -> int:
        try:
            self.relative_to(path)
        except ValueError:
            return original_write_bytes(self, data)
        raise PermissionError(f"denied: {self}")

    monkeypatch.setattr(Path, "write_bytes", guarded_write_bytes)


def test_seed_copies_zarr_json(tmp_path):
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    _make_zarr_store(final, "G", "val", [100, 3])

    from firecube.ingestor.runtime.zarr.staged_metadata import seed_staged_store_metadata

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        result = seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
        )

    assert result["G"]["seeded"] is True
    assert result["G"]["files"] >= 1
    seeded_meta = json.loads((temp / "G" / "val" / "zarr.json").read_text())
    assert seeded_meta["shape"] == [100, 3]


def test_seed_fresh_ingest_is_noop(tmp_path):
    final = tmp_path / "nonexistent.zarr"
    temp = tmp_path / "temp.zarr"

    from firecube.ingestor.runtime.zarr.staged_metadata import seed_staged_store_metadata

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        result = seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
        )

    assert result["G"]["seeded"] is False
    assert not temp.exists() or not (temp / "G").exists()


def test_seed_only_copies_zarr_json(tmp_path):
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    _make_zarr_store(final, "G", "val", [50, 3])
    # Add a chunk file that should NOT be copied
    chunk_dir = final / "G" / "val" / "c"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "0").write_bytes(b"chunk_data")

    from firecube.ingestor.runtime.zarr.staged_metadata import seed_staged_store_metadata

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
        )

    # zarr.json should be copied
    assert (temp / "G" / "val" / "zarr.json").exists()
    # chunk file should NOT be copied
    assert not (temp / "G" / "val" / "c" / "0").exists()


def test_seed_staged_store_metadata_requires_session(tmp_path):
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"

    with pytest.raises(TypeError, match="session"):
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
        )  # pyright: ignore[reportCallIssue]


def test_seed_staged_store_metadata_rejects_storage_config(tmp_path):
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    session = make_local_session(str(final))

    with pytest.raises(TypeError, match="storage_config"):
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
            storage_config={"driver": "fsspec"},  # pyright: ignore[reportCallIssue]
        )


def test_seed_staged_store_metadata_rejects_storage_options(tmp_path):
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    session = make_local_session(str(final))

    with pytest.raises(TypeError, match="storage_options"):
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
            storage_options={"key": "value"},  # pyright: ignore[reportCallIssue]
        )


@pytest.mark.parametrize("exc_type", [RuntimeError, ValueError])
def test_seed_staged_store_metadata_strict_wraps_unexpected_exception(tmp_path, exc_type):
    """Strict mode wraps any non-OSError failure as StagedMetadataError, cause chained."""
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    broken_session = MagicMock(spec=StorageSession)
    broken_session.fs.side_effect = exc_type("boom")

    with pytest.raises(StagedMetadataError, match="boom") as exc_info:
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=broken_session,
            strict=True,
        )
    assert isinstance(exc_info.value.__cause__, exc_type)


def test_seed_strict_raises_on_existing_target_failure(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """When final target exists but write to temp fails, strict mode must raise."""
    from firecube.ingestor.runtime.zarr.staged_metadata import (
        StagedMetadataError,
        seed_staged_store_metadata,
    )

    final = tmp_path / "final.zarr"
    arr = final / "G" / "val"
    arr.mkdir(parents=True)
    (arr / "zarr.json").write_text(json.dumps({"node_type": "array", "shape": [100, 3]}))

    temp = tmp_path / "temp.zarr"
    temp.mkdir()
    _deny_write_bytes_under(temp, monkeypatch)

    session = make_local_session(str(final))
    with pytest.raises(StagedMetadataError), assert_no_fsspec_bypass():
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            strict=True,
            session=session,
        )


def test_seed_strict_false_on_nonexistent_target(tmp_path):
    """When final target does not exist, strict=True still just skips (fresh ingest)."""
    from firecube.ingestor.runtime.zarr.staged_metadata import seed_staged_store_metadata

    final = tmp_path / "nonexistent.zarr"
    temp = tmp_path / "temp.zarr"

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        result = seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            strict=True,
            session=session,
        )

    assert result["G"]["seeded"] is False


def test_seed_partial_failure_raises_in_strict_mode(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Partial failure (1 of 2 groups fails) still raises in strict mode."""
    from firecube.ingestor.runtime.zarr.staged_metadata import (
        StagedMetadataError,
        seed_staged_store_metadata,
    )

    final = tmp_path / "final.zarr"
    for grp in ["G1", "G2"]:
        arr = final / grp / "val"
        arr.mkdir(parents=True)
        (arr / "zarr.json").write_text(json.dumps({"node_type": "array", "shape": [50, 3]}))

    temp = tmp_path / "temp.zarr"
    temp.mkdir()
    (temp / "G1").mkdir()
    g2 = temp / "G2"
    g2.mkdir()
    _deny_write_bytes_under(g2, monkeypatch)

    session = make_local_session(str(final))
    with pytest.raises(StagedMetadataError), assert_no_fsspec_bypass():
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G1", "G2"],
            strict=True,
            session=session,
        )


_T = ["timestamp"]
_COORDINATE_CHUNK_ROWS = (
    ("test_is_coordinate_chunk_matches_timestamp_chunk_v3", "data/timestamp/c/0", _T, True),
    ("test_is_coordinate_chunk_matches_nested_chunk_path", "data/timestamp/c/0/0", _T, True),
    ("test_is_coordinate_chunk_rejects_zarr_json", "data/timestamp/zarr.json", _T, False),
    (
        "test_is_coordinate_chunk_rejects_substring_collision",
        "data/timestamp_bounds/c/0",
        _T,
        False,
    ),
    ("test_is_coordinate_chunk_rejects_data_array_chunk", "data/val/c/0", _T, False),
    ("test_is_coordinate_chunk_returns_false_for_empty_coords", "data/timestamp/c/0", [], False),
    ("test_is_coordinate_chunk_returns_false_for_none_coords", "data/timestamp/c/0", None, False),
    ("test_is_coordinate_chunk_handles_trailing_slash", "data/timestamp/c/0/", _T, True),
    ("test_is_coordinate_chunk_matches_alternate_coord_name", "data/time/c/0", ["time"], True),
)


def test_is_coordinate_chunk(subtests):
    for row_id, path, coords, expected in _COORDINATE_CHUNK_ROWS:
        with subtests.test(msg=row_id):
            assert _is_coordinate_chunk(path, coords and list(coords)) is expected


def test_seed_copies_coord_chunks_when_coordinate_arrays_provided(tmp_path):
    """coordinate_arrays=['timestamp'] copies timestamp/c/* chunks but NOT val chunks."""
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    _make_zarr_store(final, "G", "timestamp", [2])
    ts_chunk_dir = final / "G" / "timestamp" / "c"
    ts_chunk_dir.mkdir(parents=True, exist_ok=True)
    (ts_chunk_dir / "0").write_bytes(b"timestamp_chunk_payload")
    _make_zarr_store(final, "G", "val", [2])
    val_chunk_dir = final / "G" / "val" / "c"
    val_chunk_dir.mkdir(parents=True, exist_ok=True)
    (val_chunk_dir / "0").write_bytes(b"val_chunk_payload")

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        result = seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
            coordinate_arrays=["timestamp"],
        )

    assert result["G"]["seeded"] is True
    ts_chunk_root = temp / "G" / "timestamp" / "c"
    assert ts_chunk_root.exists(), "timestamp chunk root not copied"
    assert any(ts_chunk_root.rglob("*")), "no timestamp chunk files copied"
    val_chunk_root = temp / "G" / "val" / "c"
    assert not val_chunk_root.exists() or not any(val_chunk_root.rglob("*")), (
        "val (data) chunks were wrongly copied"
    )


def test_seed_skips_coord_chunks_when_coordinate_arrays_is_none(tmp_path):
    """coordinate_arrays=None preserves exact legacy metadata-only behavior."""
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    _make_zarr_store(final, "G", "timestamp", [2])
    ts_chunk_dir = final / "G" / "timestamp" / "c"
    ts_chunk_dir.mkdir(parents=True, exist_ok=True)
    (ts_chunk_dir / "0").write_bytes(b"timestamp_chunk_payload")

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        result = seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
        )

    assert result["G"]["seeded"] is True
    ts_chunk_root = temp / "G" / "timestamp" / "c"
    assert not ts_chunk_root.exists() or not any(ts_chunk_root.rglob("*")), (
        "coord chunks copied despite coordinate_arrays=None"
    )


def test_seed_skips_coord_chunks_when_coordinate_arrays_is_empty_list(tmp_path):
    """coordinate_arrays=[] is semantically identical to None."""
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    _make_zarr_store(final, "G", "timestamp", [2])
    ts_chunk_dir = final / "G" / "timestamp" / "c"
    ts_chunk_dir.mkdir(parents=True, exist_ok=True)
    (ts_chunk_dir / "0").write_bytes(b"timestamp_chunk_payload")

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        result = seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
            coordinate_arrays=[],
        )

    assert result["G"]["seeded"] is True
    ts_chunk_root = temp / "G" / "timestamp" / "c"
    assert not ts_chunk_root.exists() or not any(ts_chunk_root.rglob("*")), (
        "coord chunks copied despite coordinate_arrays=[]"
    )


def test_seed_does_not_overwrite_existing_workspace_coord_chunk(tmp_path):
    """If a workspace coord chunk already exists, seeding must NOT overwrite it."""
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    _make_zarr_store(final, "G", "timestamp", [2])
    ts_chunk_dir = final / "G" / "timestamp" / "c"
    ts_chunk_dir.mkdir(parents=True, exist_ok=True)
    (ts_chunk_dir / "0").write_bytes(b"timestamp_chunk_payload")

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
            coordinate_arrays=["timestamp"],
        )

    ts_chunk_root = temp / "G" / "timestamp" / "c"
    chunk_files = [p for p in ts_chunk_root.rglob("*") if p.is_file()]
    assert chunk_files, "first seed should produce a chunk file"
    chunk_path = chunk_files[0]
    sentinel = b"\x00" * chunk_path.stat().st_size
    chunk_path.write_bytes(sentinel)

    with assert_no_fsspec_bypass():
        seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
            coordinate_arrays=["timestamp"],
        )

    assert chunk_path.read_bytes() == sentinel, (
        "second seed overwrote an existing workspace coord chunk"
    )


def test_seed_tolerates_group_lacking_named_coord_array(tmp_path):
    """A group that lacks the named coord array must not raise; silent no-op for those chunks."""
    final = tmp_path / "final.zarr"
    temp = tmp_path / "temp.zarr"
    _make_zarr_store(final, "G", "val", [2])

    session = make_local_session(str(final))
    with assert_no_fsspec_bypass():
        result = seed_staged_store_metadata(
            temp_store_uri=str(temp),
            final_target_uri=str(final),
            groups=["G"],
            session=session,
            coordinate_arrays=["timestamp"],
        )

    assert "G" in result
    ts_chunk_root = temp / "G" / "timestamp" / "c"
    assert not ts_chunk_root.exists() or not any(ts_chunk_root.rglob("*"))


# --- seeding under a metadata authority (DirectZarr) -------------------------------------


def _make_v3_target(final: Path) -> None:
    """A real Zarr v3 target: root group, group G, a written coordinate and a data array."""
    import numpy as np
    import zarr

    group = zarr.open_group(str(final), mode="w", zarr_format=3).require_group("G")
    group.attrs["title"] = "target"
    ts = group.create_array("timestamp", shape=(2,), chunks=(2,), dtype="int64")
    ts[:] = np.array([10, 20], dtype="int64")
    val = group.create_array("val", shape=(2, 3), chunks=(1, 3), dtype="float32")
    val[:] = np.ones((2, 3), dtype="float32")
    val.attrs["firecube_static_written"] = True


_SEEDED_DOCUMENTS = {"zarr.json", "G/zarr.json", "G/timestamp/zarr.json", "G/val/zarr.json"}


def _workspace_authority(temp: Path):
    from firecube.core.zarr.guard.metadata_authority import MetadataAuthority
    from firecube.core.zarr.region_writer import RegionZarrWriter

    return MetadataAuthority.process_local(RegionZarrWriter(str(temp)).metadata_scope())


def _workspace_files(temp: Path) -> dict[str, bytes]:
    return {p.relative_to(temp).as_posix(): p.read_bytes() for p in temp.rglob("*") if p.is_file()}


def _seed(temp: Path, final: Path, *, strict: bool = True, authority=None) -> dict:
    return seed_staged_store_metadata(
        temp_store_uri=str(temp),
        final_target_uri=str(final),
        groups=["G"],
        session=make_local_session(str(final)),
        strict=strict,
        coordinate_arrays=["timestamp"],
        authority=authority,
    )


def test_seed_with_authority_writes_every_document_through_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Each seeded zarr.json is published under the authority, with the plain path's bytes."""
    from firecube.ingestor.runtime.zarr import staged_metadata

    final = tmp_path / "final.zarr"
    _make_v3_target(final)
    plain = tmp_path / "plain.zarr"
    _seed(plain, final)  # no authority: today's path

    temp = tmp_path / "temp.zarr"
    authority = _workspace_authority(temp)
    published: list[tuple[str, object, str]] = []

    def recording(name: str):
        original = getattr(staged_metadata, name)

        def record(auth, store, key, payload):
            published.append((name, auth, key))
            return original(auth, store, key, payload)

        return record

    for name in ("publish_document", "publish_document_if_absent"):
        monkeypatch.setattr(staged_metadata, name, recording(name))
    result = _seed(temp, final, authority=authority)

    assert result["G"]["seeded"] is True
    # The root is replaced as before; every other document only where absent.
    assert sorted((name, key) for name, _, key in published) == sorted(
        [("publish_document", "zarr.json")]
        + [("publish_document_if_absent", key) for key in _SEEDED_DOCUMENTS - {"zarr.json"}]
    )
    assert all(auth is authority for _, auth, _ in published)
    # Same bytes as the plain path, the coordinate chunk included (an object copy).
    assert _workspace_files(temp) == _workspace_files(plain)
    assert "G/timestamp/c/0" in _workspace_files(temp)
    assert "G/val/c/0/0" not in _workspace_files(temp)


@pytest.mark.parametrize("strict", [True, False])
def test_seed_with_released_authority_refuses_and_leaves_no_document(tmp_path, strict):
    """A released authority refuses in strict and best-effort mode; no document lands."""
    from firecube.core.zarr.guard.metadata_authority import MetadataAuthorityError

    final = tmp_path / "final.zarr"
    _make_v3_target(final)
    temp = tmp_path / "temp.zarr"
    authority = _workspace_authority(temp)
    authority.revoke()

    with pytest.raises(MetadataAuthorityError, match="released"):
        _seed(temp, final, strict=strict, authority=authority)

    # No document, and no directory created for one.
    assert not temp.exists()


def test_seed_for_batch_does_not_swallow_a_refused_authority(tmp_path):
    """The best-effort batch wrapper propagates the refusal instead of logging it away."""
    from firecube.core.zarr.guard.metadata_authority import MetadataAuthorityError
    from firecube.ingestor.runtime.zarr.batch_runner import seed_staged_metadata_for_batch

    final = tmp_path / "final.zarr"
    _make_v3_target(final)
    temp = tmp_path / "temp.zarr"
    authority = _workspace_authority(temp)
    authority.revoke()
    ctx = MagicMock()
    ctx.storage.output = make_local_session(str(final))

    with pytest.raises(MetadataAuthorityError):
        seed_staged_metadata_for_batch(
            ctx=ctx,
            store_uri=str(temp),
            final_target_uri=str(final),
            resume_existing=False,
            force_reingest=False,
            write_mode="staged",
            logger=MagicMock(),
            coordinate_arrays=["timestamp"],
            authority=authority,
        )

    assert not temp.exists() or not list(temp.rglob("zarr.json"))


def test_staged_session_seeding_authority_is_process_local_and_released_on_close(tmp_path):
    """The session's seeding authority covers the whole workspace and ends with the session."""
    import contextlib

    from firecube.core.zarr.guard.metadata_authority import AuthorityBasis
    from firecube.ingestor.runtime.zarr.write_session import ZarrWriteSession

    def session(write_mode: str, workspace_uri: str | None) -> ZarrWriteSession:
        return ZarrWriteSession(
            product="p",
            run_id="r",
            write_mode=write_mode,
            window=None,
            slot_group=None,
            acquire_claim=lambda **_kw: contextlib.nullcontext(),
            open_root=lambda _uri: None,
            open_final_root=lambda _ctx: None,
            workspace_uri=workspace_uri,
        )

    temp = tmp_path / "temp.zarr"
    staged = session("staged", str(temp))
    authority = staged.staged_seeding_authority()
    assert authority is not None
    assert authority.basis is AuthorityBasis.PROCESS_LOCAL
    assert authority.scope.prefix == ""
    assert authority.scope.store == str(temp.resolve())
    assert authority.active
    assert session("direct", str(temp)).staged_seeding_authority() is None

    staged.close()
    assert not authority.active


def test_engine_seeds_under_the_host_seeding_authority(monkeypatch: pytest.MonkeyPatch):
    """A host with the seeding hook gets its authority passed to the pre-batch seeding."""
    from types import SimpleNamespace

    from firecube.ingestor.runtime import engine
    from firecube.ingestor.runtime.zarr import batch_runner

    seed = MagicMock()
    monkeypatch.setattr(batch_runner, "seed_staged_metadata_pre_batch", seed)
    authority = object()
    host = SimpleNamespace(
        _log=MagicMock(),
        _resolve_time_dim_name=lambda: "time",
        _staged_seeding_authority=lambda: authority,
    )
    ctx = SimpleNamespace(output_format="zarr")

    hook = engine._zarr_pre_batch_hook(host, ctx)  # type: ignore[arg-type]
    assert hook is not None
    hook()

    seed.assert_called_once_with(
        host=host, ctx=ctx, logger=host._log, coordinate_arrays=["time"], authority=authority
    )


class _PausingFilesystem:
    """Delegates to a real filesystem; pauses once after reading the source at *suffix*."""

    def __init__(self, inner, suffix: str, paused, resume) -> None:
        self._inner, self._suffix, self._paused, self._resume = inner, suffix, paused, resume

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def open(self, uri, mode: str = "rb"):
        handle = self._inner.open(uri, mode)
        if uri.path.endswith(self._suffix):
            self._paused.set()
            assert self._resume.wait(10), "seeder A was never resumed"
        return handle


def test_concurrent_seed_never_replaces_a_document_a_writer_changed(tmp_path):
    """Seeder A decides to copy a document; B seeds it and the writer stamps it; A must not revert it.

    Order: A finds G/val/zarr.json absent in the workspace and reads the target
    copy, then pauses. B seeds the same document and the writer stamps the
    static marker on it under the workspace authority. A resumes. The marker
    must survive: a reverted marker lets a divergent static write through.
    """
    import threading

    from zarr.storage import LocalStore

    from firecube.core.zarr.guard.metadata_authority import set_attributes

    final = tmp_path / "final.zarr"
    _make_v3_target(final)
    temp = tmp_path / "temp.zarr"
    authority = _workspace_authority(temp)
    paused, resume = threading.Event(), threading.Event()
    session_a = make_local_session(str(final))
    pausing_fs = _PausingFilesystem(session_a.fs(), "G/val/zarr.json", paused, resume)
    session_a.fs = lambda: pausing_fs  # type: ignore[method-assign]
    errors: list[BaseException] = []

    def seeder_a() -> None:
        try:
            seed_staged_store_metadata(
                temp_store_uri=str(temp),
                final_target_uri=str(final),
                groups=["G"],
                session=session_a,
                coordinate_arrays=["timestamp"],
                authority=authority,
            )
        except BaseException as exc:  # reported below
            errors.append(exc)

    thread_a = threading.Thread(target=seeder_a)
    thread_a.start()
    try:
        assert paused.wait(10), "seeder A never reached the document read"
        _seed(temp, final, authority=authority)  # seeder B
        set_attributes(authority, LocalStore(temp), "G/val", {"firecube_static_written": True})
    finally:
        resume.set()
        thread_a.join(10)

    assert errors == []
    stored = json.loads((temp / "G" / "val" / "zarr.json").read_bytes())
    assert stored["attributes"].get("firecube_static_written") is True, (
        "seeder A replaced the document the writer had stamped"
    )
