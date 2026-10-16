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

import concurrent.futures
import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from moto.server import ThreadedMotoServer

from firecube.core.config import StorageConfig
from firecube.core.credentials import Credentials
from firecube.core.errors import StorageError
from firecube.core.product.identity import ProductIdentity
from firecube.core.storage import StorageWriteResult
from firecube.core.storage import session as session_module
from firecube.core.storage.binding import StorageBinding
from firecube.core.storage.driver_config import StorageDriverConfig
from firecube.core.storage.session import StorageSession  # pyright: ignore[reportMissingImports]
from firecube.core.storage.uri import StorageUri  # pyright: ignore[reportMissingImports]
from firecube.core.zarr.promotion_plan import (
    PlannedObject,
    PromotionAction,
    PromotionPlan,
    build_promotion_plan,
    list_workspace_objects,
)
from tests.helpers.storage import make_test_session


def _session(uri: str, *, storage_config: StorageConfig | None = None) -> StorageSession:
    path = Path(uri)
    if storage_config and storage_config.storage_type == "s3":
        return make_test_session(
            path.parent,
            product=path.name,
            protocol="s3",
            authority=getattr(storage_config, "bucket", None) or "bucket",
        )
    return make_test_session(path.parent, product=path.name)


def _write_zarr_json(path: Path, shape0: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"shape": [shape0, 50, 50]}), encoding="utf-8")


def test_upload_tree_accepts_file_source_parquet(tmp_path: Path) -> None:
    source = tmp_path / "product.parquet"
    source.write_bytes(b"x" * 1024)
    target = tmp_path / "target.parquet"

    result = _session(str(target)).upload_tree(
        StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    )

    assert target.read_bytes() == b"x" * 1024
    assert result.files_written == 1
    assert result.bytes_written == 1024


def test_upload_tree_accepts_directory_source_zarr(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    _write_zarr_json(source / "zarr.json", 10)
    (source / "data" / "c" / "0").parent.mkdir(parents=True)
    (source / "data" / "c" / "0").write_bytes(b"chunk")
    (source / "data" / "zarr.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "target.zarr"

    result = _session(str(target)).upload_tree(
        StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    )

    assert (target / "zarr.json").exists()
    assert (target / "data" / "c" / "0").read_bytes() == b"chunk"
    assert (target / "data" / "zarr.json").read_text(encoding="utf-8") == "{}"
    assert result.files_written == 3


def test_upload_tree_skips_zarr_json_when_source_shape_is_smaller(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    target = tmp_path / "target.zarr"
    _write_zarr_json(source / "zarr.json", 10)
    (source / "data.bin").write_bytes(b"new")
    _write_zarr_json(target / "zarr.json", 100)

    _session(str(target)).upload_tree(
        StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    )

    assert json.loads((target / "zarr.json").read_text(encoding="utf-8"))["shape"][0] == 100
    assert (target / "data.bin").read_bytes() == b"new"


def test_upload_tree_uploads_zarr_json_when_source_shape_is_larger(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    target = tmp_path / "target.zarr"
    _write_zarr_json(source / "zarr.json", 100)
    _write_zarr_json(target / "zarr.json", 10)

    _session(str(target)).upload_tree(
        StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    )

    assert json.loads((target / "zarr.json").read_text(encoding="utf-8"))["shape"][0] == 100


def test_upload_tree_first_error_raises_storage_error_and_aborts(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    for name, content in [("a.bin", b"a"), ("b.bin", b"b"), ("c.bin", b"c")]:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(content)
    target = tmp_path / "target.zarr"
    (target / "b.bin").parent.mkdir(parents=True, exist_ok=True)
    (target / "b.bin").mkdir()

    with pytest.raises(StorageError):
        _session(str(target)).upload_tree(
            StorageUri.from_local_path(source),
            StorageUri.from_local_path(target),
            parallel_workers=1,
        )

    assert (target / "a.bin").read_bytes() == b"a"
    assert not (target / "c.bin").exists()


def test_upload_tree_result_populates_all_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    source.write_bytes(b"12345")
    target = tmp_path / "target.h5"

    result = _session(str(target)).upload_tree(
        StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    )

    assert isinstance(result, StorageWriteResult)
    assert result.path == StorageUri.from_local_path(target).to_str()
    assert target.read_bytes() == b"12345"
    assert result.bytes_written == 5
    assert result.files_written == 1
    assert result.duration_s >= 0
    assert result.storage_type == "local"


def test_upload_tree_preserve_zarr_metadata_false_disables_skip(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    target = tmp_path / "target.zarr"
    _write_zarr_json(source / "zarr.json", 10)
    _write_zarr_json(target / "zarr.json", 100)

    _session(str(target)).upload_tree(
        StorageUri.from_local_path(source),
        StorageUri.from_local_path(target),
        preserve_zarr_metadata=False,
    )

    assert json.loads((target / "zarr.json").read_text(encoding="utf-8"))["shape"][0] == 10


def test_upload_tree_empty_source_directory_succeeds(tmp_path: Path) -> None:
    source = tmp_path / "empty.zarr"
    source.mkdir()
    target = tmp_path / "target.zarr"

    result = _session(str(target)).upload_tree(
        StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    )

    assert result.files_written == 0
    assert result.bytes_written == 0
    assert not target.exists()


class _LocalMirrorFs:
    def __init__(self, base: Path, root: str) -> None:
        self.base = base
        self.root = root.strip("/")

    def _local_path(self, path: Any) -> Path:
        rel_str = path.to_str() if hasattr(path, "to_str") else str(path)
        if "://" in rel_str:
            rel_str = rel_str.split("://", 1)[1]
        rel = rel_str.strip("/")
        if self.root and rel.startswith(self.root):
            rel = rel[len(self.root) :].strip("/")
        return self.base / rel

    def open(self, path: Any, mode: str = "rb") -> Any:
        local_path = self._local_path(path)
        if any(flag in mode for flag in ("w", "a", "+")):
            local_path.parent.mkdir(parents=True, exist_ok=True)
        return local_path.open(mode)

    def capabilities(self) -> set[type]:
        return set()

    def makedirs(self, path: Any, exist_ok: bool = False) -> None:
        self._local_path(path).mkdir(parents=True, exist_ok=exist_ok)


def test_upload_tree_file_source_to_remote_destination_uses_open_fs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "product.parquet"
    source.write_bytes(b"remote")
    remote_root = tmp_path / "remote"
    bindings: list[Any] = []
    fake_fs = _LocalMirrorFs(remote_root, "bucket/prefix")

    def _fake_create_filesystem(binding: Any) -> _LocalMirrorFs:
        bindings.append(binding)
        return fake_fs

    monkeypatch.setattr("firecube.core.storage.session.create_filesystem", _fake_create_filesystem)
    storage_config = StorageConfig(storage_type="s3", storage_driver="fsspec")
    storage_config.bucket = "bucket"  # type: ignore[attr-defined]
    session = _session("s3://bucket/prefix/product.parquet", storage_config=storage_config)

    result = session.upload_tree(
        StorageUri.from_local_path(source),
        StorageUri.parse("s3://bucket/prefix/product.parquet"),
    )

    assert (remote_root / "product.parquet").read_bytes() == b"remote"
    assert len(bindings) >= 1
    assert bindings[0].driver.driver == "fsspec"
    assert result.storage_type == "s3"


def test_upload_tree_directory_source_to_local_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "chunk.bin").write_bytes(b"payload")
    target = tmp_path / "target.zarr"

    result = _session(str(target)).upload_tree(
        StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    )

    assert (target / "nested" / "chunk.bin").read_bytes() == b"payload"
    assert result.path == StorageUri.from_local_path(target).to_str()
    assert result.files_written == 1


def test_upload_tree_rejects_remote_source(tmp_path: Path) -> None:
    session = _session(str(tmp_path / "target.zarr"))

    with pytest.raises(ValueError, match="local source"):
        session.upload_tree(
            StorageUri.parse("s3://bucket/source.zarr"), StorageUri.from_local_path(tmp_path)
        )


@pytest.mark.parametrize(
    ("n_files", "workers"),
    [(4, 1), (2, 4)],
    ids=[
        "test_upload_tree_parallel_workers_one_uses_sequential_fallback",
        "test_upload_tree_two_files_uses_sequential_fallback",
    ],
)
def test_upload_tree_uses_sequential_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_files: int, workers: int
) -> None:
    source = tmp_path / "source.zarr"
    for idx in range(n_files):
        (source / f"{idx}.bin").parent.mkdir(parents=True, exist_ok=True)
        (source / f"{idx}.bin").write_bytes(str(idx).encode())
    target = tmp_path / "target.zarr"

    def _fail_executor(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("parallel executor should not be used")

    monkeypatch.setattr(
        "firecube.core.storage.session.concurrent.futures.ThreadPoolExecutor", _fail_executor
    )

    result = _session(str(target)).upload_tree(
        StorageUri.from_local_path(source),
        StorageUri.from_local_path(target),
        parallel_workers=workers,
    )

    assert result.files_written == n_files
    assert (target / f"{n_files - 1}.bin").read_bytes() == str(n_files - 1).encode()


def test_upload_tree_parallel_upload_uses_four_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zarr"
    for idx in range(5):
        (source / f"{idx}.bin").parent.mkdir(parents=True, exist_ok=True)
        (source / f"{idx}.bin").write_bytes(str(idx).encode())
    target = tmp_path / "target.zarr"
    worker_counts: list[int] = []
    submit_count = 0
    real_executor = concurrent.futures.ThreadPoolExecutor

    class _ExecutorSpy(real_executor):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            worker_counts.append(int(kwargs["max_workers"]))
            super().__init__(*args, **kwargs)

        def submit(self, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
            nonlocal submit_count
            submit_count += 1
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(
        "firecube.core.storage.session.concurrent.futures.ThreadPoolExecutor", _ExecutorSpy
    )

    result = _session(str(target)).upload_tree(
        StorageUri.from_local_path(source),
        StorageUri.from_local_path(target),
        parallel_workers=4,
    )

    assert worker_counts == [4]
    assert submit_count == 5
    assert result.files_written == 5
    assert (target / "4.bin").read_bytes() == b"4"


def test_upload_tree_custom_parallel_workers_respected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zarr"
    for idx in range(10):
        (source / f"{idx}.bin").parent.mkdir(parents=True, exist_ok=True)
        (source / f"{idx}.bin").write_bytes(str(idx).encode())
    target = tmp_path / "target.zarr"
    worker_counts: list[int] = []
    real_executor = concurrent.futures.ThreadPoolExecutor

    class _ExecutorSpy(real_executor):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            worker_counts.append(int(kwargs["max_workers"]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "firecube.core.storage.session.concurrent.futures.ThreadPoolExecutor", _ExecutorSpy
    )

    _session(str(target)).upload_tree(
        StorageUri.from_local_path(source),
        StorageUri.from_local_path(target),
        parallel_workers=8,
    )

    assert worker_counts == [8]


def test_upload_tree_large_file_routes_through_multipart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    large = source / "large.bin"
    large.write_bytes(b"large")
    remote_root = tmp_path / "remote"
    put_file_calls: list[tuple[str, str]] = []

    class _RawMultipartFs(_LocalMirrorFs):
        def put_file(self, local_path: str, remote_path: str) -> None:
            put_file_calls.append((local_path, remote_path))
            with (
                Path(local_path).open("rb") as src_handle,
                self.open(remote_path, "wb") as dst_handle,
            ):
                dst_handle.write(src_handle.read())

    fake_fs = _RawMultipartFs(remote_root, "bucket/prefix")

    def _fake_create_filesystem(binding: Any) -> _RawMultipartFs:
        return fake_fs

    monkeypatch.setattr("firecube.core.storage.session.create_filesystem", _fake_create_filesystem)
    storage_config = StorageConfig(storage_type="s3", storage_driver="fsspec")
    storage_config.bucket = "bucket"  # type: ignore[attr-defined]
    session = _session("s3://bucket/prefix", storage_config=storage_config)

    result = session.upload_tree(
        StorageUri.from_local_path(source),
        StorageUri.parse("s3://bucket/prefix"),
        multipart_threshold=1,
    )

    assert put_file_calls == [(str(large), "s3://bucket/prefix/large.bin")]
    assert (remote_root / "large.bin").read_bytes() == b"large"
    assert result.bytes_written == 5


def test_upload_tree_parallel_first_error_cancels_remaining_futures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zarr"
    for idx in range(3):
        (source / f"{idx}.bin").parent.mkdir(parents=True, exist_ok=True)
        (source / f"{idx}.bin").write_bytes(str(idx).encode())
    target = tmp_path / "target.zarr"
    futures: list[concurrent.futures.Future[tuple[int, int]]] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class _FailingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def submit(self, *args: Any, **kwargs: Any) -> concurrent.futures.Future[tuple[int, int]]:
            future: concurrent.futures.Future[tuple[int, int]] = concurrent.futures.Future()
            if not futures:
                future.set_exception(RuntimeError("boom"))
            futures.append(future)
            return future

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            shutdown_calls.append((wait, cancel_futures))

    def _as_completed(items: Any) -> list[concurrent.futures.Future[tuple[int, int]]]:
        return list(items)

    monkeypatch.setattr(
        "firecube.core.storage.session.concurrent.futures.ThreadPoolExecutor", _FailingExecutor
    )
    monkeypatch.setattr(
        "firecube.core.storage.session.concurrent.futures.as_completed", _as_completed
    )

    with pytest.raises(StorageError, match="boom"):
        _session(str(target)).upload_tree(
            StorageUri.from_local_path(source),
            StorageUri.from_local_path(target),
            parallel_workers=4,
        )

    assert shutdown_calls == [(False, True)]
    assert all(future.cancelled() for future in futures[1:])


def test_upload_tree_file_source_does_not_duplicate_filename(tmp_path: Path) -> None:
    """Regression: dst is exact destination, not parent dir, for file source."""
    source = tmp_path / "product.parquet"
    source.write_bytes(b"data")
    target = tmp_path / "out" / "product.parquet"

    _session(str(target)).upload_tree(
        StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    )

    assert target.read_bytes() == b"data"
    assert not target.is_dir()


# --- promotion plan (pure delta rules) ---------------------------------------


def _plan(
    workspace: dict[str, int],
    written: set[str] | None = None,
    absent: set[str] | None = None,
    unchanged: set[str] | None = None,
) -> PromotionPlan:
    return build_promotion_plan(
        workspace_objects=workspace,
        written_units=written or set(),
        absent_units=absent or set(),
        is_unchanged=lambda key: key in (unchanged or set()),
    )


def test_promotion_plan_absent_not_subset_of_written_raises() -> None:
    with pytest.raises(ValueError, match="a/c/9"):
        _plan({}, written={"a/c/0"}, absent={"a/c/0", "a/c/9"})


def test_promotion_plan_zarr_json_is_last_and_follows_shape_rule() -> None:
    plan = _plan(
        {"zarr.json": 1, "a/zarr.json": 2, "a/c/0": 3, "b/c/0": 4},
        written={"a/c/0", "b/c/9"},
        absent={"b/c/9"},
        unchanged={"a/zarr.json"},
    )

    keys = [o.key for o in plan.objects]
    assert keys[-2:] == ["a/zarr.json", "zarr.json"]
    assert [o.reason for o in plan.objects[-2:]] == ["metadata", "metadata"]
    assert [o.action for o in plan.objects[-2:]] == [PromotionAction.UNCHANGED, PromotionAction.PUT]
    assert all(not o.is_metadata for o in plan.objects[:-2])


def test_list_workspace_objects_lists_only_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws.zarr"
    (workspace / "a" / "c").mkdir(parents=True)
    (workspace / "a" / "c" / "0").write_bytes(b"12345")
    (workspace / "zarr.json").write_text("{}", encoding="utf-8")
    (workspace / "empty").mkdir()

    assert list_workspace_objects(workspace) == {"a/c/0": 5, "zarr.json": 2}


# --- promotion publication (session plan_promotion / publish_plan) -----------


def _put_files(root: Path, objects: dict[str, bytes]) -> None:
    for key, payload in objects.items():
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _group_doc(name: str) -> bytes:
    return json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"n": name}}).encode()


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_mtime_ns


@pytest.mark.parametrize("driver", ["fsspec", "obstore"])
def test_publish_plan_puts_deletes_and_leaves_unchanged_objects_alone(
    tmp_path: Path, driver: str
) -> None:
    source = tmp_path / "ws.zarr"
    target = tmp_path / "target.zarr"
    _put_files(
        source,
        {
            "v/c/0": b"new0",
            "t/c/0": b"same",
            "t/c/1": b"newt",
            "zarr.json": _group_doc("workspace"),
        },
    )
    _write_zarr_json(source / "v" / "zarr.json", 10)
    _put_files(
        target,
        {
            "v/c/0": b"old0",
            "v/c/1": b"old1",
            "t/c/0": b"same",
            "t/c/1": b"oldt",
            "w/c/0": b"untouched",
            "zarr.json": _group_doc("target"),
        },
    )
    _write_zarr_json(target / "v" / "zarr.json", 10)
    kept = {key: _stamp(target / key) for key in ("t/c/0", "v/zarr.json", "w/c/0")}
    session = make_test_session(tmp_path, product="target.zarr", driver=driver)  # type: ignore[arg-type]
    src, dst = StorageUri.from_local_path(source), StorageUri.from_local_path(target)

    plan = session.plan_promotion(
        src, dst, written_units={"v/c/0", "v/c/1", "v/c/2"}, absent_units={"v/c/1", "v/c/2"}
    )
    result = session.publish_plan(plan, src, dst)

    assert [(o.key, o.action.value, o.reason) for o in plan.objects] == [
        ("v/c/0", "put", "written-unit"),
        ("v/c/1", "delete", "deleted-unit"),
        ("v/c/2", "delete", "deleted-unit"),
        ("t/c/0", "unchanged", "same-bytes"),
        ("t/c/1", "put", "unclaimed"),
        ("v/zarr.json", "unchanged", "metadata"),
        ("zarr.json", "put", "metadata"),
    ]
    assert (target / "v/c/0").read_bytes() == b"new0"
    assert not (target / "v/c/1").exists()
    assert not (target / "v/c/2").exists()  # deleting what the target lacks is a no-op
    assert (target / "t/c/1").read_bytes() == b"newt"
    assert (target / "zarr.json").read_bytes() == _group_doc("workspace")
    assert {key: _stamp(target / key) for key in kept} == kept
    assert (target / "w/c/0").read_bytes() == b"untouched"
    assert (result.files_written, result.files_deleted) == (3, 1)
    assert result.bytes_written == 4 + 4 + len(_group_doc("workspace"))


def test_publish_plan_driver_without_single_object_delete_refuses_before_any_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ws.zarr"
    _put_files(source, {"v/c/0": b"new0"})
    remote_root = tmp_path / "remote"
    _put_files(remote_root, {"v/c/0": b"old0", "v/c/1": b"old1"})

    class _GlobRmFs(_LocalMirrorFs):
        """Offers only the protocol's ``rm``, which may expand a key as a pattern."""

        def read_bytes(self, path: Any) -> bytes:
            return self._local_path(path).read_bytes()

        def rm(self, path: Any, recursive: bool = False) -> None:
            raise AssertionError("rm must not be used for a promotion delete")

    fake_fs = _GlobRmFs(remote_root, "bucket/prefix")
    monkeypatch.setattr("firecube.core.storage.session.create_filesystem", lambda binding: fake_fs)
    storage_config = StorageConfig(storage_type="s3", storage_driver="fsspec")
    storage_config.bucket = "bucket"  # type: ignore[attr-defined]
    session = _session("s3://bucket/prefix", storage_config=storage_config)
    src, dst = StorageUri.from_local_path(source), StorageUri.parse("s3://bucket/prefix")
    plan = session.plan_promotion(
        src, dst, written_units={"v/c/0", "v/c/1"}, absent_units={"v/c/1"}
    )

    with pytest.raises(StorageError, match="no single-object delete"):
        session.publish_plan(plan, src, dst)

    assert (remote_root / "v/c/0").read_bytes() == b"old0"
    assert (remote_root / "v/c/1").read_bytes() == b"old1"


def test_publish_plan_failure_reports_k_of_n_and_stops_before_metadata(tmp_path: Path) -> None:
    source = tmp_path / "ws.zarr"
    target = tmp_path / "target.zarr"
    _put_files(source, {"a/c/0": b"new0", "a/c/1": b"new1"})
    _write_zarr_json(source / "a" / "zarr.json", 20)
    _put_files(target, {"a/c/2": b"old2"})
    _write_zarr_json(target / "a" / "zarr.json", 10)
    (target / "a" / "c" / "1").mkdir()  # the put of a/c/1 cannot replace a directory
    session = _session(str(target))
    src, dst = StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    plan = session.plan_promotion(
        src, dst, written_units={"a/c/0", "a/c/1", "a/c/2"}, absent_units={"a/c/2"}
    )
    assert [(o.key, o.action.value) for o in plan.objects] == [
        ("a/c/0", "put"),
        ("a/c/1", "put"),
        ("a/c/2", "delete"),
        ("a/zarr.json", "put"),
    ]

    with pytest.raises(StorageError) as excinfo:
        session.publish_plan(plan, src, dst, parallel_workers=1)

    message = str(excinfo.value)
    assert "failed after 1 of 4 object operation(s)" in message
    assert "not done (3): put a/c/1, delete a/c/2, put a/zarr.json" in message
    assert (target / "a/c/0").read_bytes() == b"new0"
    assert (target / "a/c/2").read_bytes() == b"old2"  # nothing starts after the failure
    assert json.loads((target / "a/zarr.json").read_text(encoding="utf-8"))["shape"][0] == 10


def test_publish_plan_metadata_goes_last_through_publish_metadata(tmp_path: Path) -> None:
    source = tmp_path / "ws.zarr"
    target = tmp_path / "target.zarr"
    data = {f"a/c/{i}": f"new{i}".encode() for i in range(5)}
    _put_files(source, {**data, "zarr.json": _group_doc("ws")})
    _write_zarr_json(source / "a" / "zarr.json", 20)
    _put_files(target, {"a/c/9": b"old9"})
    session = _session(str(target))
    src, dst = StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    plan = session.plan_promotion(src, dst, written_units={*data, "a/c/9"}, absent_units={"a/c/9"})
    calls: list[tuple[str, bytes, bool]] = []

    def _publish_metadata(key: str, payload: bytes) -> None:
        data_done = not (target / "a/c/9").exists() and all(
            (target / key_).is_file() and (target / key_).read_bytes() == value
            for key_, value in data.items()
        )
        calls.append((key, payload, data_done))

    result = session.publish_plan(
        plan, src, dst, parallel_workers=4, publish_metadata=_publish_metadata
    )

    assert calls == [
        ("a/zarr.json", (source / "a/zarr.json").read_bytes(), True),
        ("zarr.json", _group_doc("ws"), True),
    ]
    assert not (target / "zarr.json").exists()  # the driver never wrote metadata itself
    assert not (target / "a/zarr.json").exists()
    assert (result.files_written, result.files_deleted) == (7, 1)


@pytest.mark.parametrize(
    "bad",
    [
        PlannedObject("../other/a/c/0", PromotionAction.DELETE, 0, "deleted-unit"),
        PlannedObject("a//c/0", PromotionAction.DELETE, 0, "deleted-unit"),
        PlannedObject("a/c/0/", PromotionAction.DELETE, 0, "deleted-unit"),
        PlannedObject("a/./c/1", PromotionAction.DELETE, 0, "deleted-unit"),
        PlannedObject("a\\c\\1", PromotionAction.DELETE, 0, "deleted-unit"),
        PlannedObject("a/c/0", PromotionAction.DELETE, 0, "deleted-unit"),  # listed twice
        PlannedObject("a/zarr.json", PromotionAction.DELETE, 0, "deleted-unit"),
        PlannedObject("a/c/7", PromotionAction.PUT, 3, "written-unit"),  # not in the workspace
    ],
    ids=[
        "dotdot",
        "empty-segment",
        "trailing-slash",
        "dot",
        "backslash",
        "duplicate",
        "delete-metadata",
        "put-missing",
    ],
)
def test_publish_plan_refuses_unsafe_plans_before_any_operation(
    tmp_path: Path, bad: PlannedObject
) -> None:
    source = tmp_path / "ws.zarr"
    target = tmp_path / "target.zarr"
    victim = tmp_path / "other" / "a" / "c" / "0"
    _put_files(source, {"a/c/0": b"new0", "a/zarr.json": b"{}"})
    _put_files(target, {"a/c/0": b"old0", "a/c/1": b"old1", "a/zarr.json": b"{}"})
    _put_files(tmp_path, {"other/a/c/0": b"victim"})
    plan = PromotionPlan((PlannedObject("a/c/0", PromotionAction.PUT, 4, "written-unit"), bad))

    with pytest.raises(StorageError, match="promotion plan refused"):
        _session(str(target)).publish_plan(
            plan, StorageUri.from_local_path(source), StorageUri.from_local_path(target)
        )

    assert victim.read_bytes() == b"victim"
    assert (target / "a/c/0").read_bytes() == b"old0"
    assert (target / "a/c/1").read_bytes() == b"old1"
    assert (target / "a/zarr.json").read_bytes() == b"{}"


# --- promotion publication against real stores (moto S3, local) -------------

_S3_REGION = "us-east-1"
_S3_KEY = "testing"


@pytest.fixture(scope="module")
def _moto_endpoint() -> Iterator[str]:
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.stop()


class _Target:
    """Read and seed one product's objects, on local disk or in a moto bucket."""

    def __init__(self, tmp_path: Path, kind: str, endpoint: str | None) -> None:
        self.kind = kind
        self.product = "prod.zarr"
        if kind == "local":
            self.root = tmp_path / self.product
            self.session = make_test_session(tmp_path, product=self.product)
            self.uri = StorageUri.from_local_path(self.root)
            return
        driver = kind.removeprefix("s3-")
        self.bucket = f"q2-{uuid.uuid4().hex[:12]}"
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=_S3_KEY,
            aws_secret_access_key=_S3_KEY,
            region_name=_S3_REGION,
        )
        self.client.create_bucket(Bucket=self.bucket)
        self.uri = StorageUri(protocol="s3", authority=self.bucket, path=f"/{self.product}")
        binding = StorageBinding(
            identity=ProductIdentity.from_uri(self.uri, "zarr", product_name="q2"),
            driver=StorageDriverConfig(
                driver=driver,  # type: ignore[arg-type]
                endpoint_url=endpoint,
                credentials=Credentials(access_key=_S3_KEY, secret_key=_S3_KEY),
                region=_S3_REGION,
                path_style=True,
            ),
        )
        self.session = StorageSession(binding)

    def put(self, objects: dict[str, bytes]) -> None:
        if self.kind == "local":
            _put_files(self.root, objects)
            return
        for key, payload in objects.items():
            self.client.put_object(Bucket=self.bucket, Key=f"{self.product}/{key}", Body=payload)

    def get(self, key: str) -> bytes | None:
        if self.kind == "local":
            path = self.root / key
            return path.read_bytes() if path.is_file() else None
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=f"{self.product}/{key}")["Body"]
        except self.client.exceptions.NoSuchKey:
            return None
        return body.read()


@pytest.fixture
def _target(request: pytest.FixtureRequest, tmp_path: Path) -> _Target:
    kind = request.param
    endpoint = None if kind == "local" else request.getfixturevalue("_moto_endpoint")
    return _Target(tmp_path, kind, endpoint)


_TARGET_KINDS = [
    "local",
    pytest.param("s3-fsspec", marks=pytest.mark.s3),
    pytest.param("s3-obstore", marks=pytest.mark.s3),
]


@pytest.mark.parametrize("_target", _TARGET_KINDS, indirect=True)
def test_publish_plan_deletes_exactly_the_literal_object_of_a_glob_like_key(
    tmp_path: Path, _target: _Target
) -> None:
    source = tmp_path / "ws.zarr"
    source.mkdir()
    kept = {"victim/c/0": b"victim", "vx/c/0": b"vx", "v/c/0": b"v", "wx/c/0": b"wx"}
    _target.put({"v*/c/0": b"fill-me", **kept})
    src = StorageUri.from_local_path(source)
    units = {"v*/c/0", "w?/c/0"}  # w?/c/0 is absent from the target as well
    plan = _target.session.plan_promotion(src, _target.uri, written_units=units, absent_units=units)

    result = _target.session.publish_plan(plan, src, _target.uri)

    assert [(o.key, o.action.value) for o in plan.objects] == [
        ("v*/c/0", "delete"),
        ("w?/c/0", "delete"),
    ]
    assert _target.get("v*/c/0") is None
    assert {key: _target.get(key) for key in kept} == kept
    # A missing object is a no-op; only a local store reports it missing.
    assert result.files_deleted == (1 if _target.kind == "local" else 2)


@pytest.mark.s3
@pytest.mark.parametrize("_target", ["s3-fsspec"], indirect=True)
def test_publish_plan_refused_delete_fails_the_promotion_before_metadata(
    tmp_path: Path, _target: _Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiobotocore.client import AioBaseClient
    from botocore.exceptions import ClientError

    source = tmp_path / "ws.zarr"
    _put_files(source, {"a/c/0": b"new0"})
    _write_zarr_json(source / "a" / "zarr.json", 20)
    old_doc = json.dumps({"shape": [10, 50, 50]}).encode()
    _target.put({"a/c/0": b"old0", "a/c/1": b"old1", "a/zarr.json": old_doc})
    src = StorageUri.from_local_path(source)
    plan = _target.session.plan_promotion(
        src, _target.uri, written_units={"a/c/0", "a/c/1"}, absent_units={"a/c/1"}
    )
    real_call = AioBaseClient._make_api_call

    async def _deny_deletes(self: Any, operation_name: str, api_params: dict[str, Any]) -> Any:
        if operation_name == "DeleteObjects":  # S3 reports per-key errors in a 200 response
            keys = [entry["Key"] for entry in api_params["Delete"]["Objects"]]
            return {
                "Deleted": [],
                "Errors": [{"Key": k, "Code": "AccessDenied", "Message": "Denied"} for k in keys],
            }
        if operation_name == "DeleteObject":
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Denied"}}, operation_name
            )
        return await real_call(self, operation_name, api_params)

    monkeypatch.setattr(AioBaseClient, "_make_api_call", _deny_deletes)

    error: StorageError | None = None
    deleted = None
    try:
        deleted = _target.session.publish_plan(
            plan, src, _target.uri, parallel_workers=1
        ).files_deleted
    except StorageError as exc:
        error = exc

    assert error is not None, f"a refused delete was reported done (files_deleted={deleted})"
    assert "delete a/c/1" in str(error)
    assert _target.get("a/c/1") == b"old1"
    assert _target.get("a/zarr.json") == old_doc  # no metadata after a failed data operation


def _array_doc(shape: list[int], attrs: dict[str, Any]) -> bytes:
    return json.dumps(
        {"zarr_format": 3, "node_type": "array", "shape": shape, "attributes": attrs}
    ).encode()


@pytest.mark.parametrize(
    ("workspace_doc", "target_doc", "action"),
    [
        (_array_doc([2, 5], {"v": "new"}), _array_doc([2, 3], {"v": "old"}), "put"),
        (_array_doc([2, 5], {"v": "new"}), _array_doc([2, 5], {"v": "old"}), "put"),
        (_array_doc([4, 5], {"v": "x"}), _array_doc([2, 5], {"v": "x"}), "put"),
        (_array_doc([2, 5], {"v": "x"}), _array_doc([2, 5], {"v": "x"}), "unchanged"),
        (_array_doc([2, 5], {"v": "x"}), _array_doc([4, 5], {"v": "x"}), "unchanged"),
        (_group_doc("ws"), _group_doc("target"), "put"),
        (_group_doc("ws"), _group_doc("ws"), "unchanged"),
    ],
    ids=[
        "wider-new-attrs",
        "same-shape-new-attrs",
        "longer-workspace",
        "byte-equal",
        "longer-target-same-rest",
        "group-new-attrs",
        "group-byte-equal",
    ],
)
def test_plan_promotion_metadata_is_unchanged_only_when_the_target_supersedes_it(
    tmp_path: Path, workspace_doc: bytes, target_doc: bytes, action: str
) -> None:
    source = tmp_path / "ws.zarr"
    target = tmp_path / "target.zarr"
    _put_files(source, {"a/zarr.json": workspace_doc, "a/c/0/0": b"new"})
    _put_files(target, {"a/zarr.json": target_doc})
    session = _session(str(target))
    src, dst = StorageUri.from_local_path(source), StorageUri.from_local_path(target)

    plan = session.plan_promotion(src, dst, written_units={"a/c/0/0"}, absent_units=set())
    session.publish_plan(plan, src, dst)

    assert [(o.key, o.action.value) for o in plan.objects] == [
        ("a/c/0/0", "put"),
        ("a/zarr.json", action),
    ]
    expected = workspace_doc if action == "put" else target_doc
    assert (target / "a/zarr.json").read_bytes() == expected


def _zarr_array_doc(shape: list[int], attrs: dict[str, Any], **changes: Any) -> bytes:
    doc: dict[str, Any] = {
        "shape": shape,
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 5]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": "NaN",
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": attrs,
        "dimension_names": ["t", "x"],
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    doc.update(changes)
    return json.dumps(doc, indent=2).encode()


@pytest.mark.parametrize("via_callable", [False, True], ids=["driver", "publish_metadata"])
@pytest.mark.parametrize("when", ["longer-at-plan", "grew-before-publish", "grew-during-data"])
def test_publish_plan_never_shrinks_a_longer_target_and_publishes_workspace_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, when: str, via_callable: bool
) -> None:
    source = tmp_path / "ws.zarr"
    target = tmp_path / "target.zarr"
    workspace_doc = _zarr_array_doc([2, 5], {"run": "A"})
    grown_doc = _zarr_array_doc([4, 5], {"run": "B"})
    _put_files(source, {"a/zarr.json": workspace_doc, "a/c/0/0": b"A0"})
    plan_time_doc = (
        grown_doc if when == "longer-at-plan" else _zarr_array_doc([2, 5], {"run": "old"})
    )
    _put_files(target, {"a/zarr.json": plan_time_doc})
    session = _session(str(target))
    src, dst = StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    plan = session.plan_promotion(src, dst, written_units={"a/c/0/0"}, absent_units=set())
    assert [(o.key, o.action.value) for o in plan.objects] == [
        ("a/c/0/0", "put"),
        ("a/zarr.json", "put"),
    ]

    def _grow() -> None:  # another run grows the target and writes its tail
        _put_files(target, {"a/zarr.json": grown_doc, "a/c/3/0": b"B3"})

    if when == "grew-before-publish":
        _grow()
    if when == "grew-during-data":
        real_upload = session_module._upload_tree_upload_file

        def _upload_then_grow(*args: Any, **kwargs: Any) -> tuple[int, int]:
            outcome = real_upload(*args, **kwargs)
            if not (target / "a/c/3/0").exists():  # once, after the data put
                _grow()
            return outcome

        monkeypatch.setattr(session_module, "_upload_tree_upload_file", _upload_then_grow)
    published: dict[str, bytes] = {}

    session.publish_plan(
        plan,
        src,
        dst,
        publish_metadata=published.__setitem__ if via_callable else None,
    )

    document = published["a/zarr.json"] if via_callable else (target / "a/zarr.json").read_bytes()
    expected = json.loads(workspace_doc)
    expected["shape"] = [4, 5]
    assert json.loads(document) == expected
    assert (target / "a/c/0/0").read_bytes() == b"A0"
    if when != "longer-at-plan":
        assert (target / "a/c/3/0").read_bytes() == b"B3"


@pytest.mark.parametrize(
    "target_doc",
    [
        _zarr_array_doc([4, 5], {"run": "B"}, data_type="float64"),
        _zarr_array_doc(
            [4, 5],
            {"run": "B"},
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": [2, 5]}},
        ),
        _zarr_array_doc([4, 5], {"run": "B"}, fill_value=0.0),
        _zarr_array_doc([4, 5], {"run": "B"}, codecs=[{"name": "bytes"}]),
        _zarr_array_doc([4, 5], {"run": "B"}, dimension_names=["time", "x"]),
        _zarr_array_doc([4, 5], {"run": "B"}, node_type="group"),
        _zarr_array_doc([4, 6], {"run": "B"}),
        _zarr_array_doc([4, 5, 1], {"run": "B"}),
    ],
    ids=[
        "data_type",
        "chunk_grid",
        "fill_value",
        "codecs",
        "dimension_names",
        "node_type",
        "other-dim",
        "rank",
    ],
)
def test_publish_plan_refuses_a_longer_target_that_differs_in_structure(
    tmp_path: Path, target_doc: bytes
) -> None:
    source = tmp_path / "ws.zarr"
    target = tmp_path / "target.zarr"
    _put_files(source, {"a/zarr.json": _zarr_array_doc([2, 5], {"run": "A"}), "a/c/0/0": b"A0"})
    _put_files(target, {"a/zarr.json": target_doc, "a/c/0/0": b"B0"})
    session = _session(str(target))
    src, dst = StorageUri.from_local_path(source), StorageUri.from_local_path(target)
    plan = session.plan_promotion(src, dst, written_units={"a/c/0/0"}, absent_units=set())

    error: StorageError | None = None
    try:
        session.publish_plan(plan, src, dst)
    except StorageError as exc:
        error = exc

    assert error is not None, "a structurally different, longer target was replaced"
    assert "a/zarr.json" in str(error)
    assert (target / "a/zarr.json").read_bytes() == target_doc
    assert (target / "a/c/0/0").read_bytes() == b"B0"  # refused before any operation
