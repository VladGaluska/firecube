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

"""Maintenance ops acquire write claims before mutating product state."""

from __future__ import annotations

import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr
import zarr
from click.testing import CliRunner

from firecube.cli._errors import render
from firecube.cli.main import cli
from firecube.core.controlplane import ChunkInfo, ChunkManager
from firecube.core.controlplane.claims import FilesystemClaimService
from firecube.core.controlplane.types import DeletionPlan, WriteDomain
from firecube.core.errors import ManifestError
from firecube.core.storage.session import StorageSession
from firecube.core.storage.uri import StorageUri
from firecube.core.tensogram import restore as tgm_restore
from firecube.core.tensogram.converter import zarr_to_tgm
from firecube.core.zarr import zip_archive
from firecube.core.zarr.scrub import run_scrub
from firecube.ingestor.runtime.archive import ArchiveClaimError, create_archive, restore_archive
from tests.helpers.cli_hints import run_hint
from tests.helpers.storage import make_test_binding, make_test_session

pytestmark = pytest.mark.integration


def _local_env(tmp_path: Path) -> dict[str, str]:
    return {"FIRECUBE_STORAGE_TYPE": "local", "FIRECUBE_TARGET_PATH": str(tmp_path)}


def _claim_domain(product: str, group: str = "F024") -> WriteDomain:
    return WriteDomain(product=product, category="zarr_append", name=group)


def _claiming_manager(tmp_path: Path, product: str = "product.zarr") -> ChunkManager:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    binding = make_test_binding(tmp_path, product=product)
    return ChunkManager(binding=binding, workspace=workspace)


def _sample_chunk(product: str) -> ChunkInfo:
    return ChunkInfo(
        key="F024/FWI/c/0/0/0",
        product=product,
        chunk_type="chunk",
        size=1,
        timestamp=time.time(),
        manifest_path="",
    )


def _sample_span(product: str) -> ChunkInfo:
    return ChunkInfo(
        key="span-run-1-batch-1-F024",
        product=product,
        chunk_type="span",
        size=0,
        timestamp=time.time(),
        manifest_path="",
        meta={"group": "F024", "run_id": "run-1", "batch_id": "batch-1"},
        record={
            "span": {
                "arrays": ["F024/FWI"],
                "time_index_ranges": [[0, 0]],
                "aligned": True,
            }
        },
    )


def _make_archive(source_zarr: Path, archive_path: Path, group: str | None = "F024") -> None:
    dataset = xr.Dataset(
        {
            "FWI": (
                ["timestamp", "lat", "lon"],
                np.random.default_rng(7).random((2, 2, 2)).astype("float32"),
            )
        },
        coords={
            "timestamp": np.arange(2),
            "lat": np.linspace(-1.0, 1.0, 2, dtype="float32"),
            "lon": np.linspace(0.0, 1.0, 2, dtype="float32"),
        },
    )
    dataset.to_zarr(source_zarr, group=group)
    zarr_to_tgm(str(source_zarr), str(archive_path), group=group)


def _assert_clear_hint_runs_without_abandon(message: str) -> None:
    run_hint(message, "firecube chunks claims clear", "--yes-i-really-mean-it")
    assert "firecube chunks runs abandon" not in message


def _assert_owner_abandon_hint_runs(message: str, root: str) -> None:
    assert f"--product-name {root} --run-id running-ingest" in message
    assert "or abandon run running-ingest (status: started)," in message
    run_hint(message, "firecube chunks claims clear", "--yes-i-really-mean-it")
    run_hint(message, "firecube chunks runs abandon", "--dry-run")


def _assert_claim_facts(
    error: ArchiveClaimError, root: Path, *, owner: str, recovery: list[str], run_id: str = ""
):
    context, domain = error.context, _claim_domain(error.product).identifier
    assert (context.product_uri, context.claim_domain, context.claim_owner) == (
        root.as_uri(), domain, owner,
    )  # fmt: skip
    assert [op.name for op in error.recovery] == recovery
    assert {op.arguments["product_name"] for op in error.recovery} == {root.as_uri()}
    assert error.recovery[0].arguments["domain_id"] == domain
    assert all(op.arguments["run_id"] == run_id for op in error.recovery[1:])


def test_execute_deletion_aborts_when_product_has_active_writer_claim(tmp_path: Path) -> None:
    product = "product.zarr"
    manager = _claiming_manager(tmp_path)
    claim = manager.acquire_claim(
        product=product,
        domain=_claim_domain(product),
        owner_id="running-ingest:F024",
    )
    try:
        plan = DeletionPlan(
            chunks=[_sample_chunk(product)],
            total_size=1,
            products_affected={product},
            manifest_files=set(),
        )

        with pytest.raises(ManifestError) as excinfo:
            manager.execute_deletion(
                plan,
                delete_storage=True,
                delete_manifest=False,
                dry_run=False,
            )

        claims = manager.list_claims(product=product)
        assert [info.domain for info in claims] == [_claim_domain(product).identifier]
        assert "ChunkManager.clear_claim" in str(excinfo.value)
        _assert_clear_hint_runs_without_abandon(render(excinfo.value))
    finally:
        claim.release()
        manager.close()


def test_execute_deletion_names_abandon_for_a_recorded_owner(tmp_path: Path) -> None:
    product, manager = "product.zarr", _claiming_manager(tmp_path)
    manager.record_run_started(
        product=product, run_id="running-ingest", output_path="", output_format="zarr",
        size=0, meta={},
    )  # fmt: skip
    claim = manager.acquire_claim(
        product=product, domain=_claim_domain(product), owner_id="running-ingest:F024"
    )
    plan = DeletionPlan([_sample_chunk(product)], 1, {product}, set())
    try:
        with pytest.raises(ManifestError) as excinfo:
            manager.execute_deletion(
                plan, delete_storage=True, delete_manifest=False, dry_run=False
            )
        assert "ChunkManager.abandon_run" in str(excinfo.value)
        _assert_owner_abandon_hint_runs(render(excinfo.value), manager.get_product_root(product))
    finally:
        claim.release()
        manager.close()


def test_delete_spans_aborts_when_product_has_active_writer_claim(tmp_path: Path) -> None:
    product = "product.zarr"
    manager = _claiming_manager(tmp_path)
    claim = manager.acquire_claim(
        product=product,
        domain=_claim_domain(product),
        owner_id="running-ingest:F024",
    )
    try:
        with pytest.raises(ManifestError) as excinfo:
            manager.delete_spans([_sample_span(product)], dry_run=False)

        claims = manager.list_claims(product=product)
        assert [info.domain for info in claims] == [_claim_domain(product).identifier]
        _assert_clear_hint_runs_without_abandon(render(excinfo.value))
    finally:
        claim.release()
        manager.close()


def test_scrub_aborts_when_product_has_active_writer_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    product = "product.zarr"
    manager = _claiming_manager(tmp_path, product=product)
    claim = manager.acquire_claim(
        product=product,
        domain=_claim_domain(product),
        owner_id="running-ingest:F024",
    )
    session = make_test_session(tmp_path, product=product)

    # T3.4 migrated scrub to validate_group_with_fs (fs-aware seam) — mock that.
    monkeypatch.setattr(
        "firecube.core.zarr.scrub.validate_group_with_fs",
        lambda *args, **kwargs: SimpleNamespace(extra_chunks=[f"{product}/F024/FWI/c/0/0/0"]),
    )
    monkeypatch.setattr(
        ChunkManager,
        "list_chunks",
        lambda self, *args, **kwargs: [_sample_chunk(product)],
    )

    try:
        with pytest.raises(ManifestError) as excinfo:
            run_scrub(session, "F024")
        _assert_clear_hint_runs_without_abandon(render(excinfo.value))
    finally:
        claim.release()
        manager.close()


def _restore_with_claimed_target(
    tmp_path: Path, check: Callable[[str], None], *, record_owner: bool = False
) -> None:
    """Run a refused restore, then `check` its output while the claim is still held."""
    runner = CliRunner()
    env = _local_env(tmp_path)
    source_zarr = tmp_path / "source.zarr"
    archive_path = tmp_path / "archive.tgm"
    restored = tmp_path / "restored.zarr"
    _make_archive(source_zarr, archive_path)

    manager = _claiming_manager(tmp_path)
    if record_owner:
        manager.record_run_started(
            product="restored.zarr", run_id="running-ingest", output_path="", output_format="zarr",
            size=0, meta={},
        )  # fmt: skip
    claim = manager.acquire_claim(
        product="restored.zarr",
        domain=_claim_domain("restored.zarr"),
        owner_id="running-ingest:F024",
    )
    try:
        result = runner.invoke(
            cli,
            [
                "archive",
                "restore",
                "--archive",
                StorageUri.from_local_path(archive_path).to_str(),
                "--target",
                StorageUri.from_local_path(restored).to_str(),
                "--storage-type",
                "local",
                "--storage-driver",
                "fsspec",
            ],
            env=env,
        )
        assert result.exit_code != 0
        check(result.output)
    finally:
        claim.release()
        manager.close()


def test_archive_restore_aborts_when_product_has_active_writer_claim(tmp_path: Path) -> None:
    _restore_with_claimed_target(tmp_path, _assert_clear_hint_runs_without_abandon)


def test_archive_restore_names_abandon_for_a_recorded_owner(tmp_path: Path) -> None:
    root = StorageUri.from_local_path(tmp_path / "restored.zarr").to_str()
    _restore_with_claimed_target(
        tmp_path, lambda out: _assert_owner_abandon_hint_runs(out, root), record_owner=True
    )


def _zip_create_with_control_plane(tmp_path: Path):
    src = tmp_path / "s.zarr"
    (src / ".firecube").mkdir(parents=True, exist_ok=True)
    (src / ".firecube" / "schema.json").write_text("{}")
    cmd = f"archive create --source {src.as_uri()} --archive {tmp_path.as_uri()}/out.zip"
    cmd += " --archive-format zip"
    return CliRunner().invoke(cli, cmd.split(), env=_local_env(tmp_path))


def test_archive_create_aborts_when_product_has_active_writer_claim(tmp_path: Path) -> None:
    manager = _claiming_manager(tmp_path, product="s.zarr")
    claim = manager.acquire_claim(product="s.zarr", domain=_claim_domain("s.zarr"), owner_id="w")
    try:
        result = _zip_create_with_control_plane(tmp_path)
        assert result.exit_code != 0
        _assert_clear_hint_runs_without_abandon(result.output)
    finally:
        claim.release()
    assert not (tmp_path / "out.zip").exists()


def test_archive_refusal_after_a_lost_race_names_clear_once(tmp_path, monkeypatch):
    manager = _claiming_manager(tmp_path, product="s.zarr")
    domain = WriteDomain(product="s.zarr", category="maintenance", name="archive_create")
    claim = manager.acquire_claim(product="s.zarr", domain=domain, owner_id="other-archive")
    held = manager.list_claims(product="s.zarr")
    monkeypatch.setattr(ChunkManager, "list_claims", Mock(side_effect=[[], held]))
    result = _zip_create_with_control_plane(tmp_path)
    claim.release()
    assert result.output.count(f"--domain {domain.identifier}") == 1, result.output


def test_archive_create_keeps_archive_error_and_releases_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(zip_archive, "zarr_to_zip", Mock(side_effect=ValueError("archive boom")))
    monkeypatch.setattr(ChunkManager, "record_maintenance_failed", Mock(side_effect=OSError("rec")))
    result = _zip_create_with_control_plane(tmp_path)
    assert result.exit_code != 0
    assert "archive boom" in result.output
    assert list((tmp_path / "s.zarr" / ".firecube" / "claims").glob("*")) == []


def _restore_state(tmp_path: Path) -> tuple[list[str], list[str]]:
    manager = _claiming_manager(tmp_path, product="restored.zarr")
    try:
        runs = manager.list_runs(product="restored.zarr")
        return (
            [c.domain for c in manager.list_claims(product="restored.zarr")],
            [r.status for r in runs if r.run_id.startswith("archive-restore-")],
        )
    finally:
        manager.close()


def _overwrite_restore(
    tmp_path: Path, kind: str, monkeypatch, on_write: Callable[[], None], *, library: bool = False
):
    """Run an overwrite restore, by the command or (`library`) by `restore_archive`; `on_write`
    runs after each archive write and after the restore function returns."""
    src, target, archive = (
        tmp_path / "source.zarr",
        tmp_path / "restored.zarr",
        tmp_path / f"a.{kind}",
    )
    if kind == "tgm":
        _make_archive(src, archive, group=None)
    else:
        zarr.open_group(str(src), mode="w").create_array("a", data=np.arange(6.0))
        with zipfile.ZipFile(archive, "w") as zf:
            for p in filter(Path.is_file, src.rglob("*")):
                zf.write(p, p.relative_to(src).as_posix())
    zarr.open_group(str(target), mode="w").create_array("old", data=np.arange(3.0))
    writer = (
        (StorageSession, "put_bytes") if kind == "zip" else (tgm_restore, "_restore_data_message")
    )
    entry = (zip_archive, "zip_to_zarr") if kind == "zip" else (tgm_restore, "tgm_to_zarr")

    def spy_on(original):
        def spy(*a, **k):
            result = original(*a, **k)
            on_write()
            return result

        return spy

    for owner, name in (writer, entry):
        monkeypatch.setattr(owner, name, spy_on(getattr(owner, name)))
    if library:
        return restore_archive(
            session=make_test_session(tmp_path, product="restored.zarr"),
            archive=archive.as_uri(),
            archive_path=str(archive),
            archive_format="zip" if kind == "zip" else "tensogram",
            target=target.as_uri(),
            overwrite=True,
        )
    args = ["archive", "restore", "--archive", archive.as_uri(), "--target", target.as_uri()]
    args += ["--overwrite", "--yes-i-really-mean-it", "--storage-type", "local"]
    return CliRunner().invoke(cli, [*args, "--storage-driver", "fsspec"], env=_local_env(tmp_path))


ROUTES = [False, pytest.param(True, marks=pytest.mark.contract)]


@pytest.mark.parametrize("lib", ROUTES, ids=["cli", "library"])
@pytest.mark.parametrize("kind", ["zip", "tgm"])
def test_overwrite_restore_holds_its_claim_and_run_while_writing(tmp_path, monkeypatch, kind, lib):
    states = []
    on_write = lambda: states.append(_restore_state(tmp_path))  # noqa: E731
    _overwrite_restore(tmp_path, kind, monkeypatch, on_write, library=lib)
    domain = WriteDomain(product="restored.zarr", category="maintenance", name="archive_restore")
    assert len(states) >= 2, states
    assert all(state == ([domain.identifier], ["started"]) for state in states), states


@pytest.mark.parametrize("kind", ["zip", "tgm"])
def test_interrupted_overwrite_restore_leaves_a_started_run(tmp_path, monkeypatch, kind):
    def interrupt() -> None:
        raise SystemExit(3)

    _overwrite_restore(tmp_path, kind, monkeypatch, interrupt)
    assert _restore_state(tmp_path)[1] == ["started"]


@pytest.mark.parametrize("lib", ROUTES, ids=["cli", "library"])
def test_restore_stops_its_claim_heartbeat(tmp_path, monkeypatch, lib):
    kwdefaults = FilesystemClaimService.__init__.__kwdefaults__
    monkeypatch.setitem(kwdefaults, "heartbeat_interval_s", 1)
    handles, acquire = [], ChunkManager.acquire_claim

    def spy(self, *a, **k):
        handles.append(acquire(self, *a, **k))
        return handles[-1]

    monkeypatch.setattr(ChunkManager, "acquire_claim", spy)
    _make_archive(tmp_path / "source.zarr", tmp_path / "a.tgm")
    args = ["archive", "restore", "--archive", (tmp_path / "a.tgm").as_uri()]
    args += ["--target", (tmp_path / "r.zarr").as_uri(), "--storage-type", "local"]
    try:
        if lib:
            restore_archive(
                session=make_test_session(tmp_path, product="r.zarr"),
                archive=(tmp_path / "a.tgm").as_uri(),
                archive_path=str(tmp_path / "a.tgm"),
                archive_format="tensogram",
                target=(tmp_path / "r.zarr").as_uri(),
            )
        else:
            result = CliRunner().invoke(
                cli, [*args, "--storage-driver", "fsspec"], env=_local_env(tmp_path)
            )
            assert result.exit_code == 0, result.output
        time.sleep(2.5)
        assert list((tmp_path / "r.zarr" / ".firecube" / "claims").glob("*")) == []
    finally:
        for handle in handles:
            handle.release()


@pytest.mark.contract
@pytest.mark.parametrize("kind", ["zip", "tgm"])
def test_restore_archive_interrupt_keeps_run_started_and_frees_claim(tmp_path, monkeypatch, kind):
    interrupt = Mock(side_effect=SystemExit(3))
    with pytest.raises(SystemExit):
        _overwrite_restore(tmp_path, kind, monkeypatch, interrupt, library=True)
    assert _restore_state(tmp_path) == ([], ["started"])


@pytest.mark.contract
def test_create_archive_refuses_an_active_writer_claim_and_records_no_run(tmp_path):
    (tmp_path / "s.zarr" / ".firecube").mkdir(parents=True)
    (tmp_path / "s.zarr" / ".firecube" / "schema.json").write_text("{}")
    out = tmp_path / "out.zip"
    session = make_test_session(tmp_path, product="s.zarr")
    manager = _claiming_manager(tmp_path, product="s.zarr")
    claim = manager.acquire_claim(product="s.zarr", domain=_claim_domain("s.zarr"), owner_id="w")
    try:
        with pytest.raises(ArchiveClaimError) as refused:
            create_archive(
                session=session, archive=out.as_uri(), archive_path=str(out), archive_format="zip"
            )
        runs = manager.list_runs(product="s.zarr")
    finally:
        claim.release()
        manager.close()
    assert (refused.value.product, refused.value.operation) == ("s.zarr", "archive_create")
    _assert_claim_facts(refused.value, tmp_path / "s.zarr", owner="w", recovery=["clear_claim"])
    assert not out.exists()
    assert [r.run_id for r in runs if r.run_id.startswith("archive-create-")] == []


@pytest.mark.contract
def test_restore_archive_refusal_carries_claim_facts_and_recovery(tmp_path):
    manager = _claiming_manager(tmp_path, product="r.zarr")
    manager.record_run_started(
        product="r.zarr", run_id="running-ingest", output_path="", output_format="zarr",
        size=0, meta={},
    )  # fmt: skip
    owner = "running-ingest:F024"
    claim = manager.acquire_claim(product="r.zarr", domain=_claim_domain("r.zarr"), owner_id=owner)
    try:
        with pytest.raises(ArchiveClaimError) as refused:
            restore_archive(
                session=make_test_session(tmp_path, product="r.zarr"),
                archive=(tmp_path / "a.zip").as_uri(),
                archive_path=str(tmp_path / "a.zip"),
                archive_format="zip",
                target=(tmp_path / "r.zarr").as_uri(),
            )
    finally:
        claim.release()
        manager.close()
    _assert_claim_facts(
        refused.value, tmp_path / "r.zarr", owner=owner, run_id="running-ingest",
        recovery=["clear_claim", "abandon_run"],
    )  # fmt: skip


@pytest.mark.contract
def test_create_archive_of_a_plain_store_takes_no_claim_and_writes_nothing_to_it(tmp_path):
    source = tmp_path / "plain.zarr"
    zarr.open_group(str(source), mode="w").create_array("a", data=np.arange(6.0))
    out = tmp_path / "out.zip"
    session = make_test_session(tmp_path, product="plain.zarr")
    tree = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = create_archive(
        session=session, archive=out.as_uri(), archive_path=str(out), archive_format="zip"
    )
    assert result.run_id is None and out.exists() and not (source / ".firecube").exists()
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == tree
