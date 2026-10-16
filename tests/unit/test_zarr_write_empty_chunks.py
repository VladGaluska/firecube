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

import ast
import contextlib
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from click.testing import CliRunner
from zarr.core.sync import sync
from zarr.storage import LocalStore, MemoryStore

from firecube.cli.main import cli
from firecube.core.zarr.guard.guarded_write import (
    WriteFootprint,
    guarded_write,
    hold_array,
    unit_store_key,
)
from firecube.ingestor.runtime.zarr.strategies.indexed_region import IndexedRegionStrategy
from firecube.ingestor.runtime.zarr.write_context import ZarrWriteContext
from firecube.ingestor.templates.config import ZarrTemplateConfig

pytestmark = pytest.mark.unit


def test_default_is_false() -> None:
    assert ZarrTemplateConfig().zarr_write_empty_chunks is False


def test_option_flows_through_typed_config() -> None:
    assert (
        ZarrTemplateConfig.from_options({"zarr_write_empty_chunks": "true"}).zarr_write_empty_chunks
        is True
    )


def test_show_options_lists_the_field() -> None:
    result = CliRunner().invoke(
        cli, ["ingest", "direct_zarr_capable_test_plugin", "--show-options"]
    )

    assert result.exit_code == 0, result.output
    assert "--option zarr_write_empty_chunks" in result.output


def test_direct_zarr_dispatch_leaves_process_config_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, bool]] = []

    @contextlib.contextmanager
    def spy_config_set(config: dict[str, bool]):
        calls.append(config)
        yield

    monkeypatch.setattr(zarr.config, "set", spy_config_set)

    result = IndexedRegionStrategy(store_uri=str(tmp_path / "test.zarr")).write_groups(
        group_to_intents={},
        zarr_write_empty_chunks=True,
    )

    assert result["zarr_write_empty_chunks_effective"] is True
    assert [call for call in calls if "array.write_empty_chunks" in call] == []


def test_generic_write_context_applies_scoped_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, bool]] = []
    entered = False
    exited = False

    @contextlib.contextmanager
    def spy_config_set(config: dict[str, bool]):
        nonlocal entered, exited
        if "array.write_empty_chunks" in config:
            calls.append(config)
            entered = True
        try:
            yield
        finally:
            if "array.write_empty_chunks" in config:
                exited = True

    monkeypatch.setattr(zarr.config, "set", spy_config_set)

    with ZarrWriteContext(write_empty_chunks=True):
        assert entered is True

    assert {"array.write_empty_chunks": True} in calls
    assert exited is True


def test_array_write_empty_chunks_never_mutated_bare() -> None:
    root = Path("src/firecube")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not _is_array_write_empty_chunks_set_call(node):
                continue
            if not _has_with_ancestor(node, parents):
                offenders.append(f"{path}:{getattr(node, 'lineno', '?')}")
    assert offenders == []


def _is_array_write_empty_chunks_set_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "set":
        return False
    for arg in node.args:
        if isinstance(arg, ast.Dict):
            for key in arg.keys:
                if isinstance(key, ast.Constant) and key.value == "array.write_empty_chunks":
                    return True
    return False


def _has_with_ancestor(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.With):
            return True
        current = parents.get(current)
    return False


# guarded_write: write_empty_chunks per array handle, and the units a write leaves absent.

CHUNKED = {"chunks": (2,), "shards": None}
SHARDED = {"chunks": (2,), "shards": (4,)}


def _vector(store: Any, layout: dict[str, Any], *, ones: slice | None = None) -> Any:
    """An 8-cell float array with fill 0, optionally set to 1 over ``ones``, reopened."""
    created = zarr.create_array(
        store, name="data/v", shape=(8,), dtype="f4", fill_value=0.0, zarr_format=3, **layout
    )
    if ones is not None:
        created.with_config({"write_empty_chunks": False})[ones] = 1.0
    return zarr.open_array(store, path="data/v", mode="r+")


def _stored(arr: Any, unit: tuple[int, ...]) -> bool:
    return bool(sync(arr.store.exists(unit_store_key(arr, unit))))


def _fill(arr: Any, cells: slice, *, write_empty_chunks: bool | None = False) -> Any:
    return guarded_write(
        arr,
        WriteFootprint("data/v", ((cells.start, cells.stop),)),
        cells,
        np.zeros(cells.stop - cells.start, dtype="f4"),
        store="test",
        write_empty_chunks=write_empty_chunks,
        track_units=True,
    )


def _fill_together(keep: Any, drop: Any) -> dict[str, Any]:
    """All-fill writes to cells 0:2 of both arrays from two threads released at once."""
    barrier = threading.Barrier(2, timeout=10)
    results: dict[str, Any] = {}
    errors: list[BaseException] = []

    def write(name: str, arr: Any, setting: bool) -> None:
        try:
            barrier.wait()
            results[name] = _fill(arr, slice(0, 2), write_empty_chunks=setting)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=write, args=("keep", keep, True)),
        threading.Thread(target=write, args=("drop", drop, False)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    return results


def test_two_threads_with_different_settings_each_get_their_own() -> None:
    """Each write uses its own setting: not its handle's, not the process-global one."""
    with zarr.config.set({"array.write_empty_chunks": True}):
        global_before = zarr.config.get("array.write_empty_chunks")
        for _ in range(25):
            with zarr.config.set({"array.write_empty_chunks": False}):
                keep = _vector(MemoryStore(), CHUNKED)
            drop = _vector(MemoryStore(), CHUNKED)
            assert (keep.config.write_empty_chunks, drop.config.write_empty_chunks) == (False, True)

            results = _fill_together(keep, drop)

            assert _stored(keep, (0,)) is True
            assert _stored(drop, (0,)) is False
            assert (results["keep"].tracked, results["keep"].absent) == (False, ())
            assert (results["drop"].tracked, results["drop"].absent) == (True, ((0,),))
            assert (keep.config.write_empty_chunks, drop.config.write_empty_chunks) == (False, True)
        assert zarr.config.get("array.write_empty_chunks") is global_before


@pytest.mark.parametrize("store_kind", ["memory", "local"])
@pytest.mark.parametrize(
    ("layout", "ones", "cells", "absent", "deleted"),
    [
        pytest.param(CHUNKED, slice(0, 8), slice(0, 2), ((0,),), ((0,),), id="chunk-whole-to-fill"),
        pytest.param(CHUNKED, slice(0, 8), slice(0, 1), (), (), id="chunk-part-to-fill"),
        pytest.param(CHUNKED, None, slice(0, 2), ((0,),), (), id="chunk-never-stored"),
        pytest.param(SHARDED, slice(0, 8), slice(0, 4), ((0,),), ((0,),), id="shard-whole-to-fill"),
        pytest.param(SHARDED, slice(0, 8), slice(0, 2), (), (), id="shard-one-inner-of-two"),
        pytest.param(SHARDED, slice(0, 2), slice(0, 2), ((0,),), ((0,),), id="shard-last-inner"),
        pytest.param(SHARDED, slice(0, 2), slice(0, 1), (), (), id="shard-part-of-last-inner"),
        pytest.param(SHARDED, None, slice(0, 4), ((0,),), (), id="shard-never-stored"),
    ],
)
def test_guarded_write_reports_absent_and_deleted_units(
    tmp_path: Path,
    store_kind: str,
    layout: dict[str, Any],
    ones: slice | None,
    cells: slice,
    absent: tuple[tuple[int, ...], ...],
    deleted: tuple[tuple[int, ...], ...],
) -> None:
    store = MemoryStore() if store_kind == "memory" else LocalStore(tmp_path)
    arr = _vector(store, layout, ones=ones)

    result = _fill(arr, cells)

    assert result.tracked is True
    assert result.units == ((0,),)
    assert result.absent == absent
    assert result.deleted == deleted
    assert _stored(arr, (0,)) is (absent == ())
    expected = np.zeros(8, dtype="f4")
    if ones is not None:
        expected[ones] = 1.0
    expected[cells] = 0.0
    np.testing.assert_array_equal(arr[...], expected)


@pytest.mark.parametrize("layout", [CHUNKED, SHARDED], ids=["chunk", "shard"])
def test_guarded_write_keeping_empty_chunks_stores_every_touched_unit(
    layout: dict[str, Any],
) -> None:
    arr = _vector(MemoryStore(), layout, ones=slice(0, 8))

    result = _fill(arr, slice(0, 4), write_empty_chunks=True)

    assert (result.tracked, result.absent, result.deleted) == (False, (), ())
    assert all(_stored(arr, unit) for unit in result.units)


@pytest.mark.parametrize(
    "selection",
    [slice(0, 3), slice(None), 2, -1, np.int64(5), (Ellipsis,), Ellipsis],
    ids=["slice", "all", "int", "negative", "numpy-int", "ellipsis-tuple", "ellipsis"],
)
def test_guarded_write_refuses_a_selection_outside_its_footprint(selection: Any) -> None:
    arr = _vector(MemoryStore(), CHUNKED, ones=slice(0, 8))

    with pytest.raises(ValueError, match="outside the footprint"):
        guarded_write(arr, WriteFootprint("data/v", ((0, 2),)), selection, 0.0, store="test")

    np.testing.assert_array_equal(arr[...], np.ones(8, dtype="f4"))


@pytest.mark.parametrize(
    ("selection", "named"),
    [
        (np.array([6]), "ndarray"),
        (np.array([True, False, False, False, False, False, False, False]), "ndarray"),
        ([6], "list"),
        ([True, False, False, False, False, False, False, False], "list"),
        ((np.array([6]),), "ndarray"),
        ((slice(0, 2), [6]), "list"),
        (None, "NoneType"),
        (True, "bool"),
        (np.bool_(True), "bool"),
        ((Ellipsis, Ellipsis), "Ellipsis"),
    ],
    ids=[
        "int-array",
        "bool-mask",
        "list",
        "bool-list",
        "int-array-in-tuple",
        "list-in-tuple",
        "newaxis",
        "bool-scalar",
        "numpy-bool-scalar",
        "two-ellipses",
    ],
)
def test_guarded_write_refuses_other_than_basic_selections_before_locking(
    selection: Any, named: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    arr = _vector(MemoryStore(), CHUNKED, ones=slice(0, 8))

    def no_lock(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a unit lock was taken for an unsupported selection")

    monkeypatch.setattr("firecube.core.zarr.guard.guarded_write._UNIT_LOCKS.hold", no_lock)

    with pytest.raises(ValueError, match="int, slice and Ellipsis selections only") as raised:
        guarded_write(arr, WriteFootprint("data/v", ((0, 2),)), selection, 0.0, store="test")

    assert named in str(raised.value)
    np.testing.assert_array_equal(arr[...], np.ones(8, dtype="f4"))


@pytest.mark.parametrize(
    "selection",
    [slice(0, 2), 1, np.int64(1), (slice(0, 2),), (Ellipsis,), Ellipsis, (1, Ellipsis)],
    ids=["slice", "int", "numpy-int", "tuple", "ellipsis-tuple", "ellipsis", "int-ellipsis"],
)
def test_guarded_write_still_writes_basic_selections(selection: Any) -> None:
    arr = _vector(MemoryStore(), CHUNKED)
    expected = np.zeros(8, dtype="f4")
    expected[selection] = 1.0

    guarded_write(arr, WriteFootprint("data/v", ((0, 8),)), selection, 1.0, store="test")

    np.testing.assert_array_equal(arr[...], expected)


def test_guarded_write_refuses_a_footprint_of_another_array() -> None:
    arr = _vector(MemoryStore(), CHUNKED)

    with pytest.raises(ValueError, match="footprint names 'data/w'"):
        guarded_write(arr, WriteFootprint("data/w", ((0, 2),)), slice(0, 2), 1.0, store="test")


def test_write_footprint_spells_the_array_path_one_way() -> None:
    assert WriteFootprint("/data//v/", ((0, 2),)) == WriteFootprint("data/v", ((0, 2),))


def test_guarded_write_inside_a_hold_of_its_array_reuses_the_hold() -> None:
    arr = _vector(MemoryStore(), CHUNKED)
    done = threading.Event()

    def transaction() -> None:
        with hold_array("test", "data/v") as hold, hold_array("test", "/data/v"):
            guarded_write(
                arr, WriteFootprint("data/v", ((0, 8),)), Ellipsis, 1.0, store="test", hold=hold
            )
            guarded_write(arr, WriteFootprint("data/v", ((0, 2),)), slice(0, 2), 2.0, store="test")
        done.set()

    worker = threading.Thread(target=transaction, daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert done.is_set(), "a write inside hold_array waited on its own hold"
    np.testing.assert_array_equal(arr[...], [2, 2, 1, 1, 1, 1, 1, 1])


def test_guarded_write_of_a_scalar_array_inside_its_hold_reuses_the_hold() -> None:
    """A 0-d array's only unit key is its whole-array key: taking it again would never return."""
    store = MemoryStore()
    zarr.create_array(store, name="data/s", shape=(), dtype="f4", fill_value=0.0, zarr_format=3)
    scalar = zarr.open_array(store, path="data/s", mode="r+")
    done = threading.Event()

    def transaction() -> None:
        with hold_array("test", "data/s"):
            guarded_write(scalar, WriteFootprint("data/s", ()), (), 3.0, store="test")
        done.set()

    worker = threading.Thread(target=transaction, daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert done.is_set(), "a write inside hold_array waited on its own hold"
    np.testing.assert_array_equal(scalar[()], 3.0)


def test_guarded_write_refuses_a_hold_of_another_array() -> None:
    arr = _vector(MemoryStore(), CHUNKED)

    with hold_array("test", "data/w") as other, pytest.raises(ValueError, match="not held"):
        guarded_write(
            arr, WriteFootprint("data/v", ((0, 2),)), slice(0, 2), 1.0, store="test", hold=other
        )


@pytest.mark.parametrize(
    ("layout", "footprint", "selection", "absent"),
    [
        pytest.param(
            CHUNKED, (0, 13), slice(0, 13, 12), ((1,), (2,), (3,), (4,), (5,)), id="chunk-stepped"
        ),
        pytest.param(SHARDED, (0, 13), slice(0, 13, 12), ((1,), (2,)), id="shard-stepped"),
        pytest.param(CHUNKED, (0, 8), slice(0, 2), ((1,), (2,), (3,)), id="chunk-wider-footprint"),
        pytest.param(SHARDED, (0, 16), slice(0, 4), ((1,), (2,), (3,)), id="shard-wider-footprint"),
    ],
)
def test_keeping_empty_chunks_still_reports_units_the_write_did_not_reach(
    layout: dict[str, Any],
    footprint: tuple[int, int],
    selection: slice,
    absent: tuple[tuple[int, ...], ...],
) -> None:
    """Kept empty chunks rule out deletion, not absence: unselected units of the box stay absent."""
    zarr.create_array(
        store := MemoryStore(),
        name="data/v",
        shape=(16,),
        dtype="f4",
        fill_value=0.0,
        zarr_format=3,
        **layout,
    )
    arr = zarr.open_array(store, path="data/v", mode="r+")
    cells = len(range(*selection.indices(16)))

    result = guarded_write(
        arr,
        WriteFootprint("data/v", (footprint,)),
        selection,
        np.ones(cells, dtype="f4"),
        store="test",
        write_empty_chunks=True,
        track_units=True,
    )

    assert result.tracked is True
    assert result.absent == absent
    assert result.deleted == ()
    assert result.absent == tuple(unit for unit in result.units if not _stored(arr, unit))
