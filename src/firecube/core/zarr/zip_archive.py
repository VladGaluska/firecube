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

"""Pack an existing Zarr product into a single-file zip archive.

The archive is readable with ``zarr.storage.ZipStore``. It is written exactly
once: built under a temporary name, closed, verified, then renamed into place.
"""

from __future__ import annotations

import os
import shutil
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import xarray as xr
import zarr
from zarr.buffer import default_buffer_prototype
from zarr.storage import LocalStore

from firecube.core.controlplane.types import CLAIMS_DIRNAME, CONTROL_DIRNAME
from firecube.core.filesystem.store_factory import create_zip_store
from firecube.core.storage.session import StorageSession
from firecube.core.uris import is_remote_target, local_path_from_target
from firecube.core.zarr._zarr_compat import sync

_CLAIMS_PREFIX = f"{CONTROL_DIRNAME}/{CLAIMS_DIRNAME}/"
_CONTROL_PREFIX = f"{CONTROL_DIRNAME}/"


@dataclass(frozen=True)
class ZipArchiveResult:
    """Summary of a zip archive creation.

    Attributes:
        source: URI of the archived product.
        target: Local path of the written archive.
        keys_written: Number of entries in the archive.
        file_size_bytes: Size of the archive on disk.
    """

    source: str
    target: str
    keys_written: int
    file_size_bytes: int


def zarr_to_zip(
    session: StorageSession, target: str, *, overwrite: bool = False
) -> ZipArchiveResult:
    """Copy every key of the session's product into a local zip archive.

    Control-plane files are included, except claim files. An existing archive
    is replaced only after the new one has been written and verified.

    Args:
        session: Storage session bound to the source product.
        target: Local path or ``file://`` URI of the archive to create.
        overwrite: Replace an existing archive at ``target``.

    Returns:
        A summary of the written archive.

    Raises:
        ValueError: If ``target`` is remote, is a directory, lies inside the
            source store, or the source has no Zarr v3 root metadata.
        FileExistsError: If ``target`` exists and ``overwrite`` is False.
        RuntimeError: If the written archive fails verification.
    """
    if is_remote_target(target):
        raise ValueError(f"Remote target not yet supported for zip output: {target}")
    target_path = local_path_from_target(target)
    if target_path.is_dir():
        raise ValueError(f"Target is a directory, expected a file path: {target}")
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Target file already exists: {target}")

    source_uri = session.product.product_uri
    if not source_uri.is_remote():
        source_path = local_path_from_target(source_uri.to_str()).resolve()
        if target_path.resolve().is_relative_to(source_path):
            raise ValueError(f"Target must not be inside the source store: {target}")

    root = source_uri.to_str().rstrip("/") + "/"
    sources = {
        key: uri
        for uri in session.find(source_uri)
        if not (key := uri.to_str()[len(root) :]).startswith(_CLAIMS_PREFIX)
    }
    if "zarr.json" not in sources:
        raise ValueError(f"No Zarr v3 root metadata found at {source_uri.to_str()}")
    keys = sorted(sources)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.with_name(f"{target_path.name}.{uuid.uuid4().hex}.partial")
    try:
        handle = create_zip_store(target=str(partial_path), mode="w")
        make_buffer = default_buffer_prototype().buffer.from_bytes
        try:
            for key in keys:
                with session.open(sources[key], "rb") as src:
                    sync(handle.store.set(key, make_buffer(src.read())))
        finally:
            handle.store.close()
        _verify(str(partial_path), expected_keys=keys)
        os.replace(partial_path, target_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    return ZipArchiveResult(
        source=source_uri.to_str(),
        target=str(target_path),
        keys_written=len(keys),
        file_size_bytes=target_path.stat().st_size,
    )


def _prepare_target(target_path: Path, *, overwrite: bool) -> None:
    """Same rules as the .tgm restore: a directory holding only .firecube/ is not a store yet."""
    if not target_path.exists():
        return
    if target_path.is_dir() and not (target_path / "zarr.json").exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Target Zarr already exists: {target_path}. Use overwrite=True to replace it."
        )
    if target_path.is_dir():
        shutil.rmtree(target_path)
    else:
        target_path.unlink()


def zip_to_zarr(archive: str, target: str, *, overwrite: bool = False) -> dict:
    """Restore a zarr zip archive into a local Zarr store by copying every key."""
    archive_path = Path(archive.removeprefix("file://"))
    target_path = Path(target.removeprefix("file://"))
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")

    with zipfile.ZipFile(archive_path) as zf:
        if (corrupt := zf.testzip()) is not None:
            raise ValueError(f"Corrupt entry {corrupt!r} in archive: {archive}")
        # claims are never archived, but never restore a stale claim either
        keys = sorted(
            k for k in zf.namelist() if not k.endswith("/") and not k.startswith(_CLAIMS_PREFIX)
        )
    if "zarr.json" not in keys:
        raise ValueError(f"No Zarr v3 root metadata found in archive: {archive}")

    _prepare_target(target_path, overwrite=overwrite)

    dest = LocalStore(target_path)
    make_buffer = default_buffer_prototype().buffer.from_bytes
    try:
        with zipfile.ZipFile(archive_path) as zf:
            for key in keys:
                sync(dest.set(key, make_buffer(zf.read(key))))
    finally:
        dest.close()

    # verify: the restored store opens and lists the same groups and arrays
    root = zarr.open_group(store=str(target_path), mode="r", zarr_format=3)
    with warnings.catch_warnings():
        # the control plane (.firecube/) is not part of the zarr hierarchy
        warnings.filterwarnings("ignore", message="Object at .* is not recognized")
        members = sorted(root.members(max_depth=None))

    groups = [name for name, node in members if isinstance(node, zarr.Group)]
    arrays = {name: node for name, node in members if isinstance(node, zarr.Array)}
    coordinates: set[str] = set()
    for name, node in arrays.items():
        group, _, var = name.rpartition("/")
        if tuple(_dimension_names(node)) == (var,):
            coordinates.add(name)
        for coord in str(node.attrs.get("coordinates", "")).split():
            coordinates.add(f"{group}/{coord}" if group else coord)

    return {
        "source": str(archive_path),
        "target": str(target_path),
        "groups": groups,
        "variables": sorted(n for n in arrays if n not in coordinates),
        "coordinates": sorted(n for n in arrays if n in coordinates),
        "keys_restored": len(keys),
        "controlplane_restored": any(k.startswith(_CONTROL_PREFIX) for k in keys),
    }


def zip_info(archive: str) -> dict:
    """Collect metadata of a zarr zip archive."""
    path = archive.removeprefix("file://")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Archive not found: {archive}")

    with zipfile.ZipFile(path) as zf:
        keys = zf.namelist()
    has_controlplane = any(k.startswith(f"{CONTROL_DIRNAME}/") for k in keys)

    store = create_zip_store(target=str(path), mode="r").store
    try:
        root = zarr.open_group(store=store, mode="r", zarr_format=3)
        group_vars: dict[str, dict] = {"": {}}  # "" is the root group
        with warnings.catch_warnings():
            # the control plane (.firecube/) is not part of the zarr hierarchy
            warnings.filterwarnings("ignore", message="Object at .* is not recognized")
            members = sorted(root.members(max_depth=None))
        for name, node in members:
            if isinstance(node, zarr.Group):
                group_vars.setdefault(name, {})
                continue
            group, _, var = name.rpartition("/")
            group_vars.setdefault(group, {})[var] = {
                "shape": [int(n) for n in node.shape],
                "dtype": str(node.dtype),
                "dims": _dimension_names(node),
                "zarr_chunks": [int(n) for n in node.chunks],
                "compressors": [_codec_name(c) for c in node.compressors],
            }
    finally:
        store.close()

    return {
        "path": path,
        "format": "zip",
        "size_bytes": os.path.getsize(path),
        "entries": len(keys),
        "groups": sorted(group_vars),
        "has_controlplane": has_controlplane,
        "group_details": [{"group": g, "variables": group_vars[g]} for g in sorted(group_vars)],
    }


def list_zip(archive: str) -> list[tuple[str, str]]:
    """List the contents of a zarr zip archive.

    Returns:
        One ``(group, dataset)`` pair per group that holds arrays, where
        ``dataset`` is the xarray representation of that group. ``""`` is the
        root group.
    """
    path = archive.removeprefix("file://")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Archive not found: {archive}")
    store = create_zip_store(target=str(path), mode="r").store
    try:
        root = zarr.open_group(store=store, mode="r", zarr_format=3)
        with warnings.catch_warnings():
            # the control plane (.firecube/) is not part of the zarr hierarchy
            warnings.filterwarnings("ignore", message="Object at .* is not recognized")
            members = sorted(root.members(max_depth=None))
        groups = [""] + [name for name, node in members if isinstance(node, zarr.Group)]
        listing = []
        for group in groups:
            node = root[group] if group else root
            if not isinstance(node, zarr.Group):
                continue
            if not any(isinstance(m, zarr.Array) for _, m in node.members()):
                continue
            ds = xr.open_zarr(store, group=group or None, consolidated=False)
            listing.append((group, repr(ds)))
    finally:
        store.close()
    return listing


def validate_zip(archive: str, *, quick: bool = False) -> list[str]:
    """Return the problems found in a zarr zip archive; an empty list means valid.

    quick: check the zip directory and Zarr metadata only, skip the CRC check of every entry.
    """
    path = Path(archive.removeprefix("file://"))
    if not path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")

    issues: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            keys = zf.namelist()
            if not quick and (corrupt := zf.testzip()) is not None:
                issues.append(f"CRC mismatch in entry {corrupt!r}")
    except zipfile.BadZipFile as exc:
        return [f"not a zip file: {exc}"]

    if "zarr.json" not in keys:
        issues.append("no Zarr v3 root metadata (zarr.json)")
        return issues

    store = create_zip_store(target=str(path), mode="r").store
    try:
        root = zarr.open_group(store=store, mode="r", zarr_format=3)
        with warnings.catch_warnings():
            # the control plane (.firecube/) is not part of the zarr hierarchy
            warnings.filterwarnings("ignore", message="Object at .* is not recognized")
            root.members(max_depth=None)  # parses the metadata of every group and array
    except Exception as exc:
        issues.append(f"cannot open as Zarr: {type(exc).__name__}: {exc}")
    finally:
        store.close()
    return issues


def _dimension_names(array: zarr.Array) -> list[str]:
    """Return the dimension names of a Zarr v3 array, empty when unnamed."""
    names = getattr(array.metadata, "dimension_names", None) or ()
    return [str(name) for name in names]


def _codec_name(codec: object) -> str:
    """Return the registered name of a Zarr v3 codec."""
    to_dict = getattr(codec, "to_dict", None)
    config = cast(dict[str, Any], to_dict()) if callable(to_dict) else {}
    return str(config.get("name", type(codec).__name__))


def _verify(path: str, *, expected_keys: list[str]) -> None:
    """Check a freshly written archive: valid, and holding exactly the expected keys."""
    issues = validate_zip(path)
    if not any(i.startswith("not a zip file") for i in issues):
        with zipfile.ZipFile(path) as zf:
            if sorted(zf.namelist()) != expected_keys:
                issues.append("entries differ from the keys written")
    if issues:
        raise RuntimeError(f"Zip archive verification failed for {path}: {'; '.join(issues)}")
