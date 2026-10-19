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

from typing import TYPE_CHECKING, Any, cast

import xarray as xr
import zarr

if TYPE_CHECKING:
    from firecube.core.filesystem.store_factory import ZarrStoreHandle
    from firecube.core.storage.session import StorageSession
    from firecube.core.storage.uri import StorageUri


class ZarrIO:
    def __init__(self, session: StorageSession) -> None:
        self._session = session

    def create_store(self, uri: StorageUri, mode: str = "w") -> ZarrStoreHandle:
        """Create a driver-aware Zarr store handle for a typed storage URI.

        The returned handle can be splatted into zarr/xarray APIs via
        ``handle.zarr_kwargs()`` and is constructed using the parent storage
        session's configured driver.
        """
        from firecube.core.filesystem import create_session_zarr_store
        from firecube.core.storage.session import storage_config_from_binding

        return create_session_zarr_store(
            uri=uri,
            storage_config=storage_config_from_binding(self._session._binding),
            mode=mode,
        )

    def open_group(self, uri: StorageUri, mode: str = "r") -> Any:
        kwargs: dict[str, Any] = {**self.create_store(uri, mode).zarr_kwargs(), "mode": mode}
        return zarr.open_group(**kwargs, use_consolidated=False)

    def open_dataset(self, uri: StorageUri, group: str = "", **xr_kwargs: Any) -> xr.Dataset:
        """Open an xarray dataset from a Zarr group using the session store.

        ``uri`` must be a canonical ``StorageUri``; raw URI strings are not
        accepted by this elevated session API. The helper always reads live metadata, so
        a caller cannot pass ``consolidated``. Other keyword arguments go to ``xr.open_zarr``.
        """
        handle = self.create_store(uri, mode="r")
        return xr.open_zarr(**handle.zarr_kwargs(), group=group, consolidated=False, **xr_kwargs)

    def write_dataset(
        self,
        ds: xr.Dataset,
        uri: StorageUri,
        group: str,
        mode: str = "w",
        **kwargs: Any,
    ) -> None:
        """Write an xarray dataset to a Zarr group using the session store.

        Store construction is centralized through ``create_store`` so local,
        remote, fsspec, and obstore targets follow the same session-bound driver
        configuration. ``zarr_format`` may be supplied for xarray-style callers;
        only format 3 is supported by the underlying Firecube write helper. The write never
        creates a consolidated snapshot, so a caller cannot pass ``consolidated``.
        """
        handle = self.create_store(uri, mode)
        zarr_format = kwargs.pop("zarr_format", 3)
        if zarr_format != 3:
            raise ValueError("session.zarr.write_dataset only supports zarr_format=3")
        ds.to_zarr(
            **handle.zarr_kwargs(),
            group=group,
            mode=cast(Any, mode),
            zarr_format=zarr_format,
            consolidated=False,
            **kwargs,
        )

    def refresh_consolidated_metadata(self, uri: StorageUri) -> None:
        """Rewrite the consolidated snapshot of each group under *uri* that already has one.

        A group without a snapshot stays without it, so the call never creates one.
        """
        from firecube.core.zarr.coord_consolidation import _refresh_consolidated_metadata

        _refresh_consolidated_metadata(self.open_group(uri, mode="r+"))
