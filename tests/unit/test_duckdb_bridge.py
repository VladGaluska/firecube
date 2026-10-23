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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from firecube.core.credentials import Credentials
from firecube.core.duckdb.bridge import (
    _DUCKDB_OBSTORE_REMOTE_MESSAGE,
    apply_duckdb_storage,
)
from firecube.core.product.identity import ProductIdentity
from firecube.core.storage.binding import StorageBinding
from firecube.core.storage.driver_config import StorageDriverConfig
from firecube.core.storage.session import StorageSession
from firecube.core.storage.uri import StorageUri
from tests.helpers.rows import table_row


def _session(uri: str, driver_config: StorageDriverConfig) -> StorageSession:
    product_uri = StorageUri.from_local_path(uri) if uri.startswith("/") else StorageUri.parse(uri)
    return StorageSession(
        StorageBinding(
            identity=ProductIdentity.from_uri(product_uri, "parquet", product_name="test_product"),
            driver=driver_config,
        )
    )


@dataclass
class _RecordingConnection:
    statements: list[str] = field(default_factory=list)

    def execute(self, sql: str) -> None:
        self.statements.append(sql)


def test_duckdb_storage_bridge_routing(subtests: Any, tmp_path: Path) -> None:
    with table_row(subtests, "apply_duckdb_storage_fsspec_remote_https_vhost", tmp_path):
        session = _session(
            "s3://bucket/data.parquet",
            StorageDriverConfig(
                driver="fsspec",
                endpoint_url="https://minio.example:9000",
                credentials=Credentials(access_key="access", secret_key="secret"),
                region="eu-central-1",
                path_style=False,
            ),
        )
        con = _RecordingConnection()

        assert apply_duckdb_storage(con, session) is None
        assert con.statements == [
            "INSTALL httpfs; LOAD httpfs;",
            "SET s3_url_style='vhost'",
            "SET s3_endpoint='minio.example:9000'",
            "SET s3_use_ssl=true",
            "SET s3_region='eu-central-1'",
            "SET s3_access_key_id='access'",
            "SET s3_secret_access_key='secret'",
        ]

    with table_row(subtests, "apply_duckdb_storage_fsspec_remote_http_path_style", tmp_path):
        session = _session(
            "s3://bucket/data.parquet",
            StorageDriverConfig(
                driver="fsspec",
                endpoint_url="http://minio.internal:9000",
                path_style=True,
            ),
        )
        con = _RecordingConnection()

        assert apply_duckdb_storage(con, session) is None
        assert con.statements == [
            "INSTALL httpfs; LOAD httpfs;",
            "SET s3_url_style='path'",
            "SET s3_endpoint='minio.internal:9000'",
            "SET s3_use_ssl=false",
        ]

    with table_row(subtests, "apply_duckdb_storage_fsspec_local_is_noop", tmp_path):
        session = _session("/tmp/data.parquet", StorageDriverConfig(driver="fsspec"))
        con = _RecordingConnection()

        assert apply_duckdb_storage(con, session) is None
        assert con.statements == []

    with table_row(subtests, "apply_duckdb_storage_obstore_remote_raises", tmp_path):
        session = _session("s3://bucket/data.parquet", StorageDriverConfig(driver="obstore"))

        with pytest.raises(RuntimeError, match=_DUCKDB_OBSTORE_REMOTE_MESSAGE):
            apply_duckdb_storage(_RecordingConnection(), session)

    with table_row(subtests, "apply_duckdb_storage_obstore_local_is_noop", tmp_path):
        session = _session("/tmp/data.parquet", StorageDriverConfig(driver="obstore"))
        con = _RecordingConnection()

        assert apply_duckdb_storage(con, session) is None
        assert con.statements == []

    with table_row(subtests, "obstore_remote_output_raises", tmp_path):
        session = _session("/tmp/data.parquet", StorageDriverConfig(driver="obstore"))

        with pytest.raises(RuntimeError, match=_DUCKDB_OBSTORE_REMOTE_MESSAGE):
            apply_duckdb_storage(
                _RecordingConnection(), session, output_uri="s3://bucket/out.parquet"
            )

    with table_row(subtests, "obstore_local_source_and_output_ok", tmp_path):
        session = _session("/tmp/data.parquet", StorageDriverConfig(driver="obstore"))
        con = _RecordingConnection()

        assert apply_duckdb_storage(con, session, output_uri="/tmp/out.parquet") is None
        assert con.statements == []
