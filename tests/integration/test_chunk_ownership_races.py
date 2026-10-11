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


from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from firecube.core.zarr.region_writer import RegionZarrWriter
from tests.helpers.contention import force_contended_static, force_contended_units
from tests.integration.test_engine_owned_arrays import budget_plugin as budget_plugin
from tests.integration.test_engine_owned_arrays import cli_run, root

pytestmark = [pytest.mark.integration, pytest.mark.concurrency, pytest.mark.gate]
WORKERS = ("--option", "pipeline_workers=2", "--option", "pipeline_batch_size=1")


@pytest.mark.parametrize("sharded", [False, True], ids=["chunk", "shard"])
def test_pipeline_batches_sharing_stored_units_keep_every_slot(
    tmp_path, monkeypatch, budget_plugin, sharded
):
    budget_plugin.axis = "grid"
    budget_plugin.unit = 1 if sharded else 4
    budget_plugin.shards = 4 if sharded else None
    target = tmp_path / "cube.zarr"
    assert cli_run(target, "zarr preallocate").exit_code == 0
    gates = force_contended_units(
        monkeypatch, (target, "data/vec/c/0", (0,)), (target, "data/img/c/0/0/0", (0, 0, 0))
    )
    result = cli_run(target, "ingest", "direct", *WORKERS)
    assert result.exit_code == 0, result.output
    assert all(g.arrived.is_set() for g in gates)
    np.testing.assert_array_equal(root(target)["data/vec"][:], [1, 2, 3, 4])
    for slot, plane in enumerate(root(target)["data/img"][:]):
        np.testing.assert_array_equal(plane, np.full((2, 2), slot + 1))


def test_supplied_localstores_addressing_one_directory_share_one_lock(tmp_path, monkeypatch):
    target = tmp_path / "cube.zarr"
    zarr.open_group(str(target), mode="w").create_array(
        "data/vec", shape=(2,), chunks=(2,), dtype="float32", fill_value=np.nan
    )
    writers = [
        RegionZarrWriter(str(tmp_path / label), store=LocalStore(target))
        for label in ("label-a", "label-b")
    ]
    gates = force_contended_units(monkeypatch, (target, "data/vec/c/0", (0,)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(w.write_1d, "data", "vec", slot, np.asarray(slot + 1.0))
            for slot, w in enumerate(writers)
        ]
        for future in futures:
            future.result(timeout=60)
    assert gates[0].arrived.is_set()
    np.testing.assert_array_equal(root(target)["data/vec"][:], [1, 2])


def test_pipeline_static_transaction_refuses_divergent_second_batch(
    tmp_path, monkeypatch, budget_plugin
):
    budget_plugin.axis = "grid"
    budget_plugin.static = True
    budget_plugin.items = [0, 1]
    target = tmp_path / "cube.zarr"
    assert cli_run(target, "zarr preallocate").exit_code == 0
    held, first_payload = force_contended_static(monkeypatch, target, "data/lat")
    result = cli_run(target, "ingest", "direct", *WORKERS)
    assert held.is_set()
    arr = root(target)["data/lat"]
    np.testing.assert_array_equal(arr[:], first_payload[0])
    assert arr.attrs["firecube_static_written"] is True
    assert result.exit_code != 0
    assert "diverged" in result.output
