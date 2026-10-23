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

"""The only module that imports from ``zarr.core``.

``sync`` (run a coroutine on zarr's event loop) and ``codecs_from_list``
(validate a codec list) have no public equivalent in zarr. Keeping them behind
this module means one file to fix if zarr moves them; a test forbids any other
``zarr.core`` import under ``src/``.

Tested zarr versions: 3.2.1, 3.3.0, 3.4.1.
"""

from __future__ import annotations

from zarr.core.codec_pipeline import codecs_from_list
from zarr.core.sync import sync

__all__ = ["codecs_from_list", "sync"]
