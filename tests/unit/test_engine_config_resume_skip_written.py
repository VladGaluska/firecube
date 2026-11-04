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

import pytest

from firecube.ingestor.config.engine import EngineConfig

pytestmark = pytest.mark.unit


def test_resume_skip_written_defaults_true() -> None:
    assert EngineConfig().resume_skip_written is True


def test_from_options_coerces_bool_strings(subtests: pytest.Subtests) -> None:
    for raw, expected in (("false", False), ("true", True), ("False", False), ("TRUE", True)):
        with subtests.test(msg=f"resume_skip_written={raw}"):
            cfg = EngineConfig.from_options({"resume_skip_written": raw})
            assert cfg.resume_skip_written is expected
