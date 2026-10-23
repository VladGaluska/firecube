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

"""Calendar-declared time-axis arithmetic (internal; nothing is re-exported).

Modules, bottom-up: ``_codecs`` (lazy xarray coders), ``calendar``
(calendar vocabulary), ``guard`` (non-Gregorian value refusal), ``gregorian``
(the no-xarray fast path), ``encode`` (encoding onto a declared calendar axis),
``canonical`` (coordinate canonicalisation), ``convert``
(``datetime64`` conversion) and ``encoding`` (the storage encoding a
coordinate array uses). A module imports only from modules listed before it.
"""
