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

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import click

from firecube.core.storage.uri import StorageUri

SCHEME_TO_STORAGE_TYPE: Final[Mapping[str, str]] = {"file": "local", "s3": "s3"}


def _bare_path_suggestion(value: str) -> str:
    """Return " Did you mean file:///...?" for a bare path, or "" when none can be built."""
    if ":" in value:
        return ""
    try:
        suggestion = StorageUri.from_local_path(Path(value).expanduser().absolute()).to_str()
    except (RuntimeError, OSError):
        return ""
    return f" Did you mean {suggestion}?"


def parse_and_validate_uri(value: str | None, *, param_hint: str) -> StorageUri:
    """Admit one CLI URI value and return it as a StorageUri.

    Accepts lowercase ``file://`` and ``s3://`` only. A ``file://`` URI names no host,
    ``localhost`` included, and a path below the root. An ``s3://`` URI names a bucket. A
    value with ``?``, ``#`` or a control character is refused. The ``file://`` path is
    literal and never percent-decoded, because Firecube serializes ``file://`` URIs
    literally. The check is one pass over the value and does no I/O, except that the
    working directory and ``HOME`` may be read to build the bare-path suggestion.

    Args:
        value: The raw option or argument value.
        param_hint: The flag (``--archive``) or argument metavar (``A_URI``) named in the refusal.

    Returns:
        The admitted URI.

    Raises:
        click.BadParameter: The first rule the value fails, with ``param_hint`` as the hint.
    """

    def refuse(sentence: str) -> click.BadParameter:
        return click.BadParameter(sentence, param_hint=[param_hint])

    candidate = value or ""
    if not candidate.strip():
        raise refuse("URI must not be empty.")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in candidate):
        raise refuse("URI must not contain control characters.")
    if "://" not in candidate:
        suggestion = _bare_path_suggestion(candidate)
        raise refuse(f"URI scheme required (file:// or s3://).{suggestion}")

    scheme, rest = candidate.split("://", 1)
    if scheme.lower() not in SCHEME_TO_STORAGE_TYPE:
        raise refuse(f"URI scheme '{scheme}' not supported.")
    if scheme != scheme.lower():
        raise refuse(
            "URI scheme must be lowercase (file:// or s3://). "
            f"Did you mean {scheme.lower()}://{rest}?"
        )
    if "?" in rest or "#" in rest:
        raise refuse("URI must not contain '?' or '#'.")

    authority = rest.split("/", 1)[0]
    if scheme == "file" and authority:
        raise refuse(f"file:// URI must not name a host (got '{authority}'). Use file:///abs/path.")
    if scheme == "s3" and not authority:
        raise refuse("s3:// URI must name a bucket.")

    try:
        uri = StorageUri.parse(candidate)
    except ValueError as exc:
        raise refuse(str(exc)) from exc
    if scheme == "file" and uri.path == "/":
        raise refuse("file:// URI must name a path, not the root directory.")
    return uri


def resolve_storage_type(uri: StorageUri) -> str | None:
    """Return the storage type matching an admitted URI, if known."""
    return SCHEME_TO_STORAGE_TYPE.get(uri.protocol)


def validate_uri_storage_coherence(uri: StorageUri, storage_type: str) -> None:
    """Raise UsageError when an explicit storage type contradicts the URI scheme."""
    expected = resolve_storage_type(uri)
    if expected is None:
        raise click.UsageError(f"URI scheme '{uri.protocol}' not supported.")
    if expected == storage_type:
        return

    alternate = "an s3:// URI" if storage_type == "s3" else "a file:// URI"
    raise click.UsageError(
        f"--storage-type '{storage_type}' is incompatible with URI scheme '{uri.protocol}'. "
        f"Use --storage-type {expected} for {uri.protocol}:// targets, "
        f"or change --target to {alternate}."
    )


def apply_smart_default(uri: StorageUri, storage_type: str | None) -> str:
    """Resolve storage_type: explicit CLI value wins, else inferred from scheme."""
    if storage_type is not None:
        validate_uri_storage_coherence(uri, storage_type)
        return storage_type
    return SCHEME_TO_STORAGE_TYPE[uri.protocol]
