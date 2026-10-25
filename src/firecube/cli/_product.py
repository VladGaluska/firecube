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

import click

from firecube.cli._errors import render
from firecube.core.product.identity import ProductIdentity
from firecube.core.product.resolver import ProductResolver
from firecube.core.storage.uri import StorageUri


def resolve_product_identity(
    uri: StorageUri,
    *,
    format: str,
    product_name: str,
) -> ProductIdentity:
    """Resolve an admitted product URI into a ProductIdentity.

    Raises:
        click.UsageError: The resolver refuses the URI.
    """
    try:
        return ProductResolver.resolve(uri.to_str(), format=format, product_name=product_name)
    except ValueError as exc:
        raise click.UsageError(render(exc)) from exc
