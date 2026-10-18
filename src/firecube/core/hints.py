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

"""Recovery commands that control-plane refusals print for the operator to run."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from firecube.core.controlplane.repo import ManifestRepository
    from firecube.core.controlplane.types import ClaimInfo

__all__ = ["abandon_run_command", "clear_claim_command", "maintenance_refusal_message"]


def abandon_run_command(product_uri: str, run_id: str) -> str:
    """Return the `firecube chunks runs abandon` command for one run, reason left as a placeholder.

    The product URI and run id are shell-quoted. The command omits `--yes-i-really-mean-it`,
    so the CLI still prompts or refuses.
    """
    return (
        f"firecube chunks runs abandon --product-name {shlex.quote(str(product_uri))} "
        f'--run-id {shlex.quote(str(run_id))} --reason "<reason>"'
    )


def clear_claim_command(product_uri: str, domain: str, *, force: bool = False) -> str:
    """Return the `firecube chunks claims clear` command for one claim domain.

    The product URI and domain are shell-quoted. Pass `force=True` for a claim that is not
    stale. The command omits `--yes-i-really-mean-it`, so the CLI still prompts or refuses.
    """
    command = (
        f"firecube chunks claims clear --product-name {shlex.quote(str(product_uri))} "
        f"--domain {shlex.quote(str(domain))}"
    )
    return f"{command} --force" if force else command


def maintenance_refusal_message(
    *,
    repo: ManifestRepository,
    product: str,
    operation: str,
    detail: str,
    claims: Sequence[ClaimInfo],
) -> str:
    """Return the message that refuses a maintenance operation, with commands that run.

    Name each blocking claim and its owner. Print the clear command for it (`--force` when the
    claim is fresh) and, when its owner resolves to exactly one recorded run, the abandon
    command. Read errors drop the owner line and the abandon command, never the clear command.
    """
    message = f"Cannot run {operation} for product {product}: {detail}."
    if not claims:
        return message
    try:
        product_uri = repo.get_product_root_uri(product)
    except Exception as exc:
        return f"{message} The product root could not be resolved: {exc}."
    owner_runs: dict[str, Any] = {}
    for claim in claims:
        clear = clear_claim_command(product_uri, claim.domain, force=not claim.stale)
        abandon = ""
        if claim.owner_id not in owner_runs:
            try:
                owner_runs[claim.owner_id] = repo.resolve_claim_owner(
                    product=product, owner_id=claim.owner_id
                ).run
            except Exception:
                owner_runs[claim.owner_id] = None
        run = owner_runs[claim.owner_id]
        if run is not None:
            abandon = (
                f"\nor abandon run {run.run_id} (status: {run.status}), which releases the claims "
                "it alone owns:"
                f"\n  {abandon_run_command(product_uri, run.run_id)}"
            )
        message += (
            f"\nIf no process of {claim.owner_id} is running, clear the claim {claim.domain}:"
            f"\n  {clear}{abandon}"
        )
    return message
