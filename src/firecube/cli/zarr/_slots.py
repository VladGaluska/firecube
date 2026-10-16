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

"""``firecube zarr slots``: chunk-aligned parallel ingestion plans."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import click

from firecube.cli import _slot_planning
from firecube.cli._ctx import get_storage_config
from firecube.cli._input_filters import input_filters_option, resolve_input_filters
from firecube.cli._product import require_full_uri, resolve_product_identity
from firecube.cli._shared_options import (
    format_option,
    product_name_option,
    storage_driver_option,
    storage_type_option,
)
from firecube.cli._slot_planning import (
    PLAN_SCHEMA_VERSION,
    _chunk_aligned_remaining,
    _complement_intervals,
    _iter_partitions,
    _merge_intervals,
    _partition_count,
    _query_slots_coverage,
    _resolve_per_group_slot_sizes,
    _smallest_executable_slot_size,
)
from firecube.cli._typed_options import TypedOptionsParam
from firecube.cli._uri_policy import (
    apply_smart_default,
    parse_product_uri,
    validate_uri_storage_coherence,
)
from firecube.cli.zarr._common import _configure_ingestor_for_cli
from firecube.core.controlplane import ChunkManager
from firecube.core.storage.binding import StorageBinding
from firecube.core.storage.driver_config import StorageDriverConfig
from firecube.ingestor.api import ZarrGroupSpec
from firecube.ingestor.errors import ConfigurationError
from firecube.ingestor.registry import loader as ingestor_loader
from firecube.ingestor.registry.loader import discover_ingestors
from firecube.ingestor.templates.direct_zarr import DirectZarrIngestor


def _warn_stderr(message: str) -> None:
    """Write an operator warning to stderr without contaminating Click stdout capture."""
    os.write(2, f"{message}\n".encode())


class _WarnFilter(logging.Filter):
    """Block WARNING+ records for a target logger and its descendants.

    Thread-safe replacement for `logger.disabled = True`, which mutates a
    shared attribute and races between CLI threads. This filter is per-instance
    and only silences WARNING and above; DEBUG/INFO on the target logger stay
    visible if the operator has enabled them.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self._target = name
        self._target_dot = name + "."

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != self._target and not record.name.startswith(self._target_dot):
            return True
        return record.levelno < logging.WARNING


@contextlib.contextmanager
def _suppress_logger_warnings(logger_name: str) -> Generator[None]:
    """Temporarily suppress WARNING+ records that would corrupt JSON CLI output.

    Attaches ``_WarnFilter`` to the target logger and to every handler along
    the target's ancestor chain (up to and including root) because Python's
    ``logging`` does not consult ancestor-logger filters when a descendant
    record propagates: only handler-level filters are called on the way up.
    Attaching at handlers is what actually blocks descendant WARNING records
    before they reach the stderr sink that would corrupt JSON output.
    """
    target = logging.getLogger(logger_name)
    flt = _WarnFilter(logger_name)
    target.addFilter(flt)
    handlers_touched: list[logging.Handler] = []
    current: logging.Logger | None = target
    while current is not None:
        for handler in list(current.handlers):
            handler.addFilter(flt)
            handlers_touched.append(handler)
        if not current.propagate:
            break
        current = current.parent
    try:
        yield
    finally:
        for handler in handlers_touched:
            handler.removeFilter(flt)
        target.removeFilter(flt)


def _discover_ingestors_quietly() -> dict[str, type[Any]]:
    """Discover plugins without allowing registry warnings to contaminate JSON output."""
    with _suppress_logger_warnings("firecube.registry"):
        return discover_ingestors()


@click.command(
    "slots",
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="""\b
Examples:
  # Emit JSON plan for Argo/Kubeflow (resume-aware by default)
  firecube zarr slots <plugin> --target file:///tmp/x.zarr --product-name <name> \\
      --storage-type local --storage-driver fsspec --write-mode direct

  # Human-readable table output for inspection
  firecube zarr slots <plugin> --target file:///tmp/x.zarr --product-name <name> \\
      --storage-type local --storage-driver fsspec --write-mode direct -f table

  # Override slot partition size
  firecube zarr slots <plugin> --target file:///tmp/x.zarr --product-name <name> \\
      --storage-type local --storage-driver fsspec --write-mode direct --slot-size 200
""",
)
@click.argument("plugin")
@click.option(
    "--target",
    required=True,
    help="Target product URI required (file:///abs/path or s3://bucket/key).",
)
@product_name_option(required=True)
@storage_type_option(
    required=False, extra_help="Inferred from URI scheme when omitted (file://→local, s3://→s3)."
)
@storage_driver_option(required=False)
@click.option(
    "-w",
    "--write-mode",
    "write_mode",
    required=True,
    type=click.Choice(["staged", "direct"], case_sensitive=False),
    help="Write mode.",
)
@click.option(
    "--input-data",
    "input_data",
    default=None,
    help="Source input URI or path for AUTO discovery (optional).",
)
@input_filters_option
@click.option(
    "--slot-size",
    "slot_size",
    type=int,
    default=None,
    help=(
        "Slot partition size. Default per group: the least common multiple of the time units "
        "of its time-indexed arrays that are not engine-owned, the time coordinate included "
        "(declared chunks or shards, else one slot), and of the stored chunk or shard of each "
        "pod-written array the target already has; the whole axis when there is none."
    ),
)
@click.option(
    "--no-resume",
    "no_resume",
    is_flag=True,
    help="Disable resume-aware narrowing; emit full [0, total_slots) ranges.",
)
@click.option(
    "--yes-i-really-mean-it",
    "yes_i_really_mean_it",
    is_flag=True,
    help=(
        f"Emit a plan with more ranges than the ceiling "
        f"({_slot_planning.PLAN_RANGE_CEILING:,} ranges in the whole plan)."
    ),
)
@format_option(default="json")
@click.option(
    "--option",
    "option",
    multiple=True,
    type=TypedOptionsParam(),
    help="Plugin/engine option in key=value form.",
)
@click.pass_context
def slots(
    ctx: click.Context,
    plugin: str,
    target: str,
    product_name: str,
    storage_type: str,
    storage_driver: str,
    storage_anonymous: bool | None,
    write_mode: str,
    input_data: str | None,
    input_filters: list[str] | None,
    slot_size: int | None,
    no_resume: bool,
    yes_i_really_mean_it: bool,
    output_format: str,
    option: tuple[tuple[str, object], ...] = (),
) -> None:
    """Emit chunk-aligned slot ranges for orchestrated parallel ingestion.

    Read-only: does NOT mutate target storage or tracking state.
    JSON output is Argo withItems / Kubeflow ParallelFor compatible.

    Existing pod-written arrays in the target constrain the slot size by their stored chunk or shard.
    The command writes each range as it generates it, so memory does not grow with the range count.
    Above 100,000 ranges in the plan it prints one hint on stderr before the plan.
    Above 1,000,000 ranges it refuses the plan; pass --yes-i-really-mean-it to emit it anyway.
    """
    input_filters = resolve_input_filters(ctx, plugin, input_filters)

    require_full_uri(target, option_name="--target")
    parsed = parse_product_uri(target)
    storage_type = apply_smart_default(parsed, storage_type)
    validate_uri_storage_coherence(parse_product_uri(target), storage_type)

    storage_config = get_storage_config(
        ctx,
        overrides={
            "storage_type": storage_type,
            "storage_driver": storage_driver,
            "anonymous": storage_anonymous,
        },
        cache=False,
    )
    resolved_storage_driver = storage_config.storage_driver
    identity = resolve_product_identity(
        target,
        format="zarr",
        product_name=product_name,
        option_name="--target",
    )
    binding = StorageBinding(
        identity=identity,
        driver=StorageDriverConfig.from_storage_config(storage_config),
    )
    chunk_manager = ChunkManager(binding=binding)

    plugins = (
        ingestor_loader.AVAILABLE_INGESTORS
        if plugin in ingestor_loader.AVAILABLE_INGESTORS
        else _discover_ingestors_quietly()
    )
    if plugin not in plugins:
        raise click.ClickException(f"Unknown plugin '{plugin}'.")
    ingestor_cls = cast(Any, plugins[plugin])

    ingestor = (
        ingestor_cls(chunk_manager=chunk_manager) if input_data is not None else ingestor_cls()
    )

    if not isinstance(ingestor, DirectZarrIngestor):
        raise click.ClickException(
            f"Plugin '{plugin}' does not support slot-range planning. "
            "This command is available for plugins that write Zarr stores in parallel."
        )
    with _suppress_logger_warnings("firecube.registry"):
        plugin_ctx = _configure_ingestor_for_cli(
            ingestor,
            target=target,
            options=option,
            input_filters=input_filters,
            source=input_data or "",
            run_id="zarr-slots",
        )

    try:
        from firecube.core.index_spec import AUTO, IrregularTimeAxis
        from firecube.ingestor.api import BaseIngestor

        index_spec = ingestor.index_spec(plugin_ctx)
        if (
            input_data is None
            and index_spec is not None
            and any(
                isinstance(axis, IrregularTimeAxis) and axis.values is AUTO
                for axis in index_spec.groups.values()
            )
            and type(ingestor).discover_source_files is BaseIngestor.discover_source_files
        ):
            raise click.ClickException(
                f"Plugin '{plugin}' requires --input-data for AUTO discovery. "
                "Pass --input-data <source> or override discover_source_files() in the plugin."
            )
        ingestor._bind_index_at_startup(plugin_ctx)
        resolved_index = ingestor.resolved_index(plugin_ctx)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Plugin '{plugin}' returned no index_spec. "
            "See the plugin's documentation for parallel-write support."
        ) from exc
    current_record = chunk_manager.get_resolved_index(product=identity.product_name)
    if current_record is not None and current_record.identity_hash != resolved_index.identity_hash:
        _warn_stderr(
            "Plugin resolved index differs from persisted record; "
            "run `firecube zarr index rebuild` to reconcile."
        )
    elif (
        current_record is None
        and chunk_manager.get_slot_index_model(product=identity.product_name) is not None
    ):
        _warn_stderr(
            "Legacy slot-index record detected without a resolved-index record; "
            "run `firecube zarr index rebuild` to migrate."
        )
    global_schema = {group: resolved_index.size(group) for group in resolved_index.groups}
    for group_name, expected in global_schema.items():
        if expected <= 0:
            raise click.ClickException(
                f"Plugin '{plugin}' returned a non-positive count {expected} "
                f"for group '{group_name}'. Expected positive integers only."
            )

    schema = ingestor.zarr_schema(plugin_ctx)
    from firecube.ingestor.runtime.parallel_gate import validate_global_expected_subset_of_schema

    try:
        validate_global_expected_subset_of_schema(global_schema, schema)
    except ConfigurationError as exc:
        raise click.ClickException(str(exc)) from exc

    from firecube.ingestor.runtime.parallel_gate import warn_on_chunk_alignment

    if output_format == "json":
        with _suppress_logger_warnings("firecube.ingestor.runtime.parallel_gate"):
            warn_on_chunk_alignment(global_schema, schema)
    else:
        warn_on_chunk_alignment(global_schema, schema)

    stored_units = _stored_time_units(
        target,
        storage_config,
        schema,
        time_coord_name=ingestor._resolve_time_dim_name(),
    )
    resolved_slot_sizes = _resolve_per_group_slot_sizes(
        schema, slot_size, stored_units=stored_units
    )

    if no_resume:
        coverage_by_group: dict[str, list[tuple[int, int]]] = {g: [] for g in global_schema}
    else:
        try:
            coverage_by_group = _query_slots_coverage(
                ctx,
                target=target,
                product_name=product_name,
                storage_type=storage_type,
                storage_driver=resolved_storage_driver,
                groups=list(global_schema.keys()),
            )
        except Exception as exc:
            raise click.ClickException(
                f"firecube zarr slots: resume-aware coverage lookup failed ({exc}). "
                f"Either fix the underlying failure or pass --no-resume to emit "
                f"full ranges without coverage narrowing."
            ) from exc

    plans: list[_GroupPlan] = []
    for group_name in sorted(global_schema.keys()):
        total = int(global_schema[group_name])
        covered = _merge_intervals(coverage_by_group.get(group_name, []))
        remaining = _complement_intervals(covered, total)
        group_slot_size = resolved_slot_sizes.get(group_name, total)
        aligned_remaining, blocked_for_group = _chunk_aligned_remaining(
            remaining, group_slot_size, total
        )
        plans.append(
            _GroupPlan(
                name=group_name,
                total=total,
                slot_size=group_slot_size,
                covered=covered,
                remaining=remaining,
                aligned=aligned_remaining,
                blocked=blocked_for_group,
            )
        )
    groups_out = [plan.summary() for plan in plans]

    blocked_groups = [
        (group["name"], group["blocked_ranges"]) for group in groups_out if group["blocked_ranges"]
    ]
    if blocked_groups:
        details = "; ".join(
            f"group '{name}' has blocked ranges {ranges}" for name, ranges in blocked_groups
        )
        raise click.ClickException(
            "firecube zarr slots: cannot emit executable slots — chunk-alignment coverage would silently drop work. "
            f"Blocked: {details}. "
            "Remediation: preview with "
            "`firecube chunks delete-span --product-name <product-name> --run-id <id> --force --dry-run`, "
            "then commit with `--yes-i-really-mean-it` and re-ingest using `firecube ingest ... --option force_reingest=true`. "
            "See docs/operations/parallel-zarr-writes.md#recover-a-blocked-plan."
        )

    range_count = sum(plan.range_count for plan in plans)
    if range_count > _slot_planning.PLAN_RANGE_CEILING and not yes_i_really_mean_it:
        units = _resolve_per_group_slot_sizes(schema, None, stored_units=stored_units)
        raise click.ClickException(_ceiling_refusal(plans, range_count, units))
    if range_count > _slot_planning.PLAN_RANGE_HINT:
        _warn_stderr(_range_hint(plans, range_count))

    header = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "product_name": product_name,
        "target": target,
        "storage_type": storage_type,
        "storage_driver": resolved_storage_driver,
        "write_mode": write_mode,
        "strategy": "indexed-region",
        "groups": groups_out,
    }
    _emit_slots_output(header, plans, output_format)


def _stored_time_units(
    target: str,
    storage_config: Any,
    schema: Sequence[ZarrGroupSpec],
    *,
    time_coord_name: str,
) -> dict[str, list[int]]:
    """Return the stored chunk or shard time extent of each pod-written array the target has.

    Existing pod-written arrays constrain the slot size by their stored chunk or shard,
    the unit pod startup judges windows by. The target opens read-only and nothing is
    created in it; a target without a root group, or without an array, adds no unit.
    """
    import zarr
    from firecube.core.filesystem.ops import create_filesystem_for_uri
    from firecube.core.filesystem.store_factory import create_zarr_store
    from firecube.core.zarr.guard.engine_owned import is_pod_written_slot_array
    from firecube.core.zarr.guard.unit_ownership import stored_write_unit

    unreadable = f"firecube zarr slots: cannot read the layout of existing arrays in {target}"
    units: dict[str, list[int]] = {}
    try:
        # Existence goes through the storage driver before any zarr store is built:
        # building one can create the target, and only an absent root means no units.
        fs, target_uri = create_filesystem_for_uri(target, storage_config, format="zarr")
        if not fs.exists(target_uri.join("zarr.json")):
            return units
        handle = create_zarr_store(uri=target, storage_config=storage_config, mode="r")
        root = cast(
            Any,
            zarr.open_group(
                **handle.zarr_kwargs(), mode="r", zarr_format=3, use_consolidated=False
            ),
        )
        for group_spec in schema:
            for arr_spec in group_spec.arrays:
                if not is_pod_written_slot_array(arr_spec, time_coord_name=time_coord_name):
                    continue
                path = f"{group_spec.group}/{arr_spec.name}"
                # An absent array adds nothing; one that exists must parse, or the plan fails.
                if not fs.exists(target_uri.join(*path.strip("/").split("/"), "zarr.json")):
                    continue
                arr = root[path]
                if getattr(arr, "ndim", 0) == 0:
                    continue
                units.setdefault(group_spec.group, []).append(int(stored_write_unit(arr)[0][0]))
    except Exception as exc:
        raise click.ClickException(f"{unreadable} ({type(exc).__name__}: {exc}).") from exc
    return units


@dataclass(frozen=True)
class _GroupPlan:
    """One group's slot plan as intervals; its ranges are generated on demand, never stored."""

    name: str
    total: int
    slot_size: int
    covered: list[tuple[int, int]]
    remaining: list[tuple[int, int]]
    aligned: list[tuple[int, int]]
    blocked: list[tuple[int, int]]

    @property
    def range_count(self) -> int:
        """Return the number of ranges `ranges()` yields."""
        return _partition_count(self.aligned, self.slot_size)

    def ranges(self) -> Iterator[tuple[int, int]]:
        """Yield the group's ranges in slot order, one at a time."""
        return _iter_partitions(self.aligned, self.slot_size)

    def summary(self) -> dict[str, Any]:
        """Return the group's entry of the plan's `groups` list.

        `static_owner` is the group's first range: the first slot_size piece of the first
        aligned interval, the lowest range since aligned intervals are sorted.
        """
        static_owner = None
        if self.aligned:
            owner_start, interval_end = self.aligned[0]
            owner_end = min(owner_start + self.slot_size, interval_end)
            static_owner = {"slot_start": owner_start, "slot_end": owner_end}
        return {
            "name": self.name,
            "total_slots": self.total,
            "slot_size": self.slot_size,
            "covered_ranges": [[s, e] for s, e in self.covered],
            "remaining_ranges": [[s, e] for s, e in self.remaining],
            "blocked_ranges": [[s, e] for s, e in self.blocked],
            "static_owner": static_owner,
        }


def _group_counts(plans: Iterable[_GroupPlan]) -> str:
    """Describe each group's range count and slot size, for operator messages."""
    return "; ".join(
        f"group '{plan.name}': {plan.range_count} ranges at slot size {plan.slot_size}"
        for plan in plans
    )


def _ceiling_refusal(plans: list[_GroupPlan], range_count: int, units: dict[str, int]) -> str:
    """Build the refusal for a plan above the range ceiling, naming an executable size if any."""
    ceiling = _slot_planning.PLAN_RANGE_CEILING
    suggested = _smallest_executable_slot_size(
        {plan.name: plan.remaining for plan in plans},
        {plan.name: plan.total for plan in plans},
        units,
        ceiling,
    )
    if suggested is None:
        remedy = (
            f"No --slot-size gives an executable plan of at most {ceiling} ranges; "
            "pass --yes-i-really-mean-it to emit this plan anyway."
        )
    else:
        remedy = (
            f"Pass --slot-size {suggested} for an executable plan of at most {ceiling} ranges, "
            "or --yes-i-really-mean-it to emit this plan anyway."
        )
    return (
        f"firecube zarr slots: the plan has {range_count} ranges, more than the limit of "
        f"{ceiling} ({_group_counts(plans)}). {remedy}"
    )


def _range_hint(plans: list[_GroupPlan], range_count: int) -> str:
    """Build the one-line stderr hint for a plan above the range hint threshold."""
    return (
        f"firecube zarr slots: the plan has {range_count} ranges ({_group_counts(plans)}). "
        "For fewer ranges pass --slot-size N with N a multiple of each group's slot size, "
        "or run one Indexed Job over JOB_COMPLETION_INDEX with FIRECUBE_SLOT_SIZE."
    )


#: Pieces joined into one write: bounds memory while keeping the number of writes small.
_WRITE_BATCH = 1024


def _echo_batched(pieces: Iterable[str]) -> None:
    """Write the pieces to stdout in order, joined in batches of `_WRITE_BATCH`."""
    batch: list[str] = []
    for piece in pieces:
        batch.append(piece)
        if len(batch) >= _WRITE_BATCH:
            click.echo("".join(batch), nl=False)
            batch.clear()
    if batch:
        click.echo("".join(batch), nl=False)


def _json_pieces(header: dict[str, Any], plans: list[_GroupPlan]) -> Iterator[str]:
    """Yield the plan as JSON text, byte-identical to `json.dumps(plan, indent=2)` plus a newline.

    The header goes through `json.dumps` with an empty `ranges` list; each range is formatted
    at the indentation `json.dumps` gives a member of that list. Only the group name needs
    escaping, so it goes through `json.dumps` once per group.
    """
    head, _, _ = json.dumps({**header, "ranges": []}, indent=2).rpartition("[]")
    yield head
    opener = "["
    for plan in plans:
        group = json.dumps(plan.name)
        for start, end in plan.ranges():
            yield (
                f"{opener}\n"
                "    {\n"
                f'      "group": {group},\n'
                f'      "slot_start": {start},\n'
                f'      "slot_end": {end},\n'
                '      "env": {\n'
                f'        "FIRECUBE_SLOT_START": "{start}",\n'
                f'        "FIRECUBE_SLOT_END": "{end}",\n'
                f'        "FIRECUBE_SLOT_GROUP": {group}\n'
                "      },\n"
                '      "cli_args": [\n'
                '        "--slot-start",\n'
                f'        "{start}",\n'
                '        "--slot-end",\n'
                f'        "{end}",\n'
                '        "--slot-group",\n'
                f"        {group}\n"
                "      ]\n"
                "    }"
            )
            opener = ","
    yield "[]\n}\n" if opener == "[" else "\n  ]\n}\n"


def _emit_slots_output(header: dict[str, Any], plans: list[_GroupPlan], output_format: str) -> None:
    """Write the plan to stdout, generating each range as it is written.

    Raises:
        click.ClickException: `output_format` is not json, table or csv; nothing is written.
    """
    if output_format == "json":
        _echo_batched(_json_pieces(header, plans))
        return
    separators = {"table": "\t", "csv": ","}
    if output_format not in separators:
        raise click.ClickException(f"Unsupported output format: {output_format}")
    sep = separators[output_format]
    click.echo(f"group{sep}slot_start{sep}slot_end")
    _echo_batched(
        f"{plan.name}{sep}{start}{sep}{end}\n" for plan in plans for start, end in plan.ranges()
    )
