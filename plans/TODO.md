# TODO

Accepted work only. Speculative ideas go to [IDEAS.md](IDEAS.md) until a design discussion promotes them.
New decisions are recorded in [DONE.md](DONE.md) with a date.

## Active Work

### Generic numeric flag helpers (2026-09-12)

**Status**: FUTURE; no implementation in the current core-readiness work.

Explore domain-agnostic numeric flag construction only after agreeing concrete caller contracts. Thresholds, bit meanings, missing-value rules, and published metadata remain caller-owned. Require behavior tests against caller-supplied rules; do not embed product conventions or change aggregation counts.

The same readiness scope defers `firecube.testing`, numeric aggregation counts, composite-source APIs, a new extraction lifecycle API, universal item-time filtering, new group/layout/slice hooks, and plugin-specific write ordering.

### Pattern parser adoption (2026-09-12)

**Status**: OPEN; intended first consumers are plugins that parse filename fields with their own code. No plugin migration is included in core readiness.

Replace each plugin's explicit filename-field parser with `parse_pattern` after verifying its accepted names, field values, and failure behavior. Keep timestamp selection, timezone policy, and source filtering in the plugin. Adoption must remove the replaced parsing code; the two-consumer duplication-removal condition is not yet met, so this is an optional utility addition, not a completed promotion.

### Typed Zarr codec entries (2026-09-14)

**Status**: FUTURE; not part of the plugin scaffolding fixes.

Zarr v3 codec objects are bare `dict` everywhere: `ZarrTemplateConfig.zarr_codecs` (`list[dict]`), `ZarrArraySpec.filters`/`serializer`/`compressors`, the codec pipeline, and the Zarr writer. Editors show their keys and values as `Unknown`. Validation already fixes the shape (`name` required, `configuration` an object when present, no other keys).

Leaning: a public `ZarrCodec` `TypedDict` (`name: str`, `configuration: dict[str, Any]`) applied to every codec field and signature, exported from `firecube.ingestor.api` with a reference entry. A `TypedDict` keeps the plain-dict runtime shape that JSON, TOML, and `coerce_cli_value` produce; a dataclass would need conversion wherever codecs flow. Overturn it if a runtime-validation library is adopted for configuration first.

Decide before implementing:
- Whether `configuration` is required. Validation treats it as optional, but the writer fails every batch without it (confirmed in the 2026-09-14 plugin-guide validation). Either require it in validation and the type, or normalize entries before writing and type it `NotRequired`.
- How the generated `ZarrStorageConfig` example names the type. An override must repeat the parent's exact type or pyright reports an incompatible override, and a live import is unused while the example stays commented.

---

### §36 `ZarrWriteContext` async-concurrency leak — convert to scoped config

**Status**: OPEN
**Scope**: Zarr write configuration hygiene
**Priority**: medium

`src/firecube/ingestor/runtime/zarr/write_context.py` still calls
`zarr.config.set({"async.concurrency": ...})` bare during `ZarrWriteContext.__enter__`.
This pre-existing setting is outside the Wave C2 `zarr_write_empty_chunks` work,
but it should be converted to a scoped context manager so async concurrency is
restored when the write context exits.

---

### §33 DirectZarr slot-range parallelism - roadmap

- **Landed:** `IndexSpec` + `RegularTimeAxis` + `ResolvedIndex`; byte parity for the existing production parallel plugins; recursion defect fixed.
- **Landed:** `IntegerAxis` + engine-owned `.firecube/index/current.json` record (`ResolvedIndexRecord`); `firecube zarr index show/verify/rebuild` CLI; atomic reader migration path documented.
- **Landed:** `IrregularTimeAxis` + `AUTO` sentinel + content-addressed item manifest; `--dry-run` for preallocate; `--derived` for index show; closes #27 spirit. See DONE.md 2026-08-24.
- **Landed:** `IndexedWrite` high-level abstraction, merged into a single `build_write_intents` contract accepting mixed `WriteIntent | IndexedWrite` lists; documented in `docs/guides/plugins/direct-zarr.md` and the API reference. See DONE.md 2026-08-24 and its 2026-09-02 amendment.
- **Future deprecation pass:** `SlotIndexModel`, `SlotAxis`, and `as_legacy_slot_index_model()` are kept for byte-parity compatibility with cubes written before the new index model. Schedule a deprecation pass once all known production cubes have been migrated via `firecube zarr index rebuild`. The deprecation should add `DeprecationWarning` on construction, update the allowlist in `test_api_docs_coverage.py`, and remove the symbols in a subsequent major version.

---

### Parallel chunk ownership follow-ups (2026-10-11)

- **Sparse staged writes by seeding.** Staged runs that write only part of a chunk or shard are refused because promotion replaces whole objects. Supporting them means seeding the touched units of every pod-written array into the workspace (as `GenericZarrIngestor` appends already do) and relaxing the promotion check; needs a design for seeded engine-owned and unchanged objects so they are not re-uploaded.
- **obstore partial write on copy failure (pre-existing; read in code, not run).** `StreamingObstoreWriteBuffer.close()` also runs when a copy fails partway and saves what it holds. Local targets now land that in the publication temporary, which is deleted; on S3/GCS it can still overwrite the object with a partial body.
- **`StorageSession.ls` raises on local fsspec targets (latent; seen raising in a test run, no src caller by grep).** `FsspecFilesystem.ls` returns bare paths and `_storage_uri_from_path("file", ...)` passes them to `StorageUri.parse`, which rejects bare paths.
- **Plugin discovery depends on import order in tests (pre-existing; seen once, not reproduced since).** Running the verify files of the chunk-ownership work together in one pytest process once failed 8 tests of a since-deleted pods test file with `Unknown plugin`: fixture plugins are found through import side effects, not only entry points. The combined files pass (345 tests) without that file, so the symptom is gone, but the cause (discovery by import side effect) is not fixed. Make discovery independent of test order.
- **Refuse serial staged runs with partial chunk coverage earlier (optimisation; read, not run).** Slot-window staged runs are refused before any batch because `inspect_item` yields the planned slots; plain serial staged runs are refused only just before promotion, after workspace writes (target untouched, wasted compute only). Idea: when the plugin overrides `inspect_item` and has a resolved index, inspect the already-filtered batch items before the first batch and reuse `require_planned_slots_fill_units`. Estimate from review: four files (`engine.py`, `direct_zarr.py`, two test files), ~150-250 production and 300-500 test lines, 1-2 coder days plus two review rounds. Hazards: `ItemInfo` carries no group, array or row/channel footprint, so completeness of rows and channels cannot be proven early; a fresh serial run writing 6 slots into shape `(6,)` with chunks `(12,)` succeeds today and would be falsely refused if the declared 12-slot extent were used (serial creation derives the extent from the intents' `ts_index`, the `max_ts` computation in `IndexedRegionStrategy._write_groups`); discovery overcounts plugin resume-skipped writes and cannot prove force-reingest coverage; a second `inspect_item` call may cost or have side effects. The late ledger check (the `_IncompleteUnitsRefusal` call in the engine's post-pipeline lifecycle method) must stay either way.
- **A refused `zarr preallocate` can leave files outside `.firecube/` changed (run on 2026-10-12; read in code).** The ownership refusals (`_refuse_ownership_change` in `src/firecube/cli/zarr/_preallocate.py`) fire per array, inside the per-group loop of `preallocate`, after earlier work has already run. Arrays declared before the refused one in its group are created first (`_create_array`): a one-group run that adds two such arrays and then refuses exits 1 with their `zarr.json` files left behind, for both refusal kinds. Earlier groups are processed completely first, including observed coordinate values and engine-owned values (`_materialize_engine_owned_arrays`). What the order guarantees is only that the refused group's own coordinate and engine-owned materialization has not started. Separately, a preallocate that reaches an existing regular coordinate rewrites its `zarr.json`, refused or not (the `update_array_attrs` calls on an existing coordinate in `materialize_regular_coord_array`, `src/firecube/core/zarr/coord_materialization.py`): identical bytes with a new mtime when the coordinate's metadata is unchanged, new bytes when the spec's coordinate attrs changed. The adoption test watches only the engine-owned array for preallocate (`test_preallocate_then_pods_preserve_engine_arrays_and_layout` in `tests/integration/test_engine_owned_arrays.py`); its one-group scenario declares no earlier arrays, so watching the whole group passes there and would not catch any of these cases. Repair: check ownership for every group before the group loop creates or materializes anything.
- **Sharded time coordinates are claimed by inner chunk, and their layout depends on who creates them (pre-existing on `c160535`; run on 2026-10-12).** The coordinate claim compares a sharded coordinate by its inner chunk, not by its shard, so two windows in different inner chunks of one shard are admitted (the `chunk_axis_range` comparison in `ChunkManager.claim_coord_materialization_window`, `src/firecube/core/controlplane/manager.py`). One coordinate declaration also gets different layouts depending on who creates it: an observed (`mode="floor"`) preallocate makes `min(256, slots)` chunks, unsharded (`resolve_coord_chunks` in `src/firecube/core/zarr/_coord_chunks.py`, called from `src/firecube/core/zarr/coord_materialization.py`); grid or irregular preallocate and ingest make the largest divisor of the shard as chunk, inside the declared shards (`_divisor_chunks` and `resolve_effective_layout` in `src/firecube/core/zarr/_layout.py`). For a coordinate that preallocate marked, no lost update is reachable through these claims: only preallocate writes its values, under product-wide claims on which a second preallocate fails at once, and ingest workers only verify it. Not examined: `firecube zarr consolidate-time-coord`. Unmarked (legacy) coordinates, which ingest writes, are covered by the bullet on unmarked time coordinates below. Repair, if one is needed: claim a sharded coordinate by its shard, and create the coordinate with one layout whoever creates it.
- **Other consolidated-metadata opens still act on stale metadata (pre-existing on main `c160535`; reproduced by a review probe on 2026-10-12).** The chunk-ownership work makes its own ownership, layout and writer-root opens ignore consolidated metadata (`use_consolidated=False`). Three other opens still trust it: `src/firecube/cli/zarr/_consolidate.py:180` (`consolidate-time-coord`: after a consolidate and a live `firecube_coord_managed` stamp it exits 0, rewrites the coordinate and replaces that marker with `firecube_preallocated`; the unconsolidated control refuses), `src/firecube/core/zarr/validation.py:613,625` and `src/firecube/ingestor/runtime/zarr/append.py:84` (consolidated shape 2 against live shape 4 gives an append resume cursor of 2). Other default opens in staged metadata, timestamp state, comparison and validation, and the advice and export readers were seen but not probed. Repair: pass `use_consolidated=False` wherever the result drives a decision, with a test in each file's existing test home.
- **A slot-window ingest is accepted on an unmarked time coordinate, and two processes can lose a timestamp (pre-existing on `c160535`; run on 2026-10-12).** The group's time coordinate carries neither `firecube_preallocated` nor `firecube_coord_managed` when no preallocate ran or the store predates the markers. Pods then write it through the legacy path under an in-process lock only (`RegionZarrWriter._write_legacy_timestamp` in `src/firecube/core/zarr/region_writer.py`), and pod startup and the per-batch check skip the coordinate by name (`is_pod_written_slot_array` in `src/firecube/core/zarr/engine_owned.py`, used by `DirectRegionAdapter.pod_written_specs` and `IndexedRegionStrategy._pod_written_paths`). Two processes with windows [0,1) and [1,2) in one coordinate chunk, held so that both read before either wrote, both exited 0 and slot 0's timestamp stayed NaT while its data was written (`bcfa86e` and `c160535`); unforced runs can lose a value, rarely (0 of 20 in the research sample; a review sample lost 2 of 20, not re-run); a control whose values were already stored lost nothing; a fresh target with concurrent pods was not tested. The unmarked store was simulated (markers stripped, values reset to NaT). Repair: refuse a slot-window ingest whose time coordinate is absent or unmarked, naming `firecube zarr preallocate`, as the engine-owned refusal does. Preallocate refuses an unmarked observed shell (`tests/unit/test_materialize_regular_coord_array.py::test_floor_mode_existing_unmarked_array_refuses_legacy_shell`), so an unmarked observed shell has no upgrade path today (grid and irregular coordinates were reported upgradable by a review probe; not run here).
- **Staged runs do not enforce static write-once (pre-existing on `c160535`; run on 2026-10-12).** Staged runs write and check static data in the workspace copy, never in the final target, and seeding strips `firecube_static_written` from the workspace (`src/firecube/ingestor/runtime/zarr/staged_metadata.py`), so the write-once check never sees the previous run's values. Promotion publishes array metadata through the session's publisher (`update_document` in `src/firecube/core/zarr/metadata_authority.py`: target attributes win, the workspace adds missing keys), so the marker does reach and stay on the target; it is not consulted. Reproduced after the guarded write layer on 2026-10-14: after a preallocate, or on a target a staged run created, two successful staged runs (the second with `resume_existing=true`) changed a static array from [0, 0] to [1, 1], with the marker on disk throughout. A later staged run with another static payload exits 0 and overwrites it (100 to 900) in every sequence run before the layer: staged then staged, and direct then staged, each with and without a preceding preallocate. A direct run after an unmarked staged overwrite stamps the marker over data it never compared; direct after direct refuses. The `WriteIntent.static` docstring promises `SchemaDriftError` on any later run (`src/firecube/ingestor/templates/direct_zarr.py:861-864`), and the `plans/DONE.md` entry "2026-06-25 — DirectZarr plugin parity — core fixes" (T3, staged-seed marker strip) claim that the final target keeps the marker describes retention only, not enforcement. Repair: verify a staged static payload against the final target (as engine-owned verification does), at promotion or at the first batch.
- **A static array's write transaction is not serialised between processes (pre-existing on `c160535`; read in code on 2026-10-12, not reproduced).** `IndexedRegionStrategy` dispatches static intents outside the slot claims (the `static_intents` loops in `IndexedRegionStrategy._write_groups`, `src/firecube/ingestor/runtime/zarr/strategies/indexed_region.py`), and the whole-array lock that serialises a static array's marker check, write and stamp is held in one process only; non-owner suppression (`suppress_static_emission_for_non_owner`) defaults to false. Two slot-window processes that both emit the same unmarked static array can therefore both pass the marker check and write, the later replacing the earlier when their payloads differ. Identical payloads, the usual latitude and longitude grids, are harmless. Repair: let only the owner of the range that starts at slot 0 emit a static array, or claim the array across processes.

---

### GenericZarr on the guarded write layer (2026-10-14)

**Status**: FUTURE; follows the DirectZarr write session. `guarded_write`, `metadata_authority` and `promotion_plan` exist in `src/firecube/core/zarr/`, and DirectZarr uses them; `GenericZarrIngestor` uses none of the three.

GenericZarr writes through xarray (`to_zarr`) in `write_dataset_to_zarr`, with a per-batch `zarr_append` claim and the ordered gate for in-process order. Zarr leaves overlapping writers to the caller, so nothing below the claim protects a stored unit or a `zarr.json`. Four parts, in this order:

- **`DatasetRegionAdapter`.** Turn each xarray region write of `write_dataset_to_zarr` into footprints (array path plus index box in stored geometry) and send one task per stored unit (shard if sharded, else chunk) through `guarded_write`. The Dask race inside one unit is already closed: with `align_chunks=True` and `xarray>=2026.2.0` (which aligns Dask tasks to shards), a region write whose inner chunks share a shard keeps every value (run on xarray 2026.2.0 and zarr 3.2.1: `[11, 12, 13, 14]`). What the adapter adds is what xarray does not give: the chokepoint's unit locks against other writers of the same unit, absent and deleted unit reporting for region writes, and bounded concurrency. Done when region writes go through `guarded_write` and report absent and deleted units, with a test that a region write whose inner chunks share a shard keeps every value; that test passes on the current path and is kept as regression coverage, not as a fail-first test.
- **Bounded region mode (opt-in).** A preallocated extent and a slot mapping the plugin declares; disjoint units are then written concurrently, each under its unit lock. Today two concurrent appends grow the shape and one writer's tail is lost silently, so the ordered gate stays the default. Tail growth stays serial: a plugin that does not declare the mapping keeps the append path.
- **Generic staged delta.** Promote a GenericZarr staged run through `promotion_plan` and `publish_plan` instead of `upload_tree`, with the units the run proved written. This also ends the staged all-fill retention: a staged run that rewrites a whole unit to the fill value deletes the object in the workspace (`write_empty_chunks=False`) and promotion by `upload_tree` only copies, so the target keeps the old values. The engine already takes the delta path for a host that reports `_staged_promotion_units`; GenericZarr does not report them. The smaller repair (staged workspace writes keep empty chunks) is not a substitute for the delta.
- **`AppendCoordinator`.** Allocation of the append tail, the cursor and batch order stay serial and move behind one object (`AppendResumeService`'s cursor, `AppendOrder` and `OrderedWriteGate` become its parts, with the per-batch `zarr_append` claim) so the adapter and the bounded region mode share one source of the next slot.

Fold into these the GenericZarr metadata sites that bypass the metadata authority (`metadata_authority`: `zarr.json` changes only under a held authority; its five editing operations `create_or_open_array`, `grow_array`, `set_attributes`, `set_attributes_if_absent` and `update_document` re-read the node from the store under the per-node lock; `grow_array` never makes an array smaller; `update_document` applies a caller-supplied merge under that lock and does not restrict the result, so a caller can shrink; `publish_document` replaces the document it is given and `publish_document_if_absent` writes only where none exists, and neither re-reads):
- Creation and append `to_zarr`, in `write_dataset_to_zarr` of `src/firecube/ingestor/runtime/zarr/write.py`. xarray writes `zarr.json` itself there, so these need a seam, or the authority is held around them. The region `to_zarr` of the same function (`mode="r+"`, no append dimension, `consolidated=False`) writes only chunk payload: a traced region write put a data chunk and no `zarr.json` on xarray 2025.10.1 and 2026.7.0. It belongs to the adapter above, not to this list.
- `_consolidate_metadata_best_effort` in the same file, a separate metadata writer (it rewrites consolidated metadata).
- `_restore_group_attrs` in the same file, which rewrites the first-write group attributes after an append (`attrs.put`).
- State-array creation: `ensure_timestamp_state_array` in `src/firecube/core/zarr/state.py`, reached from `ensure_existing` in `append_services.py`; it creates the array and then writes its attributes in a second metadata write.
- The shrink in `_truncate_group` of `append_failure.py` (`repair_failed_group` resizes every append-dimension array back to the batch start; a fresh group is deleted). The authority has no shrink or delete operation, because DirectZarr never needs one; this one needs a decision on whether the authority gains it.
- `root.attrs["receipts"]` in the `write_lock` docstring example of `GenericZarrIngestor` (`src/firecube/ingestor/templates/generic.py`): a plugin hook writes root attributes from the main thread, outside any authority. Only the example is in the repository; plugin code that follows it is not covered.

---

### §35 Repo-wide cross-package private-import hardening

**Status**: OPEN
**Scope**: architecture invariant, static test
**Priority**: medium

### Motivation

Post-implementation review during PR #40 preparation found that `test_import_boundaries.py` protects the templates/plugins boundary but not the CLI, and that no test enforces "no cross-package underscore imports" anywhere in the tree. RF-12 (`test_cli_no_private_cross_subsystem_imports.py`, added by PR #40) closes the CLI-specific gap, but the underlying repo-wide invariant is still missing. AST scan of `src/` at PR #40 review time found several cross-subsystem private imports that should be triaged individually:

- `src/firecube/ingestor/runtime/telemetry.py:22` — `from firecube.core.observability.metrics import _to_number, normalize_run_summary`
- `src/firecube/ingestor/runtime/tensogram/strategy.py:28` — `from firecube.core.tensogram.converter import _find_time_dim`
- `src/firecube/core/zarr/validation.py` (the `_open_fsspec_url` import) — `from firecube.core.filesystem.ops import _open_fsspec_url  # type: ignore` (already an accepted deviation per DESIGN.md fsspec allowlist)
- `src/firecube/core/tensogram/converter.py:31` — `from firecube.core.filesystem.ops import _open_fsspec_url` (same — accepted deviation)
- Possibly others surfacing when the invariant is added

### Proposed approach

1. AST-scan `src/firecube/**/*.py` for `ImportFrom` nodes with `name` starting with a single underscore and `module` in a different first-level subpackage than the importing file (e.g., `firecube.ingestor.X` importing from `firecube.core.Y` is cross-subsystem; `firecube.core.zarr.X` importing from `firecube.core.zarr._Y` is same-subsystem).
2. Build a `_PERMANENT_ALLOWLIST` seeded from the AST scan, with an explicit rationale per allowed entry (mirror the `test_no_raw_fsspec_usage.py` pattern).
3. For each currently-allowlisted entry, decide: **promote** (rename to public), **accept** (document in DESIGN.md §"Accepted Deviations"), or **defer** (leave in allowlist with a TODO reference).
4. Reject any NEW cross-subsystem private import that is not in the allowlist.

### Not this branch

PR #40 closes the CLI-specific gap only. The repo-wide invariant, allowlist seeding, and individual triage are follow-up work.

### References

- `tests/architecture/test_cli_no_private_cross_subsystem_imports.py` — RF-12, the CLI-scoped precursor
- `tests/architecture/test_import_boundaries.py` — the existing templates/plugins invariant to model after
- `tests/unit/test_no_raw_fsspec_usage.py` — the canonical `_PERMANENT_ALLOWLIST` pattern

### §29 Test suite overhaul

**Goal:** Reduce noisy static/docs failures and raise behavioral bug-detection
power in the default test loop.

**Current state:**
- The default test loop must prioritize runtime behavior over static prose,
  broad snapshots, and implementation-shape checks.
- Test quality rules live in [TESTING_STANDARDS.md](TESTING_STANDARDS.md).
- High-risk missing behavior coverage lives in [TEST_GAPS.md](TEST_GAPS.md).
- Generic tests must use neutral sample product/plugin names unless the test is
  explicitly about an external plugin migration path.

**Direction:**
1. Keep the fast lane focused on behavior and failure modes.
2. Keep docs/static/help checks narrow, semantic, and outside the default
   behavioral signal when practical.
3. Replace mock-heavy or assertion-light tests with real behavior tests around
   deletion, workspace cleanup, storage operations, plugin management, and
   source readers.
4. Prioritize the P0/P1 gaps in [TEST_GAPS.md](TEST_GAPS.md).

**Acceptance criteria:**
- Default development and CI lanes are explicit and documented.
- A one-line help/prose diff does not fail the fast behavioral lane.
- Strict collection has no unexpected dependency skips.
- Every remaining xfail names an accepted TODO item and removal condition.
- The highest-risk control-plane and deletion bugs are covered by behavior
  tests, not only static checks.

---

### §8 Tensogram ingest routing

**Goal:** Clarify whether `GenericTensogramIngestor` is a public plugin base class, an internal write-path adapter, or both.

**Current state:**
- `GenericTensogramIngestor` exists in `src/firecube/ingestor/templates/generic_tensogram.py` and is exported from `firecube.ingestor.api`.
- `firecube ingest --output-format tensogram` currently accepts only plugins based on `GenericZarrIngestor`, then dynamically borrows `GenericTensogramIngestor._process_batch`.
- A plugin that directly subclasses `GenericTensogramIngestor` appears likely to be rejected by the current CLI guard because it is not a `GenericZarrIngestor` subclass.
- `GenericTensogramIngestor` currently supports only local `.tgm` targets.

**Decision (2026-06-11):** Tensogram is BOTH the archive format and a proper public output format, with the same first-class status as `GenericZarrIngestor`. `GenericTensogramIngestor` is a public plugin base class: direct subclassing must be supported and CLI-routable, the runtime `type()` class synthesis in `cli/main.py` must be replaced with explicit strategy/capability routing, and the local-only target restriction must be lifted via staged upload (reusing the `archive create` temp-file + transfer pattern). See `plans/AUDIT.md` (C4).

**Phase 0 — DONE (2026-06-11):** tensogram dependency upgraded `0.17.0` → `0.21.0` (all three packages, pins `>=0.21.0,<0.22`). See DONE.md 2026-06-11 for details, including the legacy-archive validate fix and the checked-in 0.17 compat fixture.

**Verified broken today (evidence for Phase 1, confirmed identical on tensogram 0.17 and 0.21 — pre-existing, not upgrade regressions):**
- `firecube ingest <plugin> --output-format tensogram --target file://.../out.tgm` fails with `[Errno 21] Is a directory`: the control-plane root (`.firecube/`) is created *inside* the `.tgm` target path before the strategy writes, turning the target into a directory. The synthesized-class path inherits zarr-oriented target preparation that is wrong for a single-file format.
- Archive restore loses time-coordinate encoding: CF `units`/`calendar` live in xarray `.encoding` and are never serialized by the converter, so restored cubes get a raw float64 ns-epoch time coord that xarray cannot decode back to datetime64. Restored variables also receive the raw base-entry dict (`name`, `zarr_chunks`, `zarr_compressor`, `zarr_fill_value`, nested `attrs`) as variable attrs — metadata pollution. Fix: serialize variable `.encoding` (at least time units/calendar) into the firecube message metadata and reapply on restore; map only real attrs onto restored variables. **Ownership: this is a firecube bug, not a tensogram one** — the metadata frame is free-form (room to store encoding), the engine's base-entry passthrough is fed by keys we put there, and our restore path already post-processes the decoded dataset. No upstream change required; optional upstream nice-to-haves (a variable-attrs convention in tensogram-xarray base entries, a logical datetime64 dtype hint in tensogram descriptors) are tracked in IDEAS.md, not blockers.

**Phase 1 — DONE (2026-06-11):** `DatasetProducer` protocol defined in `ingestor/contracts/`; CLI `type()` synthesis removed; direct `GenericTensogramIngestor` subclasses are now CLI-routable via protocol-checked strategy selection. `_GenericBatchIngestor` dissolved; `GenericZarrIngestor` and `GenericTensogramIngestor` re-parented directly onto `BaseIngestor`. Evidence: `tests/unit/test_tensogram_routing.py`, `tests/unit/test_template_hierarchy.py`. See DONE.md 2026-06-11 audit dispositions C3, C4.

**Phase 2 plan (separate feature branch) — OPEN:**
1. Fix single-file target handling: control-plane root must not be created inside a `.tgm` path (product-local control plane next to the file, or workspace-local) — resolves the `Is a directory` failure above.
2. Lift the local-only restriction via staged write: `.tgm` to the run workspace, upload through `StorageSession`/`create_filesystem` (extract the existing `archive create` remote pattern). `--write-mode staged` for remote targets; `direct` stays local-only with a loud error. One-driver invariant respected.
3. Control-plane parity: record spans with `SpanCoverage(write_strategy="tensogram", time_dim_name=...)`. Decide ResumeGuard semantics for an overwrite-only format (recommended: each run replaces prior spans for the same slice, `force_reingest`-like).
4. Restore fidelity: fix the time-encoding loss and attrs pollution described above; adopt `verify_hash=True` on decode/restore when frames carry `HASH_PRESENT` (0.21 feature; legacy frames skip with a note, mirroring the validate behavior shipped in Phase 0).
5. Config hygiene: implement or delete the unused `tensogram_message_granularity` field in `TensogramTemplateConfig`.
6. Public surface: scaffolding template, `tests/fixtures/tensogram_capable_test_plugin` (added to the conftest sessionstart guard), remote staged-upload integration test (moto), golden-help regen.
7. Docs + doctrine: output-format page under `docs/concepts/output-formats/` (via `write-user-doc`/`write-plugin-doc` prompts), DONE.md entry closing §8, AGENTS.md "Where things live" update.

---

### §5 Catalog and standards integration

**Goal:** Build a bridge from Firecube's internal view (Zarr + WAL-backed control-plane records) to higher-level discovery standards.

**Direction:**
- Starting from existing Intake support, explore emitting STAC or OGC EDR-style metadata as optional helpers.
- Keep this layer optional and generic. Firecube should not lock users into a single discovery stack.

---

### §7 Safe parallel ingestion patterns

**Goal:** Parallelize ingestion on Kubeflow/Argo safely without ACID/Icechunk complexity.

**Current state:**
- `AppendStrategy` (used by `GenericZarrIngestor`) serializes writes with a process-local lock. `pipeline_workers>1` parallelizes preprocessing but still writes one batch at a time.
- `IndexedRegionStrategy` (used by `DirectZarrIngestor`) writes pre-planned, disjoint index ranges via `RegionZarrWriter`. No appends. `IndexedRegionStrategy.write_groups()` auto-computes `expected_time_count` from `WriteIntent.ts_index`; undersized timestamped arrays hard-fail instead of resizing at write time. This is the building block for safe parallelism.
- `FilesystemClaimService` (`src/firecube/core/controlplane/claims.py`) provides exclusive write claims with heartbeat-based stale detection.
- T3 (§7-GENERIC) complete: `GenericZarrIngestor` now uses `WriteDomain(category="zarr_append", ...)` — the construction lives in `src/firecube/ingestor/runtime/zarr/batch_runner.py` since the §22 facade thinning, wired from `templates/generic.py`; integration fixtures updated in `tests/integration/test_maintenance_claims.py` and `tests/integration/test_obstore_claim_atomicity.py`. Code commit: `6329217`.
- T2 (§7-DIRECT) complete: `DirectZarrIngestor` now uses `WriteDomain(category="zarr_region", name=f"{group}:schema")` for schema setup and `WriteDomain(category="zarr_region", name=f"{group}:slot={ts_index}")` for per-slot intent dispatch via optional `claim_for_slot`. `IndexedRegionStrategy.write_groups()` groups intents by `ts_index` and uses fallback `claim_for_slot → claim_for_group → nullcontext`.

**Verified evidence (from review)**: `DirectZarrIngestor.claim_for_group()` used to build `WriteDomain(product=product, category="zarr_group", name=str(group_name))` — per-group granularity only. §7-DIRECT changed this to schema and per-slot `zarr_region` claims. The remaining safe-parallelism gap is the planner/orchestrator layer that assigns deterministic, disjoint index ranges before dispatch.

**Recommended safe model today:** Append writes serialize to one writer per `(product, group)`. DirectZarr writes may run concurrently only across pods with pre-planned disjoint slot/group ranges via the `firecube zarr slots` planner. Do not rely on intra-pod `pipeline_workers > 1` for write parallelism on DirectZarr templates: the per-slot claim has no retry loop, so any same-slot collision between in-flight batches fails hard (empirically reproduced 2026-07-12). For large per-slot payload plugins (~14.8 GiB/slot measured for the largest so far), scale via `pipeline_workers=1` × N disjoint-range pods; total cluster memory is unchanged but sized per-pod it fits standard nodes.

**§7-sub / Phase 3 planner — DONE (2026-05-28):** See DONE.md (section 7-sub) for full deliverables. Engine/template-level planner with deterministic time-to-index mappings, chunk-aligned ranges, `firecube zarr slots` JSON output, `firecube zarr preallocate` schema preflight, `--slot-start/--slot-end/--slot-size` CLI flags, K8s env discovery, 6-row ResumeGuard conflict matrix, and per-pod `run_id` derivation are all shipped. (These commands originally shipped as `firecube plan` / `firecube zarr setup-schema` and were later renamed.)

    **§7-Phase 3.1 hardening — DONE (2026-05-28):** See DONE.md §7-Phase 3.1 for full deliverables. Closes 5 safety gaps surfaced by external review: strict `global_expected` coverage enforcement, intent-group-in-schema hard fail, all-arrays chunk validation, `--slot-group` CLI flag + env var, group-aware ResumeGuard, slot-range-aware completed-span check, split bypass semantics (`resume_existing` vs `force_reingest`), and `--slot-group` propagation through capability gate + plan output. All 4 Final Verification reviewers approved. Commits: C1=94d167e, C2=4a5212e, C3=ba90260.
    **Phase 3.2 follow-up DONE (2026-05-29):** See DONE.md §7-Phase 3.2 for full deliverables. Closes 6 review issues: run_id slot_group isolation, strict schema validation via SchemaDriftError + verify_array_spec, per-group slot_size in firecube plan, _ParallelExecutionState + ctx._ctx escape removal, pod-startup schema verification + audit record, docs drift cleanup. All 4 Final Verification reviewers approved. Commits: C1=37f7521, C2=80583a2, C3=84bb1bb, C4=6b6f3a1, C5=450f8f7, C6=416a4b6.
    **Phase 3.3 external-review follow-up DONE (2026-05-29):** See DONE.md §7-Phase 3.3 for full deliverables. Closes 6 external-review issues: sequential slot filter parity, terminal partial chunk + plan-to-ingest contract, URL-encoded slot_group in run_id + WAL reader, phantom group prevention in capability gate + pod-startup, operator docs refresh, sharding test assertion strengthening. All 4 Final Verification reviewers approved. Commits: C1=9fce649, C2=545689f, C3=40977ca, C4=8fc455d, C5=c6fd238, C6=269acdf.
    **Phase 3.4 external-review follow-up DONE (2026-05-30):** See DONE.md §7-Phase 3.4 for full deliverables. Closes 3 additional review issues: phantom global_expected validation added to the planner CLI preflights (now `firecube zarr slots` + `firecube zarr preallocate`), warn_if_misaligned terminal-partial awareness, sharding test spy on xr.Dataset.chunk. All 4 Final Verification reviewers approved. Commits: C1=604a4d8, C2=a9fc6e8, C3=adf41dd.
    **Phase 3.5 external-review follow-up DONE (2026-05-30):** See DONE.md §7-Phase 3.5 for full deliverables. Closes 2 additional review issues: firecube plan fail-closed on blocked partial-chunk ranges (silent slot-loss bug fixed), zarr_schema() called exactly once per plan invocation. All 4 Final Verification reviewers approved. Commits: C1=2992425, C2=8fda71a. Phase 3.6 follow-up DONE 2026-05-30 (closes 1 additional review issue). Phase 3.7 follow-up DONE 2026-05-30 (closes 2 additional review issues: CLI error remediation text + operator docs refresh). Phase 3.8 follow-up DONE 2026-05-31 (closes 2 additional review issues: delete-span remediation now includes --product/--force/full safety flow + regression test strengthened to lock runnable shape).

    **Builds on:** §4a (DONE.md) — `IndexedRegionStrategy` is the missing primitive that enables disjoint-region writes. §21, §23, §23-AUTO, §7-DIRECT, §7-sub, and Phase 3.1 are all DONE as of 2026-05-28. Safe within-group parallelism is fully delivered for `DirectZarrIngestor` plugins.

---

### §37 `compare_zarr_stores` chunk-wise value comparison (post-2026-08-24)

`src/firecube/core/zarr/validation.py::_values_equal` currently loads whole arrays into memory via `left[:]`. For production-size cubes this will OOM. If `firecube zarr compare` is to be used as a promotion gate on real cubes, replace the full-array load with chunk-wise comparison that iterates over chunk indices and compares one chunk at a time.

**Scope**:
- Modify `_values_equal` to iterate chunks (use `zarr.Array.chunks` and `numpy.ndindex` over chunk coordinates).
- Preserve NaN-aware equality semantics per dtype kind.
- Preserve the terse per-category mismatch message format (do NOT list individual chunks).
- Add a test with a large synthetic array that would OOM with `[:]` but passes with chunk-wise comparison.

**Deferred from**: Milestone C review, 2026-08-25. V1 boolean-only compare is fine for small stores.

---

### §38 `zarr validate` static-marker check: configurable time-dimension names (post-2026-08-25)

`src/firecube/cli/zarr/_validate.py::_static_marker_failures` classifies an array as
time-indexed when its first dimension is named `timestamp`, `time`, or
`firecube_timestamp_state`; any other first-dimension name is treated as
static and must carry the `firecube_static_written` marker. A plugin using a
custom `time_dim_name` outside that set gets false
`missing_or_false_static_marker` reports on genuinely time-indexed arrays.

**Scope**:
- Add a way to extend or override the recognized names (a `--time-dim` option
  on `firecube zarr validate`, or read the dimension name from the resolved
  index record).
- Keep the default set unchanged.

**Deferred from**: Wave C2 review, 2026-08-25. The recognized set is
documented in `docs/concepts/performance.md`.

### §9 Span planning (optional pre-declared spans)

**Goal:** Make spans a stable unit of work for orchestration and maintenance without making ChunkManager product-aware.

**Current state:** Spans are recorded post-hoc from write coverage (`span.time_index_ranges` + `meta.time_min/time_max`).

**Next step:** Optional SDK hook for "span intent":
- Plugins/templates may pre-declare spans (month windows or chunk-aligned index ranges).
- Engine records `status="started"` spans at begin and finalizes to `complete/failed/noop` with actual coverage at end.

**Builds on:** §4a (DONE.md) — Write strategies now produce explicit write intents; pre-declared spans become a natural extension where templates declare spans before executing strategies.

---

### §11 Multires as a post-step — DONE (2026-05-26)

See DONE.md §25 for details.

**Current state (post-§25):** The post-step CLI `firecube zarr multires <target> --storage-type <local|s3> --storage-driver <fsspec|obstore>` exists and wraps `ZarrMultiresBuilder`. Ingest-time multires has been removed: `AppendMultiresHandler` no longer exists (deleted in §25, commit `89b2737`). The `zarr_multi_res` config field is rejected with a `ValueError` pointing to the CLI subcommand.

**Direction:** Use `firecube zarr multires` as a post-step after ingestion completes. No ingest-time multires path remains.

**Builds on:** §4a (DONE.md) — Zarr is now a runtime subsystem with explicit finalize/commit boundaries. §25 completed the removal of the broken ingest-time path.

---

## Harden Implicit Logic and Heuristics

These items identify areas where the system used greedy logic to guess user intentions. The goal is to make these explicit or more robust. Status re-verified against the code on 2026-06-11.

### §12.2 CLI and Execution Engine

- **BASENAME heuristics — DONE:** `output_name` is no longer guessed from the target path basename. `ProductIdentity.from_uri` hard-fails without an explicit product name, and the `default_output_name` config key is rejected at parse time.
- **Magic output detection — DONE:** the CLI and engine read typed `PipelineResult.outputs` / `result.output_path` attributes; no dict key sniffing remains. The legacy `output_path=` constructor kwarg was removed entirely (DONE.md 2026-06-11).
- **ingest `dry_run` — OPEN:** implement or remove the option. `EngineConfig.dry_run` only sets `RuntimeFlags.dry_run`; no write path reads it, so `firecube ingest --option dry_run=true` writes normally. The docstring marks it reserved and points at the maintenance commands' `--dry-run` for previews (2026-09-12).
- **Explicit safety — OPEN:** refusal to upload `HOME` or `/` is hardcoded in the engine's upload-source resolution; should be documented or configurable.
- **Free-form option overload — DONE (2026-06-11):** keys owned by dedicated `firecube ingest` flags (`write_mode`, `slot_start`, `slot_end`, `slot_size`, `slot_group`) are hard-rejected at `--option` parse time with remediation naming the owning flag (`_TYPED_FLAG_OWNED_KEYS` in `cli/_typed_options.py`); the silent post-resolution override is closed. All other typed flags were never config fields and were already rejected as unknown keys. Engine options without a dedicated flag (`force_reingest`, `no_progress`, ...) remain the sanctioned `--option` surface.

### §12.3 Configuration Derivation

- **Automatic env resolution — PARTIALLY FIXED:** `${VAR}` expansion is now scoped to `[storage]` values only (not all config strings), but remains unconditional within that scope — a literal `${FOO}` storage value is impossible when `FOO` is set. Remaining work: opt-in flag or escape syntax.
- **DuckDB option tier is disconnected — OPEN:**
  `src/firecube/core/config.py:get_plugin_defaults()` deliberately does not
  merge `[database.duckdb]`, while
  `src/firecube/ingestor/runtime/configure.py:TierConfigurator` allowlists
  `duckdb_max_temp_directory_size`, `duckdb_memory_limit`, and
  `duckdb_threads` only after they are already present in `ctx.options`.
  `src/firecube/cli/_typed_options.py:_enumerate_all_valid_keys()` does not
  include that tier, so `--show-options` omits the keys and typed `--option`
  rejects them. `DuckDbMixin` in
  `src/firecube/ingestor/extensions/duck.py` therefore has no coherent public
  route for receiving its resource settings. Define one typed owner for these
  fields, load `[database.duckdb]` only for plugins that declare the DuckDB
  capability, and expose the same keys through CLI discovery and coercion.
  Acceptance: config-file and `--option` values reach `DuckDbMixin` with typed
  values; `--show-options` lists the keys only for compatible plugins; plugins
  without the capability reject them; public configuration docs describe the
  same surface.
- **`default_product_name` leaks into plugin option validation — OPEN:**
  `src/firecube/cli/main.py` reads the key from plugin defaults to resolve the
  product name, but leaves it in the `options` mapping passed to
  `IngestContext`. `TierConfigurator.configure()` in
  `src/firecube/ingestor/runtime/configure.py` then rejects it because
  `SYSTEM_KEYS` in `src/firecube/ingestor/config/engine.py` does not include the
  key and `PluginConfig` in `src/firecube/ingestor/types/config.py` does not own
  it. Consume this CLI-owned key before tier validation; do not weaken strict
  unknown-key rejection. Acceptance: a config-only `default_product_name`
  completes runtime configuration, while product-name precedence remains CLI
  flag > config `default_product_name` > plugin `PRODUCT_NAME` > hard failure.

### §12.4 Plugin Heuristics

- **Option aliases — DONE (2026-06-11):** `zarr_chunk` deleted (no shim, per STYLE.md); `zarr_chunk_shape` is the single chunking option and `zarr_chunk` now fails strict unknown-key rejection. Locked by `tests/unit/test_zarr_chunk_alias_removed.py`.
- **Regex guessing — FIXED in-repo:** no filename-regex horizon extraction or `F*` folder discovery remains in core or templates. Remaining occurrences live in an external plugin repository.
- **Hardcoded defaults — DONE (2026-06-11):** lat/lon soft limits gone; multires `(1.0, 0.5)` single-sourced as `DEFAULT_MULTIRES_RESOLUTIONS` (no silent fallback); `group="FWI"` fallback removed from `core/zarr/layers.py`; `"fire_risk.duckdb"` default removed from `extensions/duck.py`. Evidence: `tests/unit/test_domain_defaults_removed.py`.
- **Typed-vs-free-form drift — DONE (2026-06-11):** strict unknown-key rejection enforced on all declared typed configs; `x_*` experimental namespace implemented — keys matching `x_*` pass through without rejection. Evidence: `tests/unit/test_experimental_options.py`.
- **Discovery exclusions — DONE (2026-09-13):** Native `--input-filters` combines additive globs and `!` exclusions in one list on ingest, slots, and preallocate. Configuration uses `input_filters`; `include_patterns` is rejected with migration guidance. CLI introspection and the discovery guide show the native flag. Evidence: `tests/unit/test_input_filters.py` and `tests/integration/test_input_filters_cli.py`.
- **Other discovery controls — DEFERRED:** `include_suffixes`, `recursive`, and `sniff_hdf5` remain Python helper arguments. No additional CLI flags or engine options are selected for implementation. Custom discovery hooks retain responsibility for applying filters to their source items.

---

### §13 Harden PluginContext boundary

**Goal:** Make plugin hook context truly read-only and non-bypassable in normal usage.

**Current state (partially addressed):**
- `options` is a detached copy wrapped in `MappingProxyType`. Mutations to `RuntimeIngestContext.options` after `PluginContext` creation are invisible to plugins.
- `option()` reads from the frozen proxy. `ctx.options["k"]` and `ctx.option("k")` are consistent.
- `__getattr__` blocks direct access to `storage`, `_chunk_manager`, `_materializer`.
- 6 boundary tests in `tests/unit/test_plugin_context_boundary.py`.

**Remaining (structural):**
- `_ctx` escape hatch: `pctx._ctx` still gives full access to `RuntimeIngestContext` (see `PluginContext.__init__` in `src/firecube/ingestor/types/context.py`). Python cannot enforce true private attributes, but a frozen-snapshot design would eliminate the reference entirely.
- **Phase 3.2 partial closure**: The `_parallel_global_schema` escape hatch (the specific `ctx._ctx._parallel_global_schema` pattern used by `DirectZarrIngestor`) was removed in Phase 3.2 C4 (commit `6b6f3a1`). The broader `_ctx` reference-sharing issue remains open.
- Frozen snapshot blocked by initialization order: `PluginContext` is created in `BaseIngestor.run()` (`src/firecube/ingestor/runtime/base.py`) before `runtime_ctx.telemetry` is assigned a few lines later. Fixing this requires restructuring `run()` so telemetry is assigned before context construction.
- Per-worker snapshots: all workers share one `PluginContext` instance wrapping the same `RuntimeIngestContext`. Safe for reads (options detached), but `_ctx` reference-sharing means a convention-breaking plugin could mutate shared state.
- No tests assert `_ctx` is inaccessible (Python cannot enforce this without structural change).

---

### §31 Parquet template configuration parity

**Goal:** Eliminate accepted Parquet configuration keys that silently have no
effect on persisted output.

**Current state:** `ParquetTemplateConfig` in
`src/firecube/ingestor/templates/config.py` declares `parquet_partition_by` and
`parquet_row_group_size`. `GenericParquetIngestor.write_parquet()` in
`src/firecube/ingestor/templates/generic.py` calls `pyarrow.parquet.write_table`
without applying either value. Both keys therefore pass strict configuration
validation but do not change the output.

**Direction:**
1. Pass `parquet_row_group_size` to the default writer and validate invalid
   values before the first write.
2. Remove `parquet_partition_by` from the accepted config until the template has
   an explicit dataset-directory layout, deterministic part naming, write-domain
   ownership, and resume semantics for partitioned output. Do not emulate
   partitioning inside the current one-file-per-batch contract.
3. Keep `docs/reference/config.md` and
   `docs/guides/plugins/generic-parquet.md` aligned with the implemented surface.

**Acceptance criteria:**
- Every accepted `ParquetTemplateConfig` field changes persisted Parquet output;
  unsupported fields fail as unknown configuration.
- Row-group behavior is verified from real Parquet metadata for local output and
  through the selected storage driver for remote output.
- No plugin needs to override `write_parquet()` to use the advertised row-group
  setting.

---

### §32 Public output-writer contract for custom pipelines

**Goal:** Let an external `BaseIngestor` plugin write through Firecube's selected
storage driver using a stable, typed public contract.

**Current state:** `StorageContext` is exported by
`src/firecube/ingestor/api.py`, and `PluginContext.storage.output` exposes the
bound session at runtime. Its annotation in
`src/firecube/ingestor/types/context.py`, however, names the concrete
`StorageSession` from `src/firecube/core/storage/session.py` under
`TYPE_CHECKING`. That concrete class is not exported by
`src/firecube/ingestor/api.py` or `src/firecube/core/api.py`. A custom
`BaseIngestor._process_batch()` implementation can therefore use the object only
through an internal import, implicit duck typing, or `Any`; none defines a
supported plugin contract.

**Direction:** Define a narrow writer `Protocol` in the plugin contract layer,
type `StorageContext.output` against it, and export it from the appropriate
public `api.py` surface. Keep the concrete `StorageSession` internal. The
protocol must expose only operations required by custom output pipelines and
must preserve the one-driver rule; it must not provide a route around Firecube's
write coordination or control-plane ownership.

**Acceptance criteria:**
- An external `BaseIngestor` plugin can type-check and perform its output writes
  using imports from `firecube.ingestor.api` and `firecube.core.api` only.
- The public protocol has behavior-backed local and S3 driver coverage for its
  declared operations.
- Existing template implementations continue to use the same bound session and
  selected driver without deep imports becoming part of the plugin API.
- Public plugin documentation names only the protocol, not
  `firecube.core.storage.session.StorageSession`.

---

### §14 Expand concurrency and race-condition coverage — DONE (2026-06-11)

**Goal:** Catch production-only failures before deployment.

All four open points closed. Evidence:
- Concurrent ingestor instances (same product, `resume_existing`/`force_reingest` combinations): `tests/integration/test_concurrent_same_product.py`
- Workspace materialization races on the same remote URI: `tests/integration/test_workspace_materialization_race.py`
- Control-plane write ordering/consistency under concurrent batch completion: `tests/integration/test_wal_concurrent_ordering.py`
- OTel/thread context propagation under error and retry scenarios: `tests/integration/test_otel_context_concurrency.py`

---

## Zarr Runtime Follow-ups

These items address gaps and bugs surfaced after §4a landed. They are sequenced as refinements/fixes to work that has already shipped, not as new architectural directions. Items §25 and §26 are runtime-crashing bugs and should be prioritised.

### §21 Strategy Protocol unification — DONE (2026-05-27)

See DONE.md §21 for details.

Follow-up: Check external plugins for `ZarrWriteStrategy` import and migrate — out of scope for Phase 1.

---

### §22 GenericZarrIngestor full thinning — DONE (2026-05-27)

**Goal:** Move `GenericZarrIngestor._process_batch` toward a true thin facade.

**Original state (pre-§22):** `_process_batch` was 144 lines and owned five responsibilities inline: staged metadata seeding, lock construction, strategy construction, claim closure construction, and metrics assembly.

**Completed (2026-05-27):** §22 helper extraction is done in `src/firecube/ingestor/runtime/zarr/batch_runner.py` with 5 named helpers and `tests/unit/test_batch_runner.py` (13 tests). T5 wired those helpers into `GenericZarrIngestor._process_batch`, replacing all five inline blocks.

The core write step remains delegated to `AppendStrategy.write_groups()`.

**Result:** `_process_batch` is now a thin orchestrator around preparation, URI/option resolution, batch_runner helper wiring, `strategy.write_groups()`, and final result construction. ~66 lines (down from 143). See DONE.md §22 (helpers) and §22 (wiring complete).

**Builds on:** §4a (DONE.md).

---

### §23 RegionZarrWriter resize-on-write contradicts documented contract — DONE (2026-05-27)

See DONE.md §23 for details.

### §23-AUTO Auto-compute expected_time_count from WriteIntent.ts_index — DONE (2026-05-27)

See DONE.md §23-AUTO for details.

---

---

### §25 AppendMultiresHandler stale kwarg — DONE (2026-05-26)

See DONE.md §25 for details.

Follow-up: Check external plugins for `zarr_multi_res` usage and update if needed — out of scope for this PR.

---

### §27 Stale `build_dataset` documentation — DONE (2026-05-26)

See DONE.md §27 for details.

---

### Phase 2.1 Legacy zarr_group claim sweep removal — DONE (2026-05-28)

`_sweep_legacy_zarr_group_claims` removed in Phase 3 (T18). See DONE.md §7-sub.

---

### §30 `ensure_timestamp_slot` should treat `time_indexed=False` arrays as coords — DONE (2026-06-25)

**Goal:** Stop static (non-time-indexed) coordinate arrays from being mistaken for time axes during the time-slot bounds check, so declaring `time_indexed=False` is sufficient on its own.

**Current state:** `RegionZarrWriter.ensure_timestamp_slot` skips an array from the `ts_index < shape[0]` bounds check only if its name is in `coord_names` (constructor param, default `{"y","x","channel"}`) or it is scalar (`ndim == 0`). It does **not** consult the schema's `ZarrArraySpec.time_indexed=False` flag. The strategy builds `coord_names_by_group` from `spec.coord_names` only.

**Bug it caused:** A DirectZarr plugin declared static 2-D `lat`/`lon` and 1-D `ny`/`nx` projection coords with `time_indexed=False` but did not also list them in `coord_names`. `ensure_timestamp_slot` then read each static coord's dim-0 (the grid size, e.g. 1072/1332) as a time axis and raised "ts_index out of bounds" for any `ts_index` beyond it — i.e. any day far from the reference epoch (slot ~622080 for a 2023 day against a 2018 epoch). Latent for any multi-day ingest; was masked earlier by the axis-size error. The plugin worked around it by adding the names to `coord_names`.

**Direction:** Fold every `time_indexed=False` array into the writer's coord-skip set (the schema already knows which arrays have no time axis), as a union with the existing `spec.coord_names`. Then `time_indexed=False` alone is sufficient and a plugin never has to *also* name the array in `coord_names`.

**Acceptance criteria:**
- A DirectZarr group whose static coords are declared only via `time_indexed=False` (not in `coord_names`) ingests a high-`ts_index` timestamp without a spurious "ts_index out of bounds" error.
- `coord_names` remains honored for back-compat (union, not replacement).

**Surfaced by:** a plugin fix declaring `lat`/`lon`/`ny`/`nx` in `coord_names`.

See: DONE.md "DirectZarr plugin parity — core fixes" (2026-06-25).

---

## Parquet slice identity and replay

The fresh-target safety fix deliberately limits `GenericParquetIngestor` to fresh targets. Do not remove the target guard merely after adding run IDs to part names. Before supporting resume or force-reingest, define a stable, plugin-declared logical slice identity independent of batching and temporary paths; specify physical part ownership, replay deduplication, exact-slice replacement, and failure-safe promotion/recording through `ChunkManager` and the selected driver. Tests must verify rows and active spans across disjoint runs, repeated inputs, changed batch sizes, partial replacement, and failed promotion. The current target guard is covered by `tests/integration/test_parquet_target_safety.py`.

## Zarr migration framework (deferred from CF-1.8 plan)

**Goal**: a generic Zarr migration framework so future migrations (rename-dim, rename-array, set-default-attrs, schema upgrades, etc.) plug in via a Protocol implementation rather than each shipping its own CLI command. Decided against `migrate-dim` and similar name-coupled one-offs.

**Shape**: `MigrationStrategy` Protocol (`@runtime_checkable`) + in-source registry + a generic safety runner.

**Generic runner responsibilities**:
- Refuse to run if `ChunkManager.list_runs(non_terminal=True)` returns any non-terminal runs.
- Refuse to run if `ChunkManager.list_claims()` returns active claims on target groups.
- Acquire and release a write claim around the migration.
- Provide `--dry-run`.
- Group-explicit `--group <g>` or `--all-groups`.
- Use `ChunkManager` facade + `create_filesystem(config)` — never raw `.firecube/` writes or `fsspec.filesystem(...)`.

**CLI shape**: `firecube zarr migrate <strategy-name> [strategy-args] [--dry-run] [--group <g> | --all-groups]`, with `firecube zarr migrate --list` and per-strategy `--help`.

**First strategy to ship**: `rename-dim` (the operation the CF-1.8 plan would have shipped). Required behavior:
- Walk every Zarr array in the target.
- Rewrite `dimension_names` in each array's metadata.
- Rename the coordinate variable folder (`{group}/timestamp` to `{group}/time`).
- Rebuild consolidated metadata if present.
- Preserve internal control-plane array names (e.g. `firecube_timestamp_state` stays — only its inner dim name changes).

**Out of scope for v1**: multi-resolution pyramid groups; `--force` to bypass safety; auto-migration on append.

**Acceptance criteria for the follow-up plan**:
- Compatibility matrix tests for at least the 8 rows defined in `cf18-compliance` plan.
- E2E `legacy cube -> migrate -> append with new dim -> readback` test.
- Integration tests covering refusal-on-active-claim and dry-run-no-mutation.

---

## CF advisor enhancements (deferred from CF-1.8 plan)

- `firecube advise compliance --profile cf-18 --all-groups`: walk all Zarr groups in a product instead of requiring `--group`.
- Tier 2 — CF Standard Name Table vocabulary validation (check `standard_name` values against the canonical table).
- Tier 3 — UDUNITS-based unit validation. Likely via optional `cf-checker` runtime dependency or a NetCDF temp-export bridge.

---

### Concurrent-metadata-read follow-ups (deferred from the 2026-06-22 `read_bytes` fix)

The 2026-06-22 `StorageFilesystem.read_bytes` change fixed the s3fs 412 crash on the
existing-cube dim-compat read. Two optional hardening items were deliberately NOT
bundled (they change a safety check's timing / touch a separate path):

- **Reduce redundant dim-compat reads.** `_verify_existing_cube_batch_groups` runs
  the dim-compat check once *per batch* (`engine._create_batches_with_parallel_filter`),
  so a pod re-reads the same group `zarr.json` many times. Cache the result per-pod,
  and/or gate the check in parallel slot-range mode where `_verify_schema_at_pod_startup`
  already guarantees the dims from the plugin-declared `time_dim_name`. Care: this
  changes when a public safety check (`verify_dim_compatibility`, DONE.md 2026-06-20)
  runs — preserve the genuine pre-existing-cube append guard.
- **Route `core/zarr/validation.py::_load_array_metadata` through `read_bytes`** for
  consistency (still uses `open().read()`). Lower priority — it's the `firecube zarr
  validate` maintenance path, not the concurrent ingest hot path.

---

### §28 Audit closeout follow-ups (minor)

- **Dead `plugin` marker** — registered in `pyproject.toml`, applied zero times. Decide: delete the registration or apply it to plugin-specific suites.
- **T4 residual** — `tests/unit/test_append_strategy.py` still patches the strategy's own internals; real-store coverage lives in `tests/integration/test_append_strategy_behavior.py`. Candidate for a behavior-based rewrite of the remaining wiring test.
- **A7 watch-item** — the archive-create path (`src/firecube/core/tensogram/converter.py`, `_find_time_dim` call around line 112) does not thread a plugin-declared `time_dim_name`; only the ingest path does.
- **C3 watch-item** — the hierarchy flatten grew `BaseIngestor` to ~815 lines (flatten-by-hoisting); the depth test pins templates only, not plugin-subclassing-plugin chains. Watch for god-base growth; consider extracting batch-lifecycle hooks into a composed collaborator if it keeps growing.

---

### §46 DirectZarr template-level codec drift is not checked on resume (2026-10-14)

**Status**: OPEN

`verify_array_spec` in `src/firecube/core/zarr/region_writer.py` compares stored codecs only with the codecs a `ZarrArraySpec` declares; a spec that declares none is never compared. Codecs that come from the template's `zarr_codecs` therefore escape the check. Run on 2026-10-14: `_setup_global_zarr_schema` over one store with `zarr_codecs` zstd level 1, then again with level 7 and no per-array declaration, raised nothing and the stored array kept level 1. Decide whether a changed template-level codec on resume is drift (refuse with `SchemaDriftError`) or is allowed and documented; either way keep the intended compatibility of a plugin that declares no codec at all, which must still append to an existing store (run: a spec with no codec was accepted against a stored zstd level 1). The schema hash is an audit fingerprint and does not enforce this; drift errors come from `verify_array_spec` for declared codecs.

### §47 GenericZarr discards the filters and the serializer of `zarr_codecs` (2026-10-14)

**Status**: OPEN

`_build_zarr_encoding` in `src/firecube/ingestor/runtime/zarr/write.py` keeps only the compressors of the validated `zarr_codecs` pipeline and passes `compressors` alone to xarray. Run on 2026-10-14: a `ZarrTemplateConfig` that validated `bytes(endian=big)` then `zstd(level=3)` was written as `bytes(endian=little)` then `zstd(level=3)`. Forward the full validated pipeline (filters, serializer, compressors) in `GenericZarrIngestor`. GenericZarr only; DirectZarr forwards all three.

### §48 Document resume-time codec drift and the migration choice (2026-10-14)

**Status**: OPEN

Neither `docs/guides/plugins/configure-zarr-compression.md` nor `docs/reference/config.md` (nor the `ZarrTemplateConfig` docstring that generates its entry) says what a resume does when the codecs change. Document, for a reader who already has a cube: a resume that declares different codecs for an array raises `SchemaDriftError` naming the field; how to keep an existing cube's codecs on append (declare the same codecs, or none); and how to re-ingest from scratch under other codecs, since no in-place codec migration exists (the error says so). Describe the template-level case as §46 settles it.

---

### §44 `delete-span` cannot mark deleted integer variables that have no fill value (2026-10-05)

**Status**: OPEN

`delete-span` fills a deleted span with the array's Zarr `fill_value`. An integer data variable with no CF `units` ... `since` and no `_FillValue` (for example a `record_status (time)` `int64` array) is therefore filled with `0`, which cannot be told from data. Options: encode a `_FillValue` when such a variable is written, or make `delete-span` warn when it zero-fills an integer array with no declared fill. Only CF time arrays get a NaT sentinel today.

### §45 `delete-span` region-fill dry run skips the pre-flight (2026-10-05)

**Status**: OPEN

For region-filled spans, `--dry-run` in `DeletionEngine._delete_span_by_region_fill` returns before `delete_span_via_region_nan_fill` runs. It can report success for a span that execution refuses (missing array, out-of-range, unknown dimensions, unfit state array) and omits the skipped-static-array warning that execution emits. Run the pre-flight in dry-run mode and return its warnings.

---

### Ingest performance

- Report real per-run byte and file counts for direct-S3 completion via filesystem instrumentation instead of returning zeros. Follow-up to the direct-S3 completion fix in 0.1.7; requires wiring per-run counters into the completion path.
