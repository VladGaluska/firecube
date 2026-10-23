# Firecube Test Suite

This directory contains Firecube's pytest suite. Before adding or reshaping
tests, read:

- [AGENTS.md](../AGENTS.md)
- [plans/TEST.md](../plans/TEST.md)
- [plans/TESTING_STANDARDS.md](../plans/TESTING_STANDARDS.md)
- [plans/TEST_GAPS.md](../plans/TEST_GAPS.md)

`plans/TEST.md` defines commands, markers, dependency checks, and skip policy.
`plans/TESTING_STANDARDS.md` defines what tests are worth adding or keeping.
`plans/TEST_GAPS.md` tracks missing high-risk behavior coverage.

## Structure

```text
tests/
├── architecture/   repository boundary and static invariant tests
├── cli/            CLI contract, semantic help, and command matrix tests
├── fixtures/       in-tree fixture plugins and test data
├── helpers/        shared test helpers
├── ingestor/       ingestor architecture and configuration tests
├── integration/    cross-component and end-to-end behavior tests
├── sdk/            public API and plugin contract tests
├── unit/           focused unit and component tests
└── test_*.py       root-level regression tests
```

Do not add new tests tied to removed external plugins unless they are explicit
fixture plugins installed from `tests/fixtures/`.

## Setup

Install test dependencies and the fixture plugins before running the suite:

```bash
uv sync --extra test
uv pip install -e tests/fixtures/firecube_test_plugins
```

That single editable distribution installs every fixture module and entry point.

## Common Commands

Every lane command (fast, docs/static, S3, slow, full run, race) is stated once,
in [plans/TEST.md](../plans/TEST.md#lanes). Do not copy them here. Two shortcuts:

```bash
# Local full suite (pyproject addopts excludes the race-marked tests)
uv run pytest -q

# Single file
uv run pytest tests/unit/test_chunk_manager.py -q
```

## Calendar Time Axes

Non-Gregorian calendar behavior (360_day, noleap) has no test directory of
its own. Each test lives with the unit it exercises:

- `unit/`, calendar names and axis declaration: `test_time_encode.py`,
  `test_time_axis_declaration.py`.
- `unit/`, resolution: `test_index_resolve_irregular.py`.
- `unit/`, coordinate materialization: `test_materialize_regular_coord_array.py`.
- `unit/`, relabel refusals: `test_coord_rematerialization.py`.
- `unit/`, `chunks list` and `chunks delete` refusals: `test_chunks_list_cli.py`,
  `test_cli_chunks_delete.py`.
- `unit/`, also: hashes, persisted payload, decode, writer, startup, archive,
  append order and coverage, chunk queries.
- `integration/`: preallocate and ingest round trips
  (`test_preallocate_regular_axis_dense.py`, `test_irregular_axis_pipeline.py`)
  and append over a February 30 span (`test_cf_time_dim_value_dedup.py`).
- `cli/`: the `archive create` refusal (`test_archive_create_contract.py`).

Three fixture plugins in `tests/fixtures/firecube_test_plugins` drive them.
Pass the calendar as an option:

| plugin name | options |
|---|---|
| `regular_axis_calendar` | `--option calendar=360_day`, `--option slot_count=90` |
| `irregular_axis_calendar` | `--option calendar=360_day` (five fixed values) |
| `cf_time_dim_calendar` | `--option x_calendar=360_day`, `--option x_start=2049-01-29` |

## Markers

Markers are defined in `pyproject.toml`; use only registered markers.

- `unit`: pure unit tests, no external I/O or services
- `integration`: multiple components together
- `architecture`: repository architecture and boundary invariants
- `contract`: public SDK, plugin, CLI, or persisted-format contracts
- `docs_static`: documentation, help-text, and static prose drift checks
- `snapshot`: reserved for deliberately approved narrow golden snapshots
- `plugin`: specific ingestor plugin behavior
- `s3`: live or mocked S3 endpoint coverage
- `slow`: long-running tests
- `concurrency`: parallel or concurrent behavior
- `race`: red-by-design legacy race characterization; excluded by default, run with `-m race`
- `gate`: promoted regression gates that must stay green

Markers select lanes; they do not make weak tests acceptable.

## Adding Or Changing Tests

New tests must protect a named behavior and risk. Prefer real filesystem/Zarr
objects and fixture plugins over mocks. Static scans and broad snapshots are
allowed only when they protect a current invariant and belong in the correct
lane. Broad prose policing, generated-site checks, and repository-wide example
scans do not belong in the default confidence path. See
[plans/TESTING_STANDARDS.md](../plans/TESTING_STANDARDS.md) for the review
checklist.
