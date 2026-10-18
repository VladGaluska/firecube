# Restore Archives

Use `firecube archive restore` to restore a `.tgm` or `.zip` archive back into
a Zarr store.

```bash
ARCHIVE_URI="file:///data/archives/MY_PRODUCT.tgm"
RESTORED_URI="file:///data/restored/MY_PRODUCT.zarr"
```

## Restore To Zarr

```bash
firecube archive restore \
  --archive "$ARCHIVE_URI" \
  --target "$RESTORED_URI"
```

Expected output resembles:

```text
restored group: data
Zarr restored: file:///data/restored/MY_PRODUCT.zarr
  Groups: data
  Variables: data
  Coordinates:
  Control-plane: restored
```

## Restore A Zip Archive

```bash
firecube archive restore \
  --archive "file:///data/archives/MY_PRODUCT.zarr.zip" \
  --target "$RESTORED_URI"
```

The target must be local. The output is the same as for `.tgm`, without the
`restored group:` lines.

Firecube refuses a zip entry that leaves the target. Such an entry has an
absolute name, a backslash, or an empty, `.` or `..` segment. The check runs
before Firecube deletes the target or restores any entry.

Firecube also refuses a corrupt entry, an archive without a root `zarr.json`,
and a `zarr.json` entry that is not valid Zarr v3 metadata. If a check fails,
Firecube deletes and replaces no existing data. It can already have written the
control-plane records of the restore.

## Verify The Restored Product

Inspect the restored product's run history:

```bash
firecube chunks runs list --product-name "$RESTORED_URI"
```

Expected output resembles:

```text
Run ID                               Status    State   Parts Events
------------------------------------------------------------------------
archive-restore-3e67c307-...         complete  active  4     4
direct_zarr_capable_test_plugin-...  complete  active  1     7
```

Inspect restored ChunkManager records:

```bash
firecube chunks list --product-name "$RESTORED_URI"
```

## Overwrite An Existing Restore Target

`--overwrite` replaces the whole target. Data and run history written to the
target after the archive was created are lost.

If the archive fails its checks, Firecube refuses it. It deletes and replaces
no existing data. It can already have written the control-plane records of
the restore. These are its claim, its run under
`.firecube/runs/archive-restore-*/`, and `.firecube/schema.json` if that file
was absent.

On a local target, the overwrite keeps these paths under `.firecube/`:

- `claims/`, which holds the claim of the restore
- `schema.json`
- `runs/<run-id>/`, which holds the records of the restore itself

Firecube deletes the rest of the old control plane and all old data. The
restore releases its claim when it ends. If the target or a kept path is a
symlink, Firecube refuses the overwrite and deletes nothing. Firecube treats a
remote `.tgm` target as before.

Preview:

```bash
firecube archive restore \
  --archive "$ARCHIVE_URI" \
  --target "$RESTORED_URI" \
  --overwrite \
  --dry-run
```

Overwrite in a non-interactive shell:

```bash
firecube archive restore \
  --archive "$ARCHIVE_URI" \
  --target "$RESTORED_URI" \
  --overwrite \
  --yes-i-really-mean-it
```

## Recover From An Interrupted Overwrite

On a local target, a restore that stops during the replacement leaves a
restore run in the `started` state. If Firecube cannot record a failure, the
run can be `started` or `failed`. In that case the restore prints a line that
starts with `Warning:`. A remote `.tgm` root restore replaces the whole store
as before, so it need not leave a run to find. Take these steps:

1. List the runs and read the status of the restore run:

   ```bash
   firecube chunks runs list --product-name "$RESTORED_URI"
   ```

2. If the restore run is still `started`, abandon it. Its ID starts with
   `archive-restore-`. Do not abandon a run in another status.

   ```bash
   firecube chunks runs abandon \
     --product-name "$RESTORED_URI" \
     --run-id "<run-id>" \
     --reason "interrupted restore" \
     --yes-i-really-mean-it
   ```

3. Run the restore again:

   ```bash
   firecube archive restore \
     --archive "$ARCHIVE_URI" \
     --target "$RESTORED_URI" \
     --overwrite \
     --yes-i-really-mean-it
   ```

The target can hold a partial restore until step 3 ends.

## Failure Recovery

| Symptom | Meaning | Recovery |
|---|---|---|
| `Invalid Tensogram archive <path>` | The archive failed its structural check. Firecube deleted and replaced no existing data. | Do not restore this archive. Create a new archive from the source product. |
| `holds a symlink at` | The target or a kept path under `.firecube/` is a symlink. Firecube deleted nothing. | Replace the symlink with a real directory, or restore to another target. |
| `Warning: archive restore failure not recorded` | The restore failed and Firecube could not record the failure. | Follow the steps in "Recover From An Interrupted Overwrite". |
| `Warning: claim <domain> at <path> not released` | The restore ended and Firecube could not release its claim. | Clear the claim: `firecube chunks claims clear --product-name "$RESTORED_URI" --domain "<domain>" --yes-i-really-mean-it`. |
| Restore target exists | Firecube will not replace it by default. | Use `--overwrite` with `--yes-i-really-mean-it`. |
| `requires the tensogram extras` | Archive support is not installed. | Install `firecube[tensogram]` in the active environment. |
| `Restoring a .zip archive to a remote target is not yet supported` | Zip restores write locally only. | Restore to a local target, then copy it to S3. |
| `Unsafe entry '<name>' in archive <path>` | The zip holds an entry that leaves the target. Firecube refused it before it restored any entry. | Do not restore this archive. Create a new archive from the source product. |
| Restored product has no expected variables | The archive is missing expected content. | Run `archive info` and `archive list`. |

## Next Steps

- **[Inspect And Validate Archives](inspect.md)** — confirm archive contents before restoring
- **[ChunkManager Operations](../chunk-manager/index.md)** — inspect or operate the restored product state
