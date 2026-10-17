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

Inspect restored control-plane records:

```bash
firecube chunks list --product-name "$RESTORED_URI"
```

## Overwrite An Existing Restore Target

`--overwrite` replaces the whole target. Data and run history written to the
target after the archive was created are lost.

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

## Failure Recovery

| Symptom | Meaning | Recovery |
|---|---|---|
| Restore target exists | Firecube will not replace it by default. | Use `--overwrite` with `--yes-i-really-mean-it`. |
| `requires the tensogram extras` | Archive support is not installed. | Install `firecube[tensogram]` in the active environment. |
| `Restoring a .zip archive to a remote target is not yet supported` | Zip restores write locally only. | Restore to a local target, then copy it to S3. |
| `Unsafe entry '<name>' in archive <path>` | The zip holds an entry that leaves the target. Firecube refused it before it restored any entry. | Do not restore this archive. Create a new archive from the source product. |
| Restored product has no expected variables | The archive is missing expected content. | Run `archive info` and `archive list`. |

## Next Steps

- **[Inspect And Validate Archives](inspect.md)** — confirm archive contents before restoring
- **[ChunkManager Operations](../chunk-manager/index.md)** — inspect or operate the restored product state
