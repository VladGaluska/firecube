# Inspect And Validate Archives

Use archive inspection commands before transferring, publishing, or restoring a
`.tgm` or `.zip` file.

```bash
ARCHIVE_URI="file:///data/archives/MY_PRODUCT.tgm"
```

## Show Metadata

```bash
firecube archive info --archive "$ARCHIVE_URI"
```

Expected output resembles:

```text
Path:           /data/archives/MY_PRODUCT.tgm
Format:         v1 (multi-group)
Size:           0.00 MB
Groups:         data
Control-plane:  present

Group: data
  Source:    file:///data/products/MY_PRODUCT.zarr
  Codec:     zstd
  Variables:
    data: shape=[1000, 10] dtype=float32 chunks=[100, 10]
```

Use JSON for scripts:

```bash
firecube archive info --archive "$ARCHIVE_URI" --format json
```

Expected output resembles:

```json
{
  "path": "/data/archives/MY_PRODUCT.tgm",
  "format": "v1",
  "size_bytes": 4432,
  "groups": ["data"],
  "has_controlplane": true
}
```

## List Contents

```bash
firecube archive list --archive "$ARCHIVE_URI"
```

Expected output resembles:

```text
Messages: 2

Group: data
  Source:  file:///data/products/MY_PRODUCT.zarr
  Codec:   zstd
  Objects: 1
    [0] data: shape=[1000, 10] dtype=float32 compression=zstd

[control-plane]
  Product: product.zarr
```

## Validate Integrity

```bash
firecube archive validate --archive "$ARCHIVE_URI"
```

Expected output:

```text
VALID: /data/archives/MY_PRODUCT.tgm (2 message(s), no issues)
```

Use `--quick` for a structure-only check:

```bash
firecube archive validate --archive "$ARCHIVE_URI" --quick
```

Expected output:

```text
VALID: /data/archives/MY_PRODUCT.tgm (2 message(s), no issues)
```

## Zip Archives

`info`, `list`, and `validate` accept `.zip` archives with the same flags:

```bash
firecube archive info --archive "file:///data/archives/MY_PRODUCT.zarr.zip"
```

Expected output resembles:

```text
Path:           /data/archives/MY_PRODUCT.zarr.zip
Format:         zip
Size:           0.01 MB
Entries:        18
Groups:         (root), data
Control-plane:  present

Group: data
  Variables:
    temperature: shape=[4, 2, 3] dtype=float64 chunks=[4, 2, 3] codec=zstd
```

`archive list` prints each group as an xarray dataset. `archive validate`
checks the CRC of every entry and that the archive opens as Zarr; `--quick`
skips the CRC check.

## Failure Recovery

| Symptom | Meaning | Recovery |
|---|---|---|
| `Path ... does not exist` | The archive path is wrong or create failed. | Check `ARCHIVE_PATH` and rerun `archive create`. |
| `requires the tensogram extras` | `.tgm` support is not installed. | Install `firecube[tensogram]` in the active environment. |
| Validation fails | The archive is incomplete or corrupt. | Recreate the archive from the source product. |

## Next Steps

- **[Create Archives](create.md)** — rebuild or replace an archive
- **[Restore Archives](restore.md)** — restore a validated archive to a Zarr product
