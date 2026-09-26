# Redaction Notice

This waveform-MCQ case-study bundle was redacted before publishing to a **public** repo.
Internal absolute filesystem paths and an internal collaborator/source directory name
have been replaced with placeholders so that no internal infrastructure or username is exposed:

| placeholder | replaces (internal, not shown) |
|---|---|
| `<ptbxl_download>` | internal copy of the public PTB-XL dataset |
| `<genecg_source>` | internal GenECG source-image directory |
| `<collaborator_data>` / `<collaborator>` | internal collaborator source-data root / name |
| `<output_root>` | internal build/output root |
| `<bundle_root>` | this bundle's own build directory (the contents you see) |
| `<internal_root>` / `<internal_user>` | internal NAS root / owner |

`MANIFEST.json` and `BUNDLE_SHA256SUMS.txt` have been **recomputed** over the redacted files
(exactly the set committed here; compiled `.pyc` caches are excluded), so integrity still verifies:

```bash
sha256sum -c BUNDLE_SHA256SUMS.txt   # must report OK per file
```

No source code logic, parameters, waveforms, hashes, or clinical content were changed —
only internal path strings were masked.
