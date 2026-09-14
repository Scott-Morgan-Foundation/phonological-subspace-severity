# Submitted-state archive (assembled September 2026)

This directory was **assembled on 14 September 2026** from the frozen evidence package
created for the YCSLA-D-26-00383 major revision. It is NOT a byte-identical copy of any
single historical commit: it reproduces, as faithfully as the surviving records allow,
the analysis code and configurations in the state used for the April 2026 submission.
Commit `12e7950` remains the untouched April 2026 provenance anchor; the differences
between that commit and this archive are documented in `docs/LINEAGE.md`.

## Contents
- `scripts/` — the 28 analysis scripts preserved in the frozen evidence package.
- `config/` — the 13 phone-feature configuration files as frozen. Note: `phone_features_hu.json`
  and `phone_features_sw.json` were not present in the local configuration directory at freeze
  time and were restored from commit `12e7950` of this repository (documented as discrepancy
  F-3); `phone_features_el.json` is a local-only file not used by the submitted paper.

## Documented substitutions (not silent repairs)
Two files contained machine-specific paths, a hostname and a username that are redacted in
this public archive. The substitutions are mechanical and fully logged; the frozen originals
retain their recorded SHA-256 in the private evidence package.

| File | Substitutions | frozen sha256 (prefix) | published sha256 (prefix) |
|---|---|---|---|
| `pseudolabel_msd_bundle.py` | 1 line(s) | `6f8303fb7802a6ad...` | `295b92c6e9072238...` |
| `refresh_master_severity.py` | 3 line(s) | `ce717e8bb7add254...` | `6d9da84a2225d0e4...` |

## Known unrecoverable gaps (identified, not hidden)
1. The leave-one-dataset-out nearest-centroid classifier script is lost; its protocol was
   recovered by grid reproduction (cosine distance, raw 5-dim profiles, >=3-of-5 valid,
   n = 2,928 exact) and the recovered implementation ships in `revision-2026/`.
2. The permutation draws behind the submitted cross-lingual null (observed 0.979, null 0.982,
   p = 0.84) were not persisted; a seeded reconstruction reproduces the observed value but not
   the null (0.976, p = 0.21).
3. The Hungarian configuration that produced the submitted Hungarian results does not survive
   in a schema the pipeline can read; a schema-translated configuration reproduces the
   submitted values at Spearman 0.946 (full re-extraction) and ships in `revision-2026/`.
4. Result tables and per-speaker data files are not distributable here (data-use agreements);
   they are preserved in the checksummed private evidence package.
