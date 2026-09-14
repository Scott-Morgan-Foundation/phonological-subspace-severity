# Corrected-revision analysis code (September 2026)

Analysis code for the YCSLA-D-26-00383 major revision: the pre-registered test battery
(Tests 2-9 of the revision analysis plan), the data-integrity repairs, and the corrected-master
rerun. Every script fixes its random seed and reads only declared inputs.

- `test2_aetiology_vs_language.py` — the decisive same-aetiology vs different-aetiology
  cross-language comparison (bootstrap-primary; permutation variants with degeneracy diagnostics).
- `test3_per_feature_effects.py` — per-feature HC-normalised effects by aetiology x language.
- `test4_*` — healthy-control direction rebuild sensitivity (weighting variants, leave-one-corpus-out,
  cross-corpus transfer). `test5_*` — generic-degradation and matched random-direction controls.
- `test6_backbone_stability.py` — cross-backbone stability with uncertainty (bootstrap CIs, Kendall's W).
- `test7_lodo_repro.py` — the recovered leave-one-dataset-out nearest-centroid protocol (canonical
  implementation; the original script is lost, see the archive README).
- `test9_recompute.py`, `test9_hu_verify.py` — full recompute battery and the Hungarian
  reproduction check. `cite_audit.py` / `cite_renumber.py` — citation-order tooling.
- `build_corrected_inputs.py`, `nl_repair.py`, `assemble_corrected_master.py`,
  `verify_lavonne_claims.py` — the audited data repairs (MDSC labels, COPAS control flags,
  SAP join canonicalisation, Dutch direction rebuild) with traceable change counts.
- `config/phone_features_hu_standard_schema.json` — the schema-translated Hungarian configuration.

Speaker-level inputs (master table, inventories, control lists) are governed by data-use
agreements and are not distributed here; scripts that read them declare the expected paths.
