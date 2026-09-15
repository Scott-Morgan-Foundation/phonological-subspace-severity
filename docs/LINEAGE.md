# Code lineage — YCSLA-D-26-00383 revision

Three states of this codebase are relevant to the Computer Speech & Language revision of
"Phonological Subspace Collapse..." and are preserved as follows. The main branch holds
only the current pipeline that reproduces the revised paper's results; historical states
live in tagged history, not in parallel directories.

1. **April 2026 provenance anchor — commit `12e7950`** (untouched). Not byte-identical to
   the frozen analysis state: the 12 shared configuration files are identical; of 10 shared
   scripts, 8 differ only in a 2-line base-path default and the feature-extraction script
   additionally carries four Greek dataset mappings; 12 scripts exist only in the repository
   and 17 only in the frozen evidence package. Do not cite `12e7950` as "the exact submitted
   pipeline".

2. **Submitted-state archive — tag `submitted-state-2026-09`** (commit `0b4a618`). A
   September-2026 assembly of the frozen evidence package's scripts and configurations,
   with two logged path redactions (before/after SHA-256 in its README) and four identified
   unrecoverable gaps (lost classifier script, unpersisted permutation draws, lost-schema
   Hungarian configuration, non-distributable data tables).

3. **Corrected revision — tag `corrected-revision-2026-09`** (commit `6b07e71`), and the
   current main branch. The revision's pre-registered analysis battery and the audited data
   repairs (MDSC labels, COPAS control flags, SAP join canonicalisation, Dutch direction
   rebuild) now live directly in `scripts/`; `config/phone_features_hu.json` is the
   schema-translated Hungarian configuration that reproduces the submitted Hungarian values
   (Spearman 0.946 on full re-extraction; the lost-schema original is preserved under the
   tags). Results reported in the revised manuscript come from these scripts run on the
   corrected master. Speaker-level inputs are governed by data-use agreements and are not
   distributed here.
