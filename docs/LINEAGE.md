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
   schema-translated Hungarian configuration. On full re-extraction from audio its values agree
   closely with the submitted values of the 28 Hungarian healthy controls (Spearman 0.983-1.000)
   but only approximately with those of the 31 Hungarian Dysarthria speakers (Spearman 0.851,
   median absolute difference 0.19); pooled over all 59 speakers the agreement is 0.946. The
   cause of the difference is unconfirmed. The configuration that produced the submitted values
   was not preserved (the lost-schema original is kept under the tags), so the revised
   manuscript keeps the submitted Hungarian values; a sensitivity check finds that no reported
   conclusion changes (`results/v19/HU_SENSITIVITY_FINDINGS.md`);
   see `paper2-revision/README.md`, section "Current reproduction route". Results reported in the revised manuscript come from these scripts run on the
   corrected master. Speaker-level inputs are governed by data-use agreements and are not
   distributed here.
