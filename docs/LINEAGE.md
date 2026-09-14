# Code lineage — submitted state, April anchor, and corrected revision

Three states of this codebase are relevant to the Computer Speech & Language revision of
"Phonological Subspace Collapse..." (manuscript YCSLA-D-26-00383), and they are deliberately
kept distinct:

## 1. April 2026 provenance anchor: commit `12e7950`
Commit `12e7950b3981b3d3e84d53deb3ef92b3cd8c31e1` (22 April 2026) is preserved untouched.
It is NOT byte-identical to the analysis state preserved in the frozen evidence package:
a full comparison (September 2026) found the 12 shared configuration files identical;
10 scripts shared, of which 8 differ only in a 2-line base-path default and
`extract_features.py` additionally carries four Greek dataset mappings absent from the
frozen state; 12 scripts exist only in this repository and 17 only in the frozen package
(figure generation and pseudo-label tooling). No one should cite `12e7950` as "the exact
submitted pipeline"; it is the contemporaneous April public release.

## 2. Submitted-state archive: `archive/submitted-state-2026-09/` (tag `submitted-state-2026-09`)
Assembled on 14 September 2026 from the checksummed frozen evidence package created for the
revision. Its README states the assembly date, logs two mechanical redactions of
machine-specific paths (with before/after SHA-256), and identifies four unrecoverable gaps
(lost classifier script, unpersisted permutation draws, lost-schema Hungarian configuration,
non-distributable data tables).

## 3. Corrected revision: `revision-2026/` (tag `corrected-revision-2026-09`)
The revision's pre-registered test battery, the audited data repairs (MDSC labels, COPAS
control flags, SAP join canonicalisation, Dutch direction rebuild), and the schema-translated
Hungarian configuration. Results reported in the revised manuscript come from these scripts
run on the corrected master; the frozen erroneous state is retained separately as evidence.

The revision's response letter cites states 2 and 3 explicitly and never describes either
as backdated or as the April commit.
