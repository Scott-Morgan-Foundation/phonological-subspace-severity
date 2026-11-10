# Hungarian sensitivity check — does the Hungarian_Dysarthria discrepancy change a reported result?

**Date:** 2026-11-10.
**Script:** `scripts/v19/v19_hu_sensitivity.py` (run ×4, then `compare`) and `scripts/v19/v19_hu_printed_values.py`.
**Outputs:** `results/v19/hu_sensitivity/` — `printed_values_by_variant.csv` (every bound printed value under each variant), `diff.json` (every changed output value), `V*/run_log.json` (inputs, exit codes, files written).

## Question and scope

The cause of the discrepancy is not investigated; no audio was re-extracted. The question is whether any reported number or claim depends on the Hungarian values. Every script that turns the speaker-level tables into a reported number was re-run **unchanged**, in a mirror of the revision folder, on four versions of the input tables:

| Variant | What changes | Rows affected |
|---|---|---|
| V0 retained | nothing (the tables behind the paper) | — |
| V1 re-extracted | the nine d′ values of all 59 Hungarian speakers in the HuBERT-base master replaced by the September 2026 re-extraction with the released Hungarian configuration (`results/test9_hu_dprime.csv`); the five other backbone tables were not re-extracted and are unchanged | 59 speakers, 526 values replaced, none newly missing |
| V2 exclude Hungarian_Dysarthria | the Hungarian_Dysarthria corpus removed (incl. the 1 PD and 2 stroke speakers in the six-group analyses); it appears only in the HuBERT-base master, not in the five other backbone tables | 39 rows in the master (31 language-hu + 8 language-en) |
| V3 exclude all Hungarian | all 59 language-hu speakers removed, so the 28 Hungarian controls also leave every healthy-control reference | 59 rows in the master; 28 controls in each of the five other backbone tables |

Scripts re-run (27 steps): the corrected battery (test2, test3, test6, test7, test9), `rebuild_phase1.py`, `aetiology_undetermined.py`, `regen_fig2_v5.py`, v6 (master analyses, classifier, cross-lingual, test 4/5 analysis), v8 (backbones, original scripts, master facts, test-2 sensitivity, token quartiles, CP retention), v9 original robustness analyses (local part), v10 (facts, Figure 4, high-risk values), v12 and v17 matched severity. Then the v15 consistency check re-derives every bound printed number of the first-revision text from the variant's outputs.

**Not re-run, with reason:** the fixed-token analysis (Table 7) — no Hungarian speaker qualifies (asserted from `results/v17/v17_fixed_token_hcdir_speakers.csv`); the DGX parts of the original scripts (fixed-token and final-layer runs on embeddings) — same reason, and they need the embedding cache.

**Baseline check:** in V0 all 27 steps exit 0; the 37 re-written outputs equal the saved canonical outputs (the only differences: one figure's file size, one bootstrap interval in `v6_crossling.json` at minimum 10 controls that is not a printed value, and the checksums of two run logs that record timings); and the consistency check passes with 0 failures over 1,245 numeric tokens.

## Result

**No reported claim changes in any variant.** Under the re-extracted values (V1), 39 of 763 bound printed values change, all in the last printed digit, and all 46 derived claims of the consistency check still hold. Removing speakers (V2, V3) changes sample counts, as it must, and moves values by similar small amounts.

| Reported result | Printed | V1 re-extracted | V2 excl. HD (31) | V3 excl. all hu (59) | Claim |
|---|---|---|---|---|---|
| Speakers in the analysis | 3,374 | 3,374 | 3,335 | 3,315 | — |
| Features with large effects (Table 3) | 10 of 13 | 10 | 10 | 10 | unchanged |
| Table 3 epsilon-squared (consonants) | 0.237–0.431 | 0.237–0.431 | 0.236–0.431 | 0.239–0.431 | unchanged; every Large/Small label kept |
| Holm-significant pairs | 12 of 15 | 12 | 12 | 12 | unchanged |
| PD vs CP/DS/stroke, mean d | 1.01 | 1.01 | 1.01 | 1.01 | unchanged |
| CP vs stroke d | 0.04 | 0.04 | 0.04 | 0.04 | unchanged |
| Moderate-only PD vs pooled execution group, d (n PD / pooled) | 0.01 (42 / 45) | 0.00 (42 / 45) | −0.01 (41 / 44) | −0.01 (41 / 44) | unchanged (no separation at moderate severity) |
| Cross-language profile cosine PD, CP, ALS | 0.978–0.987 | same | same | same | unchanged |
| Aetiology-specificity test, block permutation p | 0.055 | 0.055 | 0.055 | 0.055 | unchanged (Hungarian cells are below the 3-speaker minimum) |
| Inter-model rho (Table 6) | 0.83–0.99 | 0.83–0.99 | 0.83–0.99 | 0.84–0.99 | unchanged |
| Per-aetiology profile similarity across backbones, minimum | 0.91 | 0.91 | 0.91 | 0.91 | unchanged |
| Kendall's W (feature ordering), range | 0.33–0.71 | 0.33–0.71 | 0.33–0.71 | 0.29–0.68 | unchanged ("only partially") |
| Severity rho, HuBERT-base (SAP excluded) | −0.542 (−0.408) | −0.542 (−0.408) | −0.540 (−0.405) | −0.544 (−0.410) | unchanged |
| Leave-one-dataset-out classification, macro F1 | 21.9% | 21.6% | 21.0% | 20.9% | unchanged ("limited") |
| Table 4 mild: features differing after Holm (large) | 11 of 13 (6) | 11 (6) | 11 (6) | 11 (6) | unchanged; same features |
| Table 4 moderate: features differing after Holm | 3 of 13 | 3 | 3 | 3 | unchanged; same features (height, rounding, speech rate) |
| Table 4 moderate: largest consonant epsilon-squared | 0.0053 ("below 0.006") | 0.0050 | 0.0044 | 0.0044 | unchanged |
| Pooled-reference centred shape: statistic / null / p | 51.6 / 3.7 / 0.0001 | 51.6 / 3.7 / 0.0001 | 51.6 / 3.7 / 0.0001 | 51.0 / 3.6 / 0.0001 | unchanged |
| Mild PD centred voicing / sonorance (pooled reference) | +0.34 / −0.35 | same | same | +0.33 / −0.35 | unchanged |
| Same-language centred shape: statistic / null / p | 11.0 / 5.6 / 0.028 | 10.4 / 5.7 / 0.042 | 10.9 / 5.6 / 0.027 | 10.9 / 5.6 / 0.027 | unchanged at the 0.05 level; the p-value is the largest single change |
| Mild PD centred values, same-language reference | within ±0.12 | within ±0.12 | within ±0.12 | within ±0.12 | unchanged |

Matched-severity sample counts (Table 4, all corpora): mild PD 98, CP 110, ALS 68, DS 12 in every variant; moderate PD 42 (V2/V3: 41), CP 35, ALS 10, DS 9. The Hungarian PD speaker is the one moderate PD speaker removed in V2/V3; the two Hungarian stroke speakers never enter a matched-severity cell (stroke has no cell of five).

Where the last printed digit changes under V1 (full list in `printed_values_by_variant.csv`, column `changed_V1`): Table 3 H, p and four epsilon-squared values in the third decimal (voicing 0.424→0.423, height 0.498→0.499, lowness 0.345→0.346, rounding 0.443→0.442; rounding n 2,639→2,642); Kendall's W for stroke 0.48→0.47; Holm PD–stroke d 0.98→0.97; the DS–stroke rank-biserial −0.001→+0.001; the SAP-excluded full-sample nasality and rounding epsilon-squared 0.180→0.179 and 0.165→0.163; classifier F1 HC 0.627→0.625, CP 0.388→0.384, macro 21.9%→21.6%; moderate-only d 0.01→0.00; Figure 2 stroke ratios by at most 0.013; Table 4 moderate rounding 0.132→0.139 and sonorance 0.003→0.001; same-language statistic 11.0→10.4, p 0.028→0.042.

Two consistency-check claims fail only under the exclusion variants, both because removing speakers changes the bootstrap draws: an interval bound at minimum 3 vs 5 (or 10) speakers per cell differs by 0.001 where the text says the intervals are identical; and, when all Hungarian speakers are removed, Hungarian_Dysarthria joins CDSD and Domotica as a label whose deletion removes no speaker. Neither occurs with the retained or re-extracted values.

## Assessment

No reported claim changes. The manuscript keeps the retained (submitted) values, and the disclosure in the Data Availability statement now reports this check. If the re-extracted values were the correct ones, the printed numbers listed above would differ in the last digit and the same-language omnibus p-value would be 0.042 rather than 0.028; no conclusion drawn from any of them would change.
