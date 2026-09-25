# Paper 2 revision (Computer Speech & Language, September 2026): corrected-input analysis code

This directory holds the analysis scripts behind the revised manuscript of Paper 2. The files are byte-identical to the copies in the private verification package and are listed with SHA256 in `SHA256SUMS.txt`.

## Inputs: not public

The scripts read derived per-speaker tables and never raw audio. Those tables are **not included** here. The underlying corpora (among them SAP, OneVoice-MSD, PC-GITA, MDSC, the Hungarian dysarthric corpus and CDSD) are governed by data-use agreements that do not allow redistribution of speaker-level data. Researchers who hold access to those corpora can request the derived tables from the corresponding author, subject to those agreements.

The scripts resolve every path relative to this directory. To run them, put the inputs at:

```
paper2-revision/corrected/track4_master.csv                  # HuBERT-base master, 3,374 speakers
paper2-revision/corrected/track4_results_<backbone>.csv      # hubert-large, wavlm, wav2vec2, xlsr, mms
paper2-revision/corrected/CHANGES_step1.json                 # record of the input corrections
paper2-revision/frozen_package/results/track4_master.csv     # submitted-state master (used for comparisons)
paper2-revision/results/                                     # per-speaker intermediate tables (e.g. test4/test5 d′,
                                                             # Dutch re-scoring) and outputs written by the scripts
```

The submitted-state master and the per-speaker intermediate tables fall under the same data-use restrictions.

`scripts/v6/v6_fixed_token_dgx.py` works from phone-level embeddings. It takes the master path as its first argument and reads embeddings from the extraction tree produced by `scripts/extract_features.py` (repository root).

## Scripts and what they produce

| Script | Manuscript location |
|---|---|
| `scripts/v6/v6_test6.py`, `scripts/test6_v5.py` | Table 5, §4.3 (cross-backbone agreement, ≥3 of 5 consonant features, duplicate keys resolved keep-last), Dutch-exclusion comparison |
| `scripts/regen_fig5_v5.py` | Figure 5 |
| `scripts/regen_fig4_v5.py`, `v5_investigation/q2_fig4_adjacent_intervals.py` | Figure 4 and its adjacent-severity intervals |
| `scripts/regen_fig2_v5.py` | Figure 2 |
| `scripts/v6/v6_figures_1_3.py` | Figures 1 and 3 |
| `scripts/v6/v6_master_analyses.py` | §3 coverage counts, §4.1 bootstrap intervals, §4.4 regression, §4.5 robustness analyses (ablation, SAP-excluded, token-matched, minimum-n) |
| `scripts/v6/v6_crossling.py` | Table 4, §4.2 cross-lingual cosines |
| `scripts/corrected/test2_aetiology_vs_language.py` | §4.2 aetiology-specificity test (block permutation and bootstrap) |
| `scripts/corrected/test3_per_feature_effects.py` | Table 3 |
| `scripts/v6/v6_classifier.py` | §4.5 leave-one-dataset-out nearest-centroid classifier |
| `scripts/v6/v6_feature_importance.py` | §4.3 feature ranking |
| `scripts/v6/v6_test4_analysis.py`, `scripts/v6/v6_test45_analysis.py`, `scripts/test4_*.py`, `scripts/test5_*.py` | §4.5 direction-construction sensitivity, degradation and random-direction controls |
| `scripts/v6/v6_fixed_token_dgx.py` | §4.5 fixed-token d′ |
| `scripts/v6/v6_misc.py`, `scripts/v6/v6_table1_check.py` | Table 1 counts, Dutch re-scoring counts, companion-paper checks |
| `scripts/v6/v6_tamil_provenance.py` | §3.1 note on the 18 duplicated Tamil speaker keys |
| `results/v5/swap_run/build_swap.py`, `compare_swap.py` | §4.2 sensitivity analysis re-scoring 18 Tamil speakers in the HuBERT-base analysis |
| `results/v5/swap_run/unsourced_numbers.py`, `v5_investigation/*.py` | supporting checks (token quartiles, Figure 2 measurement, join diagnostics) |
| `scripts/rebuild_phase1.py`, `scripts/aetiology_undetermined.py`, `scripts/copas_recompute_v4.py`, `scripts/nl_repair.py`, `scripts/corrected/test9_recompute.py`, `scripts/corrected/test7_lodo_repro.py` | input-correction bookkeeping and earlier reconstruction checks |

The code for the originally submitted version stays at tag `submitted-state-2026-09`. The first corrected-revision battery stays at tag `corrected-revision-2026-09`.
