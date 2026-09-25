# Paper 2 revision (Computer Speech & Language, September 2026): corrected-input analysis code

This directory holds the analysis scripts behind the revised manuscript of Paper 2 (version v10 of the revision package). The files are byte-identical to the copies in the private verification package and are listed with SHA256 in `SHA256SUMS.txt`.

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

The figure scripts write to `paper2-revision/figures/`, so create that directory before running them.

### Original submitted scripts, re-run unchanged

`scripts/v8/originals/` and `scripts/v9/originals/` hold byte-identical copies of the analysis scripts used for the submitted manuscript (their SHA256 values are asserted by the wrappers). Each original reads `$DYSARTHRIA_BASE/results/track4/*.csv`; the wrappers `scripts/v8/v8_run_originals.py` and `scripts/v9/v9_originals_rerun.py` only stage copies of the chosen inputs (corrected or submitted) in a directory and set `DYSARTHRIA_BASE` to it, then save the console output of every run and parse it to JSON.

`scripts/v9/originals/fixed_token_dprime.py` and `robustness_pass5.py` work from phone-level embeddings (`results/track4/embeddings/*.npz`, produced by `scripts/extract_features.py` at the repository root) and the phone-feature configurations in `config/`. `scripts/v9/v9_originals_rerun.py dgx` runs them on a remote machine that holds the embeddings; it takes the connection from the environment variables `P2_DGX_TARGET` (user@host), `P2_DGX_KEY` (SSH key), `P2_DGX_REMOTE` (scratch directory) and `P2_DGX_HOME` (home directory containing `dysarthria/venv`, `dysarthria/config` and `dysarthria/results/track4/embeddings`).

## Tables and figures: generating script

| Manuscript element | Script (output) |
|---|---|
| Table 1 (corpus composition, severity source, counts) | `scripts/v8/v8_master_facts.py` (`table1`), `scripts/v6/v6_table1_check.py` |
| Table 2 (SSL backbones) | none: model descriptors from the model cards |
| Table 3 (Kruskal-Wallis H, p, ε², N per feature) | `scripts/rebuild_phase1.py` (`results/rebuild_2026-09-23/table3_regenerated.json`) |
| Table 4 (cross-lingual cosine: mean, min, max) | `scripts/v6/v6_crossling.py` (`min_n_3`); the mean and minimum are also produced by the originals `scripts/v9/originals/robustness_analyses.py` (Analysis 2) and `reviewer_experiments.py` (Experiment 2) |
| Table 5 (cross-backbone Spearman ρ, n = 3,101 per pair) | `scripts/v8/v8_test6.py` |
| Table 6 (fixed-token d′) | original `scripts/v9/originals/fixed_token_dprime.py` (Experiment A), run by `scripts/v9/v9_originals_rerun.py` |
| Figure 1 (deviation from controls) and Figure 3 (PD profiles by language) | `scripts/v6/v6_figures_1_3.py` |
| Figure 2 (group-mean radar profiles) | `scripts/regen_fig2_v5.py` |
| Figure 4 (severity gradient, 6 backbones) and its adjacent-severity intervals | `scripts/v10/regen_fig4_v10.py` (speaker metadata from the master for every panel; `results/v10/fig4_adjacent_intervals_v10.json`) |
| Figure 5 (cross-backbone agreement heat map) | `scripts/v8/regen_fig5_v8.py` |

## Text analyses: generating script

| Section | Analysis | Script |
|---|---|---|
| Contribution (b) | cross-lingual cosine with per-language-pair speaker bootstrap | original `scripts/v9/originals/robustness_analyses.py` (Analysis 2) |
| §3.1 | 18 duplicated Tamil speaker keys; HuBERT-base Tamil provenance | `scripts/v6/v6_tamil_provenance.py`, `scripts/v8/v8_tamil_hubert_provenance.py` |
| §4.1 | stratified bootstrap severity correlation (re-implementation, controls = 0; the original's result is from `scripts/v9/originals/robustness_analyses.py`, Analysis 1); pairwise d | `scripts/v6/v6_master_analyses.py` |
| §4.2 | aetiology-specificity test (block permutation, bootstrap, sensitivities) | `scripts/corrected/test2_aetiology_vs_language.py`, `scripts/v8/v8_test2_sensitivity.py`; Tamil re-scoring: `results/v5/swap_run/build_swap.py`, `compare_swap.py` |
| §4.2, §5.1 | Kruskal-Wallis across languages | `scripts/v8/v8_master_facts.py` |
| §4.2, contribution (b) | per-feature cross-lingual replication (HC-normalised effects by aetiology and language) | `scripts/corrected/test3_per_feature_effects.py` (`results/test3_effects_corrected.csv`) |
| §4.3 | per-backbone severity ρ, profile cosines, Kendall's W, Dutch exclusion | `scripts/v8/v8_test6.py` |
| §4.3 | per-backbone feature ranking | original `scripts/v8/originals/cross_model_comparison.py` |
| §4.3 | leave-one-dataset-out severity correlation (re-implementation, controls = 0; 26 labels, 24 deletions that change the sample) | `scripts/v6/v6_master_analyses.py`, `scripts/v10/v10_facts.py` (`lodo_effective`); original: `scripts/v9/originals/robustness_analyses.py` (Analysis 3) |
| §4.4 | CTC-Conf correlations, Ridge regression | `scripts/v6/v6_master_analyses.py` |
| §4.5 | severity-source ablation | `scripts/v6/v6_master_analyses.py` |
| §4.5 | minimum-n cross-lingual cosines (bootstrap over language pairs) and Holm post hoc | original `scripts/v9/originals/reviewer_experiments.py` (Experiments 2 and 3) |
| §4.5 | token-matched comparisons | original `scripts/v8/originals/reviewer_experiments.py` (Experiment 4) |
| §4.5 | groups compared at each fixed-token budget | `scripts/v10/v10_fixed_token_groups.py` |
| §4.5 | fixed-token d′ (Table 6) and common-speaker set; minimum-HC sensitivity (bootstrap over language pairs) | originals `scripts/v9/originals/fixed_token_dprime.py` (Experiment A) and `robustness_pass5.py` (Experiments 1 and 2) |
| §4.5 | SAP-excluded sensitivity | original `scripts/v8/originals/sap_excluded_sensitivity.py` |
| §4.5 | leave-one-dataset-out classifier | `scripts/v6/v6_classifier.py` |
| §4.5 | direction-construction sensitivity; degradation and random-direction controls | `scripts/v6/v6_test4_analysis.py`, `scripts/v6/v6_test45_analysis.py`, `scripts/test4_*.py`, `scripts/test5_*.py` |
| §5.1 | English CP retention by severity | `scripts/v8/v8_cp_retention.py` |
| §5.2 | token-count quartiles | `scripts/v8/v8_token_quartiles.py` |
| §3.1–§4.5 | coverage, Dutch exclusion for all six groups, VTA ratios, Table 1 severity sources, label-source audit | `scripts/v10/v10_facts.py` |
| letter | originals run on the submitted inputs against the submitted manuscript | `scripts/v10/v10_originals_reproduction.py` |
| all | number registry (single rounding), high-risk bindings and executable consistency check; planted-error test | `scripts/v10/format_numbers.py`, `scripts/v10/v10_hr_values.py`, `scripts/v10/hr_*.py`, `scripts/v10/build_hr_bindings.py` (`hr_bindings.json`), `scripts/v10/check_consistency.py`, `scripts/v10/mutation_test_v10.py` (extending `scripts/v9/`, `scripts/v8/`) |

Other scripts here are supporting checks (`v5_investigation/*.py`, `results/v5/swap_run/unsourced_numbers.py`, `scripts/v8/v8_cite_order.py`, `scripts/v10/v10_cite_order.py`, `scripts/v10/apply_v10_edits.py`, `scripts/v10/build_numbers_v10.py`, `scripts/v8/v8_measure_figs.py`, `scripts/v6/v6_measure_figs.py`, `scripts/v6/v6_misc.py`) or input-correction bookkeeping and earlier reconstruction checks (`scripts/rebuild_phase1.py`, `scripts/aetiology_undetermined.py`, `scripts/copas_recompute_v4.py`, `scripts/nl_repair.py`, `scripts/corrected/test9_recompute.py`, `scripts/corrected/test7_lodo_repro.py`, `scripts/corrected/test6_backbone_stability.py`, `scripts/test6_v5.py`, `scripts/regen_fig5_v5.py`).

The code for the originally submitted version stays at tag `submitted-state-2026-09`. The first corrected-revision battery stays at tag `corrected-revision-2026-09`. Tags `paper2-csl-revision-2026-09-25-r2` and `-r3` hold the previous revision rounds; the manuscript cites tag `paper2-csl-revision-2026-09-25-r4`.
