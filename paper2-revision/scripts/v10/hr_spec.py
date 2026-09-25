#!/usr/bin/env python3
"""v10 high-risk binding specification: for every non-structural number in the high-risk regions, in reading order,
the saved value it must equal. scripts/v10/build_hr_bindings.py turns this into frozen sentence contexts
(scripts/v10/hr_bindings.json); scripts/v10/check_consistency.py verifies them.

Reference forms (modifier after '~'):
  R:<registry id>                 value as formatted by scripts/v10/format_numbers.py (single rounding)
  R:<registry id>~abs             same, sign dropped
  H:<key>[.<subkey>]~<rule>       results/v10/v10_hr_values.json values
  X:<key>~<rule>                  results/v10/v10_external_descriptors.json values (external, as printed)
  J:<json file>#<dot.path>~<rule> any saved JSON output
Rules: Ndp, int, lt10:-k (printed exponent of a '< 10^-k' bound), lt=<v> (value < v, printed v),
gt=<v> (value > v, printed v).
A row-block reference (Table 3/4 rows) covers every token in the block; list only the block's first token."""

T2 = "J:results/test2_delta_corrected.json#"
V9 = "J:results/v9/v9_originals.json#states.corrected.parsed."
F2 = "J:results/v5/fig2_group_means.json#"

SPEC = {
    "abstract": [
        ("890", "H:p1_speakers~int"), ("5", "H:p1_languages~int"), ("3,374", "H:n_speakers~int"),
        ("25", "H:n_datasets~int"), ("12", "H:n_languages~int"), ("5", "H:n_aetiologies~int"), ("6", "H:n_backbones~int"),
        ("10", "H:n_large_main~int"), ("13", "H:n_features_main~int"), ("0.14", "H:large_threshold~2dp"),
        ("< 10$^{-11}$", "H:kw_p_max_main~lt10:-11"), ("1.01", "H:pd_exec_mean_d~2dp"), ("0.01", "H:mod_d~2dp"),
        ("21.9", "H:clf_macro_f1_pct~1dp"), ("5", "H:n_consonant_features~int"), ("0.978", "R:b_PD_mean"),
        ("0.987", "R:b_CP_mean"), ("0.93", "H:random_floor~2dp"),
        ("0.055", T2 + "perm_lang_dataset_hcnorm.p_null_ge_obs~3dp"), ("0.0004", T2 + "primary_hcnorm.delta~4dp"),
        ("-0.003", T2 + "primary_hcnorm.ci95.0~3dp"), ("0.004", T2 + "primary_hcnorm.ci95.1~3dp"),
        ("15", "H:cp_pos~int"), ("15", "H:cp_cells~int"), ("14", "H:cp_ci_excl0~int"), ("15", "H:cp_cells~int"),
        ("30", "H:pd_pos~int"), ("30", "H:pd_cells~int"), ("6", "H:n_backbones~int"),
        ("0.83", "R:xb_min_2dp"), ("0.99", "R:xb_max_2dp"), ("15", "H:n_backbone_pairs~int"),
        ("3,101", "R:t5_xlsr|mms_n_int"), ("0.83", "R:xb_min_2dp"), ("0.99", "R:xb_max_2dp"),
        ("0.33", "R:kw_min_2dp"), ("0.71", "R:kw_max_2dp"), ("467", "R:t6n_200"), ("200", "H:budget_max~int"),
        ("-0.733", "R:t6rho_200_3dp"),
    ],
    "contributions": [
        ("10", "H:n_large_main~int"), ("13", "H:n_features_main~int"), ("0.14", "H:large_threshold~2dp"),
        ("1.01", "H:pd_exec_mean_d~2dp"), ("3", "H:min_features~int"), ("0.04", "H:cp_stroke_d~2dp"),
        ("0.01", "H:mod_d~2dp"), ("42", "H:mod_pd_n~int"), ("45", "H:mod_exec_n~int"),
        ("5", "H:n_consonant_features~int"), ("0.95", "H:b_min_mean_pd_cp_als~gt=0.95"), ("6", "H:b_PD_nlang~int"),
        ("3", "H:min_cell~int"), ("4", "H:b_CP_nlang~int"), ("4", "H:b_ALS_nlang~int"), ("1,000", "H:a2_n_boot~int"),
        ("0.978", "R:b_PD_mean"), ("0.956", "R:b_PD_lo"), ("0.984", "R:b_PD_hi"), ("0.987", "R:b_CP_mean"),
        ("0.976", "R:b_CP_lo"), ("0.994", "R:b_CP_hi"), ("0.985", "R:b_ALS_mean"), ("0.954", "R:b_ALS_lo"),
        ("0.989", "R:b_ALS_hi"), ("0.980", "R:b_HC_mean"), ("0.968", "R:b_HC_lo"), ("0.986", "R:b_HC_hi"),
        ("0.937", "R:mn3_PD_min"), ("0.968", "R:b_PD_tight_cos"), ("0.893", "R:b_PD_tight_lo"),
        ("0.055", T2 + "perm_lang_dataset_hcnorm.p_null_ge_obs~3dp"), ("0.0004", T2 + "primary_hcnorm.delta~4dp"),
        ("-0.003", T2 + "primary_hcnorm.ci95.0~3dp"), ("0.004", T2 + "primary_hcnorm.ci95.1~3dp"),
        ("14", "H:cp_ci_excl0~int"), ("15", "H:cp_cells~int"), ("30", "H:pd_pos~int"), ("5", "H:de_als_n~int"),
        ("6", "H:n_backbones~int"), ("0.83", "R:xb_min_2dp"), ("0.99", "R:xb_max_2dp"), ("15", "H:n_backbone_pairs~int"),
        ("3,101", "R:t5_xlsr|mms_n_int"), ("15", "H:n_backbone_pairs~int"), ("0.98", "R:pc_min_PD_2dp"),
        ("0.96", "R:pc_min_CP_2dp"), ("0.91", "R:pc_min_ALS_2dp"), ("0.96", "R:pc_min_DS_2dp"),
        ("0.94", "R:pc_min_Stroke_2dp"), ("0.96", "H:family_min_rho~gt=0.96"), ("0.33", "R:kw_min_2dp"),
        ("0.71", "R:kw_max_2dp"),
    ],
    "conclusion": [
        ("10", "H:n_large_main~int"), ("13", "H:n_features_main~int"), ("0.14", "H:large_threshold~2dp"),
        ("0.62", "R:holm_alspd_d_2dp~abs"), ("1.05", "R:holm_dspd_d_2dp~abs"), ("0.01", "H:mod_d~2dp"),
        ("0.5", "H:holm_ns_min_p~gt=0.5"), ("21.9", "H:clf_macro_f1_pct~1dp"), ("5", "H:n_consonant_features~int"),
        ("0.978", "R:b_PD_mean"), ("0.987", "R:b_CP_mean"), ("6", "H:n_backbones~int"),
        ("5", "H:n_consonant_features~int"), ("0.83", "R:xb_min_2dp"), ("0.99", "R:xb_max_2dp"),
        ("15", "H:n_backbone_pairs~int"), ("3,101", "R:t5_xlsr|mms_n_int"), ("0.83", "R:xb_min_2dp"),
        ("0.91", "R:pc_min_all_2dp"), ("-0.747", "R:cs_rho_min_3dp"), ("-0.733", "R:cs_rho_max_3dp"),
        ("-0.453", "H:abl_rho~3dp"), ("0.188", "H:abl_eps2~3dp"), ("10", "H:min_n_threshold_top~int"),
        ("0.975", "R:mn10_PD_mean"), ("5", "H:n_consonant_features~int"),
        ("5", V9 + "reviewer_experiments.exp2_min_n_language_pair_bootstrap.10.PD.n_languages~int"),
        ("10", "H:min_n_threshold_top~int"), ("0.001", "H:tm_p_max~lt=0.001"), ("3,374", "H:n_speakers~int"),
        ("25", "H:n_datasets~int"), ("12", "H:n_languages~int"), ("5", "H:n_aetiologies~int"),
        ("12", "H:n_languages~int"), ("12", "H:n_languages~int"),
    ],
    "highlights": [("0.01", "H:mod_d~2dp"), ("0.055", T2 + "perm_lang_dataset_hcnorm.p_null_ge_obs~3dp")],
    "figure1": [("13", "H:fig1_n_features~int"), ("5", "H:fig1_n_aetiologies~int"), ("5", "H:fig1_min_cell~int")],
    "figure2": [
        ("1.0", F2 + "ratio_to_hc.HC.nasal_dprime~1dp"), ("1,445", F2 + "n_speakers.HC~int"), ("648", F2 + "n_speakers.PD~int"),
        ("379", F2 + "n_speakers.CP~int"), ("325", F2 + "n_speakers.ALS~int"), ("166", F2 + "n_speakers.DS~int"),
        ("99", F2 + "n_speakers.Stroke~int"), ("12", "H:vta_ds_n~int"),
    ],
    "figure3": [
        ("6", "H:fig3_n_langs~int"), ("3", "H:min_cell~int"), ("9", "H:n_segmental~int"), ("1.0", "H:hc_reference~1dp"),
        ("13", "H:fig3_nl_pd~int"), ("3", "H:fig3_pt_pd~int"),
    ],
    "figure4": [
        ("6", "H:n_backbones~int"), ("1,000", "H:fig4_n_boot~int"), ("0.05", "R:f4_xlsr_mm_margin"),
        ("-0.02", "R:f4_xlsr_mm_lo"), ("0.12", "R:f4_xlsr_mm_hi"), ("0.03", "R:f4_mms_mm_margin"),
        ("-0.04", "R:f4_mms_mm_lo"), ("0.08", "R:f4_mms_mm_hi"),
    ],
    "figure5": [("3,101", "R:t5_xlsr|mms_n_int"), ("0.90", "H:family_min_rho~gt=0.90")],
    "table1": [
        ("25", "H:n_datasets~int"), ("12", "H:n_languages~int"), ("26", "H:n_dataset_labels~int"), ("8", "H:t1_hu_english~int"),
        ("1,435", "H:t1_language_n.en~int"), ("1,233", "H:t1_dataset_n.SAP~int"), ("1", "H:sap_scale_min~int"),
        ("7", "H:sap_scale_max~int"), ("150", "H:t1_dataset_n.LibriSpeech_English~int"), ("15", "H:t1_dataset_n.TORGO~int"),
        ("15", "H:t1_dataset_n.UASPEECH~int"), ("13", "H:t1_dataset_n.UASPEECH_control~int"), ("602", "H:t1_language_n.sk~int"),
        ("602", "H:t1_dataset_n.EWA-DB~int"), ("370", "H:t1_language_n.pt~int"), ("370", "H:t1_dataset_n.AVFAD~int"),
        ("283", "H:t1_language_n.nl~int"), ("227", "H:t1_dataset_n.COPAS~int"), ("43", "H:t1_dataset_n.Domotica~int"),
        ("8", "H:t1_dataset_n.CHASING~int"), ("5", "H:t1_dataset_n.TreasureHunters1~int"), ("211", "H:t1_language_n.es~int"),
        ("111", "H:t1_dataset_n.Neurovoz~int"), ("100", "H:t1_dataset_n.PC-GITA~int"), ("120", "H:t1_language_n.it~int"),
        ("65", "H:t1_dataset_n.IPVS~int"), ("55", "H:t1_dataset_n.EasyCall~int"), ("100", "H:t1_language_n.zh~int"),
        ("56", "H:t1_dataset_n.MDSC~int"), ("44", "H:t1_dataset_n.CDSD~int"), ("80", "H:t1_language_n.ta~int"),
        ("50", "H:t1_dataset_n.SLR65_Tamil~int"), ("30", "H:t1_dataset_n.SSNCE_Tamil~int"), ("66", "H:t1_language_n.de~int"),
        ("53", "H:t1_dataset_n.SVD~int"), ("13", "H:t1_dataset_n.YouTube_German~int"), ("59", "H:t1_language_n.hu~int"),
        ("39", "H:t1_dataset_n.Hungarian_Dysarthria~int"), ("27", "H:t1_dataset_n.CV_Hungarian~int"),
        ("1", "H:t1_dataset_n.Hungarian_HC~int"), ("15", "H:t1_hu_labelled~int"), ("39", "H:t1_hu_total~int"),
        ("24", "H:t1_language_n.fr~int"), ("24", "H:t1_dataset_n.YouTube_French~int"), ("24", "H:t1_language_n.sw~int"),
        ("25", "H:t1_dataset_n.CDLI_Kenyan_Swahili~int"),
    ],
    "table2": [
        ("60,000", "X:hubert_large_librilight_hours~int"), ("960", "X:wavlm_base_librispeech_hours~int"),
        ("94,000", "X:wavlm_large_hours~int"), ("768", "X:hidden_hubert_base~exact"), ("1024", "X:hidden_hubert_large~exact"),
        ("768", "X:hidden_wavlm_base~exact"), ("768", "X:hidden_wav2vec2_base~exact"), ("1024", "X:hidden_xlsr_300m~exact"),
        ("128", "X:xlsr_languages~int"), ("1024", "X:hidden_mms_300m~exact"), ("1,100", "X:mms_languages~int"),
    ],
    "table3": [
        ("15", "H:n_features_all~int"), ("13", "H:n_features_main~int"), ("0.14", "H:large_threshold~2dp"),
        ("0.06", "H:medium_threshold~2dp"), ("0.14", "H:large_threshold~2dp"), ("0.06", "H:medium_threshold~2dp"),
        ("1,191.4", "R:t3row_nasal_dprime"), ("1,253.6", "R:t3row_voicing_dprime"), ("700.3", "R:t3row_sonorant_dprime"),
        ("1,239.2", "R:t3row_strident_dprime"), ("1,153.7", "R:t3row_manner_dprime"), ("1,381.6", "R:t3row_high_dprime"),
        ("961.7", "R:t3row_low_dprime"), ("1,080.0", "R:t3row_back_dprime"), ("1,172.2", "R:t3row_round_dprime"),
        ("397.4", "R:t3row_vowel_triangle_area"), ("191.7", "R:t3row_boundary_sharpness"),
        ("196.8", "R:t3row_cross_position_cosim"), ("155.4", "R:t3row_speech_rate"), ("278.6", "R:t3row_vowel_duration_cv"),
        ("64.6", "R:t3row_pause_rate"),
    ],
    "table4": [
        ("5", "H:n_consonant_features~int"), ("3", "H:min_cell~int"),
        ("4", "H:t4_nlang.CP~int"), ("0.987", "R:t4row_CP"), ("4", "H:t4_nlang.ALS~int"), ("0.985", "R:t4row_ALS"),
        ("6", "H:t4_nlang.PD~int"), ("0.978", "R:t4row_PD"), ("11", "H:t4_nlang.HC~int"), ("0.980", "R:t4row_HC"),
    ],
    "table5": [("15", "H:n_backbone_pairs~int"), ("3,101", "R:t5_xlsr|mms_n_int"), ("0.90", "H:family_min_rho~gt=0.90")] + [
        tok for pair, v in (("hubert-base|wavlm", "0.985"), ("hubert-large|wavlm", "0.968"), ("hubert-base|hubert-large", "0.963"),
                            ("hubert-base|wav2vec2", "0.949"), ("wavlm|wav2vec2", "0.948"), ("hubert-large|wav2vec2", "0.934"),
                            ("hubert-base|mms", "0.872"), ("wavlm|mms", "0.870"), ("hubert-large|mms", "0.859"),
                            ("hubert-base|xlsr", "0.859"), ("wavlm|xlsr", "0.858"), ("hubert-large|xlsr", "0.852"),
                            ("wav2vec2|xlsr", "0.839"), ("wav2vec2|mms", "0.834"), ("xlsr|mms", "0.830"))
        for tok in ((v, f"R:t5_{pair}_3dp"), ("3,101", f"R:t5_{pair}_n_int"))],
    "table6": [
        ("20", "H:budget_min~int"), ("100", "H:budget_100~int"), ("200", "H:budget_max~int"), ("200", "H:budget_max~int"),
        ("5", "H:kw_min_group~int"),
    ] + [tok for b, i, bound in (("20", 0, -78), ("50", 1, -84), ("100", 2, -94), ("200", 3, -79))
         for tok in ((b, f"H:budget_{i}~int"), (None, f"R:t6row_{b}"),
                     (None, V9 + f"fixed_token_dprime.expA_per_budget.{b}.p~lt10:{bound}"),
                     (None, f"R:t6eps_{b}"))],
}
