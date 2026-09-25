#!/usr/bin/env python3
"""v10: saved values for every number in the high-risk regions (abstract, Highlights, contributions, tables, figure
captions, Conclusion) that has no direct entry in an analysis output: design counts, tallies, threshold constants read
from the scripts that apply them, companion-paper values read from the companion manuscript, Table 1 counts from the
master, and the stated conventions. Each value carries its source. Output: results/v10/v10_hr_values.json.
External model descriptors (Table 2) are recorded separately in results/v10/v10_external_descriptors.json."""
import hashlib
import importlib.util
import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[2]
CORR = REV / "corrected"
V = {}


def put(key, value, source):
    V[key] = {"value": value, "source": source}


def jload(p):
    return json.loads((REV / p).read_text(encoding="utf-8"))


def src_const(path, pattern, cast=int):
    t = (REV / path).read_text(encoding="utf-8")
    m = re.search(pattern, t)
    if not m:
        raise SystemExit(f"constant not found in {path}: {pattern}")
    return cast(m.group(1))


m = pd.read_csv(CORR / "track4_master.csv")
spec = importlib.util.spec_from_file_location("v8t6", REV / "scripts" / "v8" / "v8_test6.py")
T6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T6)

# ---- design --------------------------------------------------------------------------------------------
put("n_speakers", int(len(m)), "corrected/track4_master.csv rows")
labels = sorted(m.dataset.unique())
put("n_dataset_labels", len(labels), "corrected/track4_master.csv distinct dataset labels")
put("n_datasets", len(labels) - 1, "dataset labels minus 1 (UASPEECH and UASPEECH_control are one dataset)")
put("n_languages", int(m.language.nunique()), "corrected/track4_master.csv distinct languages")
put("n_aetiologies", len(T6.AETS), "scripts/v8/v8_test6.py AETS (PD, CP, ALS, DS, stroke)")
put("n_backbones", len(T6.BB), "scripts/v8/v8_test6.py BB")
put("n_backbone_pairs", len(list(combinations(T6.BB, 2))), "pairs of scripts/v8/v8_test6.py BB")
put("n_consonant_features", len(T6.FEATS), "scripts/v8/v8_test6.py FEATS")
put("min_features", T6.MIN_FEATURES, "scripts/v8/v8_test6.py MIN_FEATURES")
t3 = jload("results/rebuild_2026-09-23/table3_regenerated.json")["rows"]
excluded = {"boundary_sharpness", "cross_position_cosim"}
main = [r for r in t3 if r["column"] not in excluded]
put("n_features_all", len(t3), "results/rebuild_2026-09-23/table3_regenerated.json rows")
put("n_features_main", len(main), "table3 rows minus boundary sharpness and cross-position cosine")
put("n_segmental", len([r for r in t3 if r["column"].endswith("_dprime")]), "table3 rows whose column ends _dprime")
put("large_threshold", 0.14, "stated convention (Section 4.1; Cohen 1988 heuristics: small < 0.06, medium 0.06-0.14, large > 0.14)")
put("medium_threshold", 0.06, "stated convention (Section 4.1)")
put("n_large_main", sum(1 for r in main if r["epsilon_squared"] > 0.14), "table3 main rows with epsilon_squared > 0.14")
put("kw_p_max_main", max(r["p"] for r in main), "max Kruskal-Wallis p over the 13 main features (table3)")
put("min_cell", jload("results/test2_delta_corrected.json")["min_cell"], "results/test2_delta_corrected.json min_cell")
put("kw_min_group", src_const("scripts/v9/originals/fixed_token_dprime.py", r"len\(g\) >= (\d+)"),
    "scripts/v9/originals/fixed_token_dprime.py (aetiology groups with >= 5 speakers)")
put("fig1_min_cell", src_const("scripts/v6/v6_figures_1_3.py", r"if N\[i, j\] >= (\d+)"), "scripts/v6/v6_figures_1_3.py")
put("min_n_threshold_top", max(int(x) for x in re.search(r"for min_n in \[([\d, ]+)\]",
                                                          (REV / "scripts/v8/originals/reviewer_experiments.py").read_text(encoding="utf-8")).group(1).split(",")),
    "scripts/v8/originals/reviewer_experiments.py min_n thresholds")
put("fig4_n_boot", src_const("scripts/v10/regen_fig4_v10.py", r"SEED, N_BOOT = \d+, (\d+)"), "scripts/v10/regen_fig4_v10.py")
put("a2_n_boot", 1000, "scripts/v9/originals/robustness_analyses.py Analysis 2 (1,000 speaker resamples per language pair; stated in its header)")
bud = [int(x) for x in re.search(r"TOKEN_BUDGETS = \[([\d, ]+)\]",
                                  (REV / "scripts/v9/originals/fixed_token_dprime.py").read_text(encoding="utf-8")).group(1).split(",")]
put("budget_min", bud[0], "scripts/v9/originals/fixed_token_dprime.py TOKEN_BUDGETS")
put("budget_100", bud[2], "scripts/v9/originals/fixed_token_dprime.py TOKEN_BUDGETS")
put("budget_max", bud[-1], "scripts/v9/originals/fixed_token_dprime.py TOKEN_BUDGETS")
for i, b in enumerate(bud):
    put(f"budget_{i}", b, "scripts/v9/originals/fixed_token_dprime.py TOKEN_BUDGETS")
put("hc_reference", 1.0, "definition: HC-normalised values are ratios to the same-language healthy-control mean (HC = 1.0)")

# ---- pairwise d, moderate-only ----------------------------------------------------------------------------------
pw = jload("results/rebuild_2026-09-23/pairwise_d_extracted.json")["data"]
put("pd_exec_mean_d", (pw["PD-CP"] + pw["PD-DS"] + pw["PD-Stroke"]) / 3, "mean of PD-CP, PD-DS, PD-Stroke in pairwise_d_extracted.json")
put("cp_stroke_d", pw["CP-Stroke"], "pairwise_d_extracted.json CP-Stroke")
mo = jload("results/rebuild_2026-09-23/pairwise_moderate_only.json")
put("mod_d", mo["cohen_d_PD_vs_exec_moderate_only"], "pairwise_moderate_only.json")
put("mod_pd_n", mo["PD_n"], "pairwise_moderate_only.json")
put("mod_exec_n", mo["exec_n"], "pairwise_moderate_only.json")

# ---- per-feature cross-lingual replication tallies (test3) -----------------------------------------------------
t3e = pd.read_csv(REV / "results" / "test3_effects_corrected.csv")
cp = t3e[(t3e.aetiology == "CP") & t3e.smd.notna()]
pd_ = t3e[(t3e.aetiology == "PD") & t3e.smd.notna()]
put("cp_cells", int(len(cp)), "test3_effects_corrected.csv CP rows with an estimate")
put("cp_pos", int((cp.smd > 0).sum()), "CP rows with smd > 0")
put("cp_ci_excl0", int(((cp.ci_lo > 0) | (cp.ci_hi < 0)).sum()), "CP rows whose interval excludes 0")
put("pd_cells", int(len(pd_)), "test3 PD rows with an estimate")
put("pd_pos", int((pd_.smd > 0).sum()), "PD rows with smd > 0")

# ---- cross-lingual language counts (contribution (b)) ---------------------------------------------------------------
a2 = jload("results/v9/v9_originals.json")["states"]["corrected"]["parsed"]["robustness_analyses"]["a2_crosslingual_bootstrap"]
for g in ("PD", "CP", "ALS"):
    put(f"b_{g}_nlang", a2[g]["n_languages"], "v9_originals robustness_analyses a2")
cells = jload("results/test2_delta_corrected.json")["cells"]
put("de_als_n", cells["de|ALS"], "test2_delta_corrected.json cells de|ALS")
f3 = jload("results/v6/v6_figures_1_3.json")["fig3"]
put("fig3_n_langs", len(f3["languages"]), "v6_figures_1_3.json fig3 languages")
put("fig3_nl_pd", f3["n_pd"]["nl"], "v6_figures_1_3.json fig3 n_pd nl")
put("fig3_pt_pd", f3["n_pd"]["pt"], "v6_figures_1_3.json fig3 n_pd pt")
f1 = jload("results/v6/v6_figures_1_3.json")["fig1"]
put("fig1_n_features", len(f1["features"]), "v6_figures_1_3.json fig1 features")
put("fig1_n_aetiologies", len(f1["aetiologies"]), "v6_figures_1_3.json fig1 aetiologies")
fa = jload("results/v10/v10_facts.json")
put("vta_ds_n", fa["vta"]["n"]["DS"], "v10_facts.json vta n DS")

put("b_min_mean_pd_cp_als", min(a2[g]["mean_cosine"] for g in ("PD", "CP", "ALS")),
    "min of the PD, CP and ALS mean cosines (v9_originals robustness_analyses a2); compared with the stated 0.95")
t6j = jload("results/v8/v8_test6.json")["per_pair"]
fam = ["hubert-base|hubert-large", "hubert-base|wavlm", "hubert-large|wavlm"]
put("family_min_rho", min(t6j[p]["rho"] for p in fam), "min per-speaker rho over the HuBERT-base/HuBERT-large/WavLM pairs (v8_test6.json)")
holm = jload("results/v9/v9_originals.json")["states"]["corrected"]["parsed"]["reviewer_experiments"]["exp3_holm"]["pairs"]
put("holm_n_sig", sum(1 for p in holm if p["sig"] != "ns"), "Holm pairs with p_holm below the reported level (v9_originals exp3_holm sig != ns)")
put("holm_n_pairs", len(holm), "Holm pairs (v9_originals exp3_holm)")
put("holm_ns_min_p", min(p["p_holm"] for p in holm if p["sig"] == "ns"), "min Holm-adjusted p among the non-significant pairs (v9_originals exp3_holm)")
tmj = jload("results/v8/v8_originals.json")["parsed"]["corrected"]["token_matched"]
put("tm_p_max", max(v["p"] for v in tmj.values()), "max p over the three token-matched comparisons (v8_originals.json)")
cl = jload("results/v6/v6_crossling.json")["min_n_3"]
put("t4_nlang", {g: len(cl[g]["languages"]) for g in ("CP", "ALS", "PD", "HC")}, "v6_crossling.json min_n_3 languages")

# ---- random-direction floor, classifier, ablation --------------------------------------------------------------------
put("random_floor", jload("results/v6/v6_test45_analysis.json")["test5_random_cell_cosine"]["mean"], "v6_test45_analysis.json test5_random_cell_cosine.mean")
clf = jload("results/v6/v6_classifier.json")["corrected"]
put("clf_macro_f1_pct", 100 * clf["raw_cosine_train_mean"]["macro_f1"], "v6_classifier.json corrected raw_cosine_train_mean macro_f1 x 100")
ab = jload("results/v6/v6_master_analyses.json")["severity_source_ablation"]["clinical_only"]
put("abl_rho", ab["rho"], "v6_master_analyses.json severity_source_ablation.clinical_only")
put("abl_eps2", ab["eps2"], "v6_master_analyses.json severity_source_ablation.clinical_only")

# ---- companion paper (read from the companion manuscript) -------------------------------------------------------------
p1 = (REV / "external" / "paper1_arxiv_manuscript.tex").read_text(encoding="utf-8")
mm = re.search(r"(\d{3,4}) speakers", p1)
put("p1_speakers", int(mm.group(1).replace(",", "")), "external/paper1_arxiv_manuscript.tex (first 'N speakers')")
ml = re.search(r"(\d+) languages", p1)
put("p1_languages", int(ml.group(1)), "external/paper1_arxiv_manuscript.tex (first 'N languages')")

# ---- Table 1 (master) --------------------------------------------------------------------------------------------------
put("t1_dataset_n", {d: int(n) for d, n in m.dataset.value_counts().items()}, "corrected/track4_master.csv per dataset")
put("t1_language_n", {l: int(n) for l, n in m.language.value_counts().items()}, "corrected/track4_master.csv per language")
hu = m[m.dataset == "Hungarian_Dysarthria"]
put("t1_hu_english", int((hu.language == "en").sum()), "Hungarian_Dysarthria speakers with language en")
put("t1_hu_labelled", int((hu.severity_label != "unknown").sum()), "Hungarian_Dysarthria speakers with a severity label")
put("t1_hu_total", int(len(hu)), "Hungarian_Dysarthria speakers")
put("sap_scale_min", 1, "external/processing_scripts/process_sap.py (SAP intelligibility ratings on a 1-7 scale)")
put("sap_scale_max", 7, "external/processing_scripts/process_sap.py (SAP intelligibility ratings on a 1-7 scale)")

out = {"script": "scripts/v10/v10_hr_values.py",
       "input_sha256": {"corrected/track4_master.csv": hashlib.sha256((CORR / "track4_master.csv").read_bytes()).hexdigest()},
       "values": V}
(REV / "results" / "v10" / "v10_hr_values.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(len(V), "values")
print({k: v["value"] for k, v in V.items() if not isinstance(v["value"], dict)})
