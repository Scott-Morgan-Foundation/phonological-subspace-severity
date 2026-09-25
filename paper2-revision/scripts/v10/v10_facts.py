#!/usr/bin/env python3
"""v10 saved calculations for numbers that previously had no saved source, and the Table 1 severity-source evidence.

Writes results/v10/v10_facts.json:
  coverage             2,961 / 3,374 valid nasality estimates (percentage); the 182 speakers absent from the five
                       non-HuBERT-base files and how many of them also lack any HuBERT-base d-prime (151)
  dutch_exclusion_all  per-aetiology cross-backbone profile-cosine minima (observed, over all 15 backbone pairs)
                       with and without Dutch speakers, for all six groups (PD, CP, ALS, DS, Stroke, HC), using the
                       loaders of scripts/v8/v8_test6.py (imported, unchanged): corrected inputs, master metadata,
                       >= 3 of 5 consonant features, duplicate keys keep-last
  vta                  vowel triangle area group means (six analysis groups) and HC-to-group ratios
  table1_severity_sources  per-dataset severity-label provenance with the file that documents it
  label_source_audit   which analysis scripts read speaker metadata from the backbone files
"""
import hashlib
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[2]
CORR = REV / "corrected"
spec = importlib.util.spec_from_file_location("v8t6", REV / "scripts" / "v8" / "v8_test6.py")
T6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T6)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


m = pd.read_csv(CORR / "track4_master.csv")
out = {"script": "scripts/v10/v10_facts.py", "input_sha256": {"track4_master.csv": sha(CORR / "track4_master.csv")}}

# ---- coverage ------------------------------------------------------------------------------------
D9 = ["back_dprime", "high_dprime", "low_dprime", "manner_dprime", "nasal_dprime", "round_dprime",
      "sonorant_dprime", "strident_dprime", "voicing_dprime"]
nasal_valid = int(m.nasal_dprime.notna().sum())
keys_m = set(zip(m.dataset, m.speaker_id))
b = pd.read_csv(CORR / "track4_results_xlsr.csv")
absent = keys_m - set(zip(b.dataset, b.speaker_id))
absent_rows = m[[k in absent for k in zip(m.dataset, m.speaker_id)]]
out["coverage"] = {"n_speakers": int(len(m)), "nasal_valid": nasal_valid,
                   "nasal_valid_pct": 100.0 * nasal_valid / len(m),
                   "absent_from_non_hubert_base_files": len(absent),
                   "absent_with_no_hubert_base_dprime": int((absent_rows[D9].notna().sum(1) == 0).sum()),
                   "absent_with_hubert_base_dprime": int((absent_rows[D9].notna().sum(1) > 0).sum())}

# ---- Dutch exclusion, all six groups ---------------------------------------------------------------
def pc_min(dat, groups):
    res = {}
    for aet in groups:
        vals = {}
        for b1, b2 in combinations(dat, 2):
            s1 = {k: v["vec"] for k, v in dat[b1].items() if v["aet"] == aet}
            s2 = {k: v["vec"] for k, v in dat[b2].items() if v["aet"] == aet}
            common = sorted(set(s1) & set(s2))
            if len(common) >= 5:
                vals[f"{b1}|{b2}"] = T6.cos(np.nanmean([s1[k] for k in common], 0),
                                            np.nanmean([s2[k] for k in common], 0))
        res[aet] = {"min": min(vals.values()), "min_pair": min(vals, key=vals.get), "n_pairs": len(vals)}
    return res


G = ["PD", "CP", "ALS", "DS", "Stroke", "HC"]
with_nl = {bb: T6.load(fn)[0] for bb, fn in T6.BB.items()}
without_nl = {bb: T6.load(fn, excl_nl=True)[0] for bb, fn in T6.BB.items()}
w, wo = pc_min(with_nl, G), pc_min(without_nl, G)
out["dutch_exclusion_all"] = {"with_nl": w, "without_nl": wo,
                              "abs_change": {g: abs(w[g]["min"] - wo[g]["min"]) for g in G},
                              "max_abs_change": max(abs(w[g]["min"] - wo[g]["min"]) for g in G),
                              "loader": "scripts/v8/v8_test6.py load() (excl_nl=True drops language nl)"}

# ---- LODO (re-implementation, scripts/v6/v6_master_analyses.py): folds that remove at least one speaker ----
lodo = json.loads((REV / "results" / "v6" / "v6_master_analyses.json").read_text(encoding="utf-8"))["lodo"]
full_n = max(f["n"] for f in lodo["folds"].values())
real = {k: f for k, f in lodo["folds"].items() if f["n"] < full_n}
out["lodo_effective"] = {"source": "results/v6/v6_master_analyses.json lodo.folds", "n_labels": len(lodo["folds"]),
                         "full_sample_n": full_n,
                         "labels_removing_no_speaker": sorted(k for k, f in lodo["folds"].items() if f["n"] == full_n),
                         "n_effective": len(real),
                         "rho_range": [min(f["rho"] for f in real.values()), max(f["rho"] for f in real.values())],
                         "rho_mean": float(np.mean([f["rho"] for f in real.values()])),
                         "eps2_range": [min(f["eps2"] for f in real.values()), max(f["eps2"] for f in real.values())],
                         "eps2_mean": float(np.mean([f["eps2"] for f in real.values()])),
                         "max_p_rho": max(f["p"] for f in real.values()), "max_kw_p": max(f["kw_p"] for f in real.values())}

# ---- VTA ------------------------------------------------------------------------------------------
AM = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS", "down_syndrome": "DS",
      "stroke": "Stroke"}
v = m[m.aetiology.isin(AM) & m.vowel_triangle_area.notna()]
means = v.groupby(v.aetiology.map(AM)).vowel_triangle_area.mean().to_dict()
out["vta"] = {"n_valid_six_groups": int(len(v)), "n_valid_all": int(m.vowel_triangle_area.notna().sum()),
              "means": means, "n": v.aetiology.map(AM).value_counts().to_dict(),
              "hc_ratio": {g: means["HC"] / means[g] for g in means if g != "HC"}}

# ---- Table 1 severity sources ------------------------------------------------------------------------
inv = pd.read_csv(CORR / "speaker_inventory.csv")
scale = inv.groupby("dataset").severity_scale.first().to_dict()
sev = {ds: s.severity_label.value_counts().to_dict() for ds, s in m.groupby("dataset")}
S = "frozen_package/scripts/speaker_inventory.py (severity_scale per dataset; stipancic_severity(): >94 control, 85-94 mild, 70-84 moderate, <70 severe)"
out["table1_severity_sources"] = {
    "SAP": {"cell": "SAP intelligibility ratings (1–7), thresholded",
            "evidence": ["corrected/speaker_inventory.csv severity_scale = " + str(scale.get("SAP")),
                         "external/processing_scripts/process_sap.py map_severity(): mean per-utterance intelligibility rating <= 1.5 mild, <= 3.0 moderate, > 3.0 severe (copy of the CANDOR processing script)"],
            "labels_in_master": sev["SAP"]},
    "TORGO": {"cell": "Corpus documentation", "evidence": ["severity_scale = " + str(scale.get("TORGO")), S],
              "labels_in_master": sev["TORGO"]},
    "UASPEECH": {"cell": "Corpus documentation (mild/severe only)", "evidence": ["severity_scale = " + str(scale.get("UASPEECH")), S],
                 "labels_in_master": sev["UASPEECH"]},
    "COPAS": {"cell": "Intelligibility % (Stipancic thresholds)",
              "evidence": ["severity_scale = " + str(scale.get("COPAS")),
                           "intelligibility_pct present for %d of %d COPAS speakers in corrected/speaker_inventory.csv"
                           % (int(inv[inv.dataset == "COPAS"].intelligibility_pct.notna().sum()), int((inv.dataset == "COPAS").sum())), S],
              "labels_in_master": sev["COPAS"]},
    "MDSC": {"cell": "Intelligibility % (Stipancic thresholds)",
             "evidence": ["severity_scale = " + str(scale.get("MDSC")), S,
                          "external/processing_scripts/process_mdsc.py intelligibility_to_severity(): > 94 / 85–94 / 70–84 / < 70 (five-annotator mean)"],
             "labels_in_master": sev["MDSC"]},
    "IPVS": {"cell": "Clinical (CPS3-derived)", "evidence": ["severity_scale = " + str(scale.get("IPVS"))], "labels_in_master": sev["IPVS"]},
    "Neurovoz": {"cell": "Clinical (GRBAS G)", "evidence": ["severity_scale = " + str(scale.get("Neurovoz"))], "labels_in_master": sev["Neurovoz"]},
    "PC-GITA": {"cell": "Clinical (UPDRS speech item)", "evidence": ["severity_scale = " + str(scale.get("PC-GITA"))], "labels_in_master": sev["PC-GITA"]},
    "YouTube_French": {"cell": "Author-assigned from the recordings (not clinical)",
                       "evidence": ["severity_scale = " + str(scale.get("YouTube_French")),
                                    "Docs/research/datasheet.md §22: severity labels are time-based estimates or uniform per source video, not clinical assessments"],
                       "labels_in_master": sev["YouTube_French"]},
    "YouTube_German": {"cell": "Author-assigned from the recordings (not clinical)",
                       "evidence": ["not in the April speaker inventory; Docs/research/datasheet.md §23: severity assigned per source video by the authors (same YouTube pipeline as French)"],
                       "labels_in_master": sev["YouTube_German"]},
    "Hungarian_Dysarthria": {"cell": "Collaborator-supplied labels (15 of 39 speakers)",
                             "evidence": ["frozen_package/scripts/refresh_master_severity.py docstring: Hungarian severity labels supplied by a named collaborator (partial coverage; the corpus itself has no severity labels, Docs/research/datasheet.md §18) and refreshed into the master from the speaker inventory"],
                             "labels_in_master": sev["Hungarian_Dysarthria"]},
    "CHASING": {"cell": "Listener intelligibility ratings, thresholded",
                "evidence": ["external/processing_scripts/process_chasing_radboud.py assign_severity(): mean listener intelligibility (1–5 scale) thresholds"],
                "labels_in_master": sev["CHASING"]},
    "TreasureHunters1": {"cell": "Author-assigned from the published listener results",
                         "evidence": ["external/processing_scripts/process_treasure_hunters1.py: no direct labels; severity assigned from the relative intelligibility reported by Ganzeboom et al. (2018)"],
                         "labels_in_master": sev["TreasureHunters1"]},
    "SSNCE_Tamil": {"cell": "Corpus documentation", "evidence": ["external/processing_scripts/process_ssnce_tamil.py: severity from the corpus's own severity directories"], "labels_in_master": sev["SSNCE_Tamil"]},
    "CDLI_Kenyan_Swahili": {"cell": "Corpus documentation", "evidence": ["external/processing_scripts/process_cdli_kenyan_english.py: severity from the corpus metadata column severity_speech_impairment"], "labels_in_master": sev["CDLI_Kenyan_Swahili"]},
}

# ---- label-source audit ----------------------------------------------------------------------------
out["label_source_audit"] = {
    "scripts/v8/v8_test6.py": "master metadata for every backbone (Table 5, Figure 5, per-backbone severity rho, profile cosines, Kendall's W, Dutch exclusion)",
    "scripts/v10/regen_fig4_v10.py": "master metadata for every panel (Figure 4); v5 script read each file's own labels (fixed in v10)",
    "scripts/v8/originals/cross_model_comparison.py": "original: severity and aetiology labels taken from the master (hubert-base) and joined by key (feature ranking)",
    "scripts/v10/v10_facts.py": "master metadata via v8_test6 loaders",
    "scripts/v9/originals/robustness_pass5.py Experiment 3": "reads each backbone file's own severity_label; its output (final-layer rho) is not reported in the manuscript",
    "scripts/v6/v6_test6.py, scripts/test6_v5.py, scripts/v6/v6_feature_importance.py": "superseded; no reported number comes from them",
    "all other analyses": "HuBERT-base master only",
}

(REV / "results" / "v10").mkdir(parents=True, exist_ok=True)
(REV / "results" / "v10" / "v10_facts.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("coverage", "dutch_exclusion_all", "vta")}, indent=1, default=float))
