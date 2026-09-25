#!/usr/bin/env python3
"""v8: saved calculations for master-table facts cited in the manuscript (v7 check P2 / NV-3 / NV-4).

All from corrected/track4_master.csv. Writes results/v8/v8_master_facts.json with:
  - per-dataset composition (aetiology x severity label x language) for Table 1 (#12, #13);
  - YouTube French / German composition (Ethics section, #14);
  - SVD composition and feature availability (§5.4);
  - Portuguese and French dysarthric counts (§4.2);
  - vowel triangle area means and n per group, and ratios to HC (§4.1, NV C-26);
  - Kruskal-Wallis across languages on raw d-prime within aetiology (§4.2 / §5.1 "p < 0.001");
  - block structure of the Test 2 permutation (blocks, blocks mixing >1 aetiology, speakers in
    mixing blocks, of which SAP), using the Test 2 script's speaker rule and (language, dataset) blocks.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v6"))
from common import CORR, MASTER, C5, F13, MAIN, load, f, sha256  # noqa: E402

REV = Path(__file__).resolve().parents[2]
OUT = REV / "results" / "v8" / "v8_master_facts.json"
rows = load(MASTER)

# ---- Table 1 composition --------------------------------------------------------------
table1 = {}
for ds in sorted({r["dataset"] for r in rows}):
    rr = [r for r in rows if r["dataset"] == ds]
    dys = [r for r in rr if r["is_control"] != "True"]
    table1[ds] = {
        "n": len(rr),
        "language": dict(Counter(r["language"] for r in rr)),
        "aetiology": dict(Counter(r["aetiology"] for r in rr)),
        "is_control_true": sum(r["is_control"] == "True" for r in rr),
        "severity_all": dict(Counter(r["severity_label"] for r in rr)),
        "severity_non_control_speakers": dict(Counter(r["severity_label"] for r in dys)),
        "n_non_control_with_graded_severity": sum(r["severity_label"] in ("mild", "moderate", "severe") for r in dys),
    }

# ---- VTA ------------------------------------------------------------------------------
AM = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
      "down_syndrome": "DS", "stroke": "Stroke"}
vta = defaultdict(list)
for r in rows:
    if r["aetiology"] in AM and f(r["vowel_triangle_area"]) is not None:
        vta[AM[r["aetiology"]]].append(f(r["vowel_triangle_area"]))
vta_means = {g: float(np.mean(v)) for g, v in vta.items()}
vta_out = {"mean": vta_means, "n": {g: len(v) for g, v in vta.items()},
           "n_any_group_valid": sum(1 for r in rows if f(r["vowel_triangle_area"]) is not None),
           "n_main_groups_valid": sum(len(v) for v in vta.values()),
           "ratio_hc_to_group": {g: vta_means["HC"] / m for g, m in vta_means.items() if g != "HC"}}

# ---- SVD / PT / FR --------------------------------------------------------------------
svd = [r for r in rows if r["dataset"] == "SVD"]
svd_out = {"n": len(svd), "aetiology": dict(Counter(r["aetiology"] for r in svd)),
           "features_valid_n": {c: sum(f(r[c]) is not None for r in svd) for c in C5 + ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]}}
lang_dys = {lg: dict(Counter(f'{r["dataset"]}|{r["aetiology"]}' for r in rows
                             if r["language"] == lg and r["is_control"] != "True" and r["aetiology"] != "healthy"))
            for lg in ("pt", "fr", "de")}

# ---- KW across languages within aetiology (raw d-prime) -------------------------------
kw_lang = {}
for aet in ("parkinsons", "cerebral_palsy", "als", "healthy"):
    kw_lang[aet] = {}
    for c in C5 + ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]:
        g = defaultdict(list)
        for r in rows:
            if r["aetiology"] == aet and f(r[c]) is not None:
                g[r["language"]].append(f(r[c]))
        groups = [v for v in g.values() if len(v) >= 3]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            kw_lang[aet][c] = {"H": float(H), "p": float(p), "k": len(groups), "N": sum(map(len, groups))}
kw_lang["all_speakers_pooled"] = {}
for c in C5 + ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]:
    g = defaultdict(list)
    for r in rows:
        if f(r[c]) is not None:
            g[r["language"]].append(f(r[c]))
    groups = [v for v in g.values() if len(v) >= 3]
    H, p = kruskal(*groups)
    kw_lang["all_speakers_pooled"][c] = {"H": float(H), "p": float(p), "k": len(groups), "N": sum(map(len, groups))}
max_p = {a: max(v["p"] for v in d.values()) for a, d in kw_lang.items()}

# ---- Test 2 block structure ---------------------------------------------------------
# Same speaker rule as scripts/test2_aetiology_vs_language.py load(): dysarthric = aetiology in
# PD/CP/ALS and is_control != True; permutation blocks = (language, dataset) (plan §3).
AETS3 = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS"}
blk = defaultdict(list)
for r in rows:
    if r["aetiology"] in AETS3 and r["is_control"].lower() != "true":
        blk[(r["language"], r["dataset"])].append(AETS3[r["aetiology"]])
mixing = {k: v for k, v in blk.items() if len(set(v)) > 1}
blocks = {"n_blocks": len(blk), "n_mixing_blocks": len(mixing),
          "n_speakers_in_mixing_blocks": sum(len(v) for v in mixing.values()),
          "n_speakers_in_mixing_blocks_SAP": sum(len(v) for k, v in mixing.items() if k[1] == "SAP"),
          "mixing_blocks": {f"{k[0]}|{k[1]}": dict(Counter(v)) for k, v in mixing.items()},
          "n_dysarthric_PD_CP_ALS": sum(len(v) for v in blk.values())}

out = {"definition": __doc__, "inputs": {"corrected/track4_master.csv": sha256(MASTER)},
       "table1": table1, "vta": vta_out, "svd": svd_out, "dysarthric_by_language": lang_dys,
       "kw_across_languages": kw_lang, "kw_across_languages_max_p": max_p,
       "test2_blocks": blocks}
OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", OUT)
print(json.dumps({k: out[k] for k in ("vta", "svd", "dysarthric_by_language", "kw_across_languages_max_p",
                                      "test2_blocks")}, indent=1))
