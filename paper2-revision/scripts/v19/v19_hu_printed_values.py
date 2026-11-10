#!/usr/bin/env python3
"""v19: printed values under each Hungarian variant (after v19_hu_sensitivity.py run x4 and the v15 consistency
check run inside each mirror). Writes results/v19/hu_sensitivity/printed_values_by_variant.csv with
  - every registry item of the v15 consistency check (the R1 numbers carried into v18), as printed;
  - the second-revision numbers of NUMBERS_R2_v17.md / NUMBERS_R2_v18.md that come from the master
    (matched-severity v12 / v17 outputs), at their printed precision.
The fixed-token numbers are not included: no Hungarian speaker enters that analysis."""
import csv
import json
import os
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("V19_WORK", str(REV.parent / "_v19_hu_mirrors")))
OUT = REV / "results" / "v19" / "hu_sensitivity" / "printed_values_by_variant.csv"
VARS = ["V0_retained", "V1_reextracted", "V2_excl_HD", "V3_excl_all_hu"]
CONS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]


def r2_values(v):
    m = WORK / v / "results"
    msf = json.loads((m / "v12/v12_matched_severity_features.json").read_text(encoding="utf-8"))
    sl = json.loads((m / "v17/v17_matched_severity_samelang.json").read_text(encoding="utf-8"))
    A, B = msf["A_all_corpora"], msf["B_SAP_english"]
    out = {}
    for s in ("mild", "moderate"):
        fs = A[s]["features"]
        for g in ("PD", "CP", "ALS", "DS"):
            out[f"R2:{s}_n_{g}"] = str(max((r["n"].get(g, 0) for r in fs.values()), default=0))
        out[f"R2:{s}_n_sig_holm"] = str(A[s]["summary"]["n_sig_holm_05"])
        out[f"R2:{s}_n_large"] = str(A[s]["summary"]["n_eps2_gt_0.14"])
        for f, r in fs.items():
            if "eps2" in r:
                out[f"R2:T4_{s}_{f}"] = f"{r['eps2']:.3f}{'*' if r['p_holm'] < 0.05 else ''}"
        out[f"R2:{s}_max_consonant_eps2"] = f"{max(fs[c]['eps2'] for c in CONS):.4f}"
        out[f"R2:{s}_sig_features"] = ",".join(f.replace('_dprime', '') for f, r in fs.items()
                                               if r.get("p_holm", 1) < 0.05)
    out["R2:SAP_mild_n_sig_holm"] = str(B["mild"]["summary"]["n_sig_holm_05"])
    ps = A["profile_shape"]
    out["R2:pooled_centred_stat"] = f"{ps['centred']['stat']:.1f}"
    out["R2:pooled_centred_null"] = f"{ps['centred']['null_mean']:.1f}"
    out["R2:pooled_centred_p"] = f"{ps['centred']['p']:.4f}"
    mild = ps["centred_profiles"]["mild"]
    feats = ps["profile_features"]
    vi, si = feats.index("voicing"), feats.index("sonorant")
    out["R2:pooled_mild_PD_voicing"] = f"{mild['PD'][vi]:+.2f}"
    out["R2:pooled_mild_PD_sonorance"] = f"{mild['PD'][si]:+.2f}"
    oth = [mild[g] for g in ("CP", "ALS", "DS")]
    out["R2:pooled_mild_CPALSDS_voicing_range"] = f"{min(x[vi] for x in oth):+.2f}..{max(x[vi] for x in oth):+.2f}"
    out["R2:pooled_mild_CPALSDS_sonorance_range"] = f"{min(x[si] for x in oth):+.2f}..{max(x[si] for x in oth):+.2f}"
    out["R2:hc_n_for_z"] = str(msf["hc_n_for_z"])
    SA = sl["A_all_corpora"]
    out["R2:samelang_centred_stat"] = f"{SA['centred']['stat']:.1f}"
    out["R2:samelang_centred_null"] = f"{SA['centred']['null_mean']:.1f}"
    out["R2:samelang_centred_p"] = f"{SA['centred']['p']:.3f}"
    sm = SA["centred_profiles"]["mild"]
    out["R2:samelang_mild_PD_max_abs"] = f"{max(abs(x) for x in sm['PD']):.3f}"
    so = [sm[g] for g in ("CP", "ALS", "DS")]
    out["R2:samelang_mild_CPALSDS_sonorance_range"] = f"{min(x[si] for x in so):+.2f}..{max(x[si] for x in so):+.2f}"
    out["R2:samelang_cells"] = json.dumps(SA["cells"], sort_keys=True)
    out["R2:pooled_cells"] = json.dumps(ps["cells"], sort_keys=True)
    out["R2:SAP_samelang_p"] = f"{sl['B_SAP_english']['centred']['p']:.3f}"
    return out


rows = {}
for v in VARS:
    rep = json.loads((WORK / v / "results/v15/check_consistency_report.json").read_text(encoding="utf-8"))
    for e in rep["registry"]:
        rows.setdefault(e["id"], {"src": e["src"] + ":" + ".".join(map(str, e["path"]))})[v] = e["printed"]
    for b in rep["high_risk"]["bindings"]:
        rows.setdefault(f"{b['region']}|{b['ref']}", {"src": "high-risk binding (v15 hr_bindings)"})[v] = b["value"]
    for k, val in r2_values(v).items():
        rows.setdefault(k, {"src": "R2 (v12/v17 outputs)"})[v] = val
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "source"] + VARS + ["changed_V1", "changed_V2", "changed_V3"])
    for k, r in rows.items():
        w.writerow([k, r["src"]] + [r.get(v, "") for v in VARS] +
                   [r.get(v, "") != r.get("V0_retained", "") for v in VARS[1:]])
n = {v: sum(r.get(v, "") != r.get("V0_retained", "") for r in rows.values()) for v in VARS[1:]}
print(len(rows), "items; changed:", n)
