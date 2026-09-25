"""v8 copy of v5_investigation/q3d_token_quartiles.py (unchanged logic; output to results/v8).

Phase A Q3d: severity-correlation by n_phones quartile, with p-values.

Definition pinned per results/recompute_report.md: six analysis groups (HC, PD, CP, ALS, DS, Stroke)
including controls; ordinal severity control=0 (aetiology healthy), mild=1, moderate=2, severe=3;
composite = mean of 5 consonant d-primes. Quartile boundaries = 25/50/75th percentiles of n_phones
over the analysed speakers. Reported for >=1 and >=3 composite rules, frozen and corrected master.
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[2]
FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
AET = {"healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"}
SEV = {"mild": 1, "moderate": 2, "severe": 3}


def f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


out = {}
for tag, p in (("frozen", BASE / "frozen_package/results/track4_master.csv"), ("corrected", BASE / "corrected/track4_master.csv")):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for kmin in (1, 3):
        data = []
        for r in rows:
            if r["aetiology"] not in AET:
                continue
            s = 0 if r["aetiology"] == "healthy" else SEV.get(r["severity_label"])
            if s is None:
                continue
            v = [f(r[c]) for c in FEATS]
            v = [x for x in v if x is not None]
            n = f(r["n_phones"])
            if len(v) < kmin or n is None:
                continue
            data.append((n, s, float(np.mean(v))))
        a = np.array(data)
        rho_all = spearmanr(a[:, 1], a[:, 2])
        qs = np.percentile(a[:, 0], [25, 50, 75])
        edges = [-np.inf, *qs, np.inf]
        qres = {}
        for q in range(4):
            m = (a[:, 0] > edges[q]) & (a[:, 0] <= edges[q + 1])
            rr = spearmanr(a[m, 1], a[m, 2])
            qres[f"Q{q+1}"] = {"n_phones_range": [float(a[m, 0].min()), float(a[m, 0].max())], "n": int(m.sum()),
                               "n_dysarthric": int((a[m, 1] > 0).sum()), "rho": round(float(rr.statistic), 4), "p": float(rr.pvalue)}
        out[f"{tag}_min{kmin}"] = {"n": len(a), "rho_all": round(float(rho_all.statistic), 4), "boundaries": [float(x) for x in qs], "quartiles": qres}
(BASE / "results" / "v8" / "v8_token_quartiles.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in out.items():
    print(k, "n", v["n"], "rho_all", v["rho_all"], "bounds", v["boundaries"])
    for q, r in v["quartiles"].items():
        print("   ", q, r)
