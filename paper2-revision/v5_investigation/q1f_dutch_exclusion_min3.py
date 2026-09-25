"""Phase A Q1f: Dutch-exclusion Spearman under >=1 and >=3 rules, corrected inputs, 3 dup rules."""
import csv, json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
CORR = BASE / "corrected"
FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
BB = {"hubert-base": "track4_master.csv", "hubert-large": "track4_results_hubert-large.csv",
      "wavlm": "track4_results_wavlm.csv", "wav2vec2": "track4_results_wav2vec2.csv",
      "xlsr": "track4_results_xlsr.csv", "mms": "track4_results_mms.csv"}


def f(v):
    try:
        x = float(v)
        return x if x == x else np.nan
    except ValueError:
        return np.nan


def load(fn, excl_nl, rule):
    g = defaultdict(list)
    for r in csv.DictReader(open(CORR / fn, encoding="utf-8")):
        if excl_nl and r["language"] == "nl":
            continue
        g[(r["dataset"], r["speaker_id"])].append(np.array([f(r[c]) for c in FEATS]))
    out = {}
    for k, vs in g.items():
        if rule == "last":
            out[k] = vs[-1]
        elif rule == "first":
            out[k] = vs[0]
        elif rule == "drop" and len(vs) == 1:
            out[k] = vs[0]
    return out


def comp(v, kmin):
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) >= kmin else np.nan


res = {}
for kmin in (1, 3):
    for rule in ("last", "first", "drop"):
        for tag, excl in (("with_nl", False), ("without_nl", True)):
            data = {b: load(fn, excl, rule) for b, fn in BB.items()}
            sp, ns = {}, set()
            for b1, b2 in combinations(BB, 2):
                common = sorted(set(data[b1]) & set(data[b2]))
                pr = [(comp(data[b1][k], kmin), comp(data[b2][k], kmin)) for k in common]
                pr = [(x, y) for x, y in pr if x == x and y == y]
                ns.add(len(pr))
                sp[f"{b1}|{b2}"] = round(float(spearmanr([p[0] for p in pr], [p[1] for p in pr]).statistic), 4)
            mn = min(sp, key=sp.get)
            res[f"min{kmin}_{rule}_{tag}"] = {"n_per_pair": sorted(ns), "rho_min": sp[mn], "rho_min_pair": mn,
                                              "rho_max": max(sp.values()), "all": sp}
(Path(__file__).parent / "q1f_dutch_exclusion_min3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
for k, v in res.items():
    print(k, v["n_per_pair"], v["rho_min"], v["rho_min_pair"], v["rho_max"])
