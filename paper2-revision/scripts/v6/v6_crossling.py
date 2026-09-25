"""v6: cross-lingual cosine analyses on the corrected master.

Table 4 (cells >= 3 speakers), language minimum-n (>= 5, >= 10), minimum-HC sensitivity,
severe-CP retention by language, Swahili anecdote, German ALS cell size.
Profiles: mean of the 5 consonant d-primes over speakers with >= 1 valid consonant feature
(nan-mean per feature). HC row excludes Swahili (single control speaker).
CIs: speaker bootstrap within every cell (1,000 resamples, seed 42), percentile 95%.
"""
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from common import MASTER, C5, write

SEED, NB = 42, 1000
AM = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS", "healthy": "HC",
      "down_syndrome": "DS", "stroke": "Stroke"}

m = pd.read_csv(MASTER)
m["g"] = m.aetiology.map(AM)
m["nv"] = m[C5].notna().sum(1)


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def cells(g, thr, langs=None):
    d = m[(m.g == g) & (m.nv >= 1)]
    out = {}
    for l, x in d.groupby("language"):
        if len(x) < thr or (g == "HC" and l == "sw"):
            continue
        if langs is not None and l not in langs:
            continue
        out[l] = x[C5].to_numpy(float)
    return out


def summary(cl, rng):
    ls = sorted(cl)
    prof = {l: np.nanmean(cl[l], 0) for l in ls}
    pairs = {f"{a}|{b}": cos(prof[a], prof[b]) for a, b in itertools.combinations(ls, 2)}
    vals = list(pairs.values())
    boots = []
    for _ in range(NB):
        pb = {}
        for l in ls:
            X = cl[l]
            pb[l] = np.nanmean(X[rng.integers(0, len(X), len(X))], 0)
        boots.append(np.mean([cos(pb[a], pb[b]) for a, b in itertools.combinations(ls, 2)]))
    return {"languages": ls, "n_per_language": {l: int(len(cl[l])) for l in ls},
            "mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
            "min_pair": min(pairs, key=pairs.get), "ci95": [float(np.percentile(boots, 2.5)),
                                                            float(np.percentile(boots, 97.5))],
            "pairs": pairs}


res = {"definition": __doc__}
rng = np.random.default_rng(SEED)
for thr in (3, 5, 10):
    res[f"min_n_{thr}"] = {}
    for g in ("PD", "CP", "ALS", "HC"):
        cl = cells(g, thr)
        res[f"min_n_{thr}"][g] = summary(cl, rng) if len(cl) >= 2 else {"languages": sorted(cl), "note": "<2 languages"}

# minimum-HC sensitivity: languages must have >= h healthy speakers (all, not only valid)
hc_n = m[m.g == "HC"].groupby("language").size().to_dict()
res["hc_per_language"] = {k: int(v) for k, v in hc_n.items()}
res["min_hc"] = {}
for h in (1, 5, 10, 20):
    ok = {l for l, n in hc_n.items() if n >= h}
    res["min_hc"][h] = {}
    for g in ("PD", "CP", "ALS"):
        cl = cells(g, 3, ok)
        res["min_hc"][h][g] = summary(cl, rng) if len(cl) >= 2 else {"languages": sorted(cl), "note": "<2 languages"}

# severe-CP retention: severe CP mean / same-language HC mean, per feature
ret = {}
for l in sorted(m[m.g == "CP"].language.unique()):
    sev = m[(m.g == "CP") & (m.language == l) & (m.severity_label == "severe")]
    hc = m[(m.g == "HC") & (m.language == l)]
    if len(sev) < 2 or len(hc) < 1:
        continue
    r5 = {c: float(sev[c].mean() / hc[c].mean()) for c in C5 if sev[c].notna().any() and hc[c].notna().any()}
    ret[l] = {"n_severe_cp": int(len(sev)), "n_hc": int(len(hc)), "ratio_5c": r5,
              "range_5c": [min(r5.values()), max(r5.values())] if r5 else None}
res["severe_cp_retention"] = ret

# Swahili anecdote
def prof(mask):
    return m[mask][C5].mean().to_numpy(float)
res["swahili"] = {
    "n_pd_sw": int(((m.language == "sw") & (m.g == "PD")).sum()),
    "n_cp_sw": int(((m.language == "sw") & (m.g == "CP")).sum()),
    "n_hc_sw": int(((m.language == "sw") & (m.g == "HC")).sum()),
    "pd_sw_vs_all_pd": cos(prof((m.language == "sw") & (m.g == "PD")), prof(m.g == "PD")),
    "pd_sw_vs_sk_pd": cos(prof((m.language == "sw") & (m.g == "PD")), prof((m.language == "sk") & (m.g == "PD"))),
    "cp_sw_vs_zh_cp": cos(prof((m.language == "sw") & (m.g == "CP")), prof((m.language == "zh") & (m.g == "CP"))),
}

# German ALS cell composition
de = m[(m.language == "de") & (m.g == "ALS")]
res["german_als"] = {"n_speakers": int(len(de)), "by_dataset": de.dataset.value_counts().to_dict(),
                     "n_valid_per_feature": {c: int(de[c].notna().sum()) for c in C5}}

# DS / stroke languages and datasets
res["ds_stroke_by_language_dataset"] = {
    g: m[m.g == g].groupby(["language", "dataset"]).size().rename(lambda t: f"{t[0]}|{t[1]}", axis=0).to_dict()
    if False else {f"{a}|{b}": int(n) for (a, b), n in m[m.g == g].groupby(["language", "dataset"]).size().items()}
    for g in ("DS", "Stroke")}

write("v6_crossling.json", res, [MASTER])
for k in ("min_n_3", "min_n_5", "min_n_10"):
    for g, v in res[k].items():
        if "mean" in v:
            print(k, g, len(v["languages"]), round(v["mean"], 4), [round(x, 4) for x in v["ci95"]],
                  round(v["min"], 4), round(v["max"], 4))
        else:
            print(k, g, v)
for h, d in res["min_hc"].items():
    print("min_hc", h, {g: (len(v["languages"]), round(v.get("mean", float("nan")), 4)) for g, v in d.items()})
print({l: [round(x, 3) for x in v["range_5c"]] for l, v in ret.items()})
print(res["swahili"], res["german_als"], res["ds_stroke_by_language_dataset"])
