"""v6: per-backbone feature ranking (§4.3) on the corrected inputs.

For each backbone and each of the 5 consonant d-primes: Kruskal-Wallis epsilon-squared
(i) across the six aetiology groups and (ii) across the four severity levels
(convention coding: controls = 0, six groups only). Repeated (dataset, speaker) keys
keep the last row (see Methods). Ranking = descending epsilon-squared.
"""
import numpy as np
import pandas as pd
from scipy.stats import kruskal

from common import CORR, BACKBONES, C5, SEV, write

AM = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS", "healthy": "HC",
      "down_syndrome": "DS", "stroke": "Stroke"}


def e2(groups):
    groups = [g for g in groups if len(g) >= 5]
    H, _ = kruskal(*groups)
    N, k = sum(len(g) for g in groups), len(groups)
    return float((H - k + 1) / (N - k))


out = {"definition": __doc__, "backbones": {}}
paths = []
for name, fn in BACKBONES.items():
    p = CORR / fn
    paths.append(p)
    d = pd.read_csv(p).drop_duplicates(["dataset", "speaker_id"], keep="last")
    d["g"] = d.aetiology.map(AM)
    d["s"] = d.severity_label.map(SEV)
    d.loc[d.is_control.astype(str) == "True", "s"] = 0
    d.loc[d.g.isna(), "s"] = np.nan
    a = {c: e2([x[c].dropna().values for _, x in d[d.g.notna()].groupby("g")]) for c in C5}
    s = {c: e2([x[c].dropna().values for _, x in d[d.s.notna()].groupby("s")]) for c in C5}
    out["backbones"][name] = {"aetiology_eps2": a, "aetiology_rank": sorted(a, key=a.get, reverse=True),
                              "severity_eps2": s, "severity_rank": sorted(s, key=s.get, reverse=True)}
for kind in ("aetiology_rank", "severity_rank"):
    ranks = {k: v[kind] for k, v in out["backbones"].items()}
    first = {}
    for r in ranks.values():
        first[r[0]] = first.get(r[0], 0) + 1
    top3 = [set(r[:3]) for r in ranks.values()]
    out[kind + "_summary"] = {"first_counts": first, "top3_common_to_all": sorted(set.intersection(*top3)),
                              "top3_by_backbone": {k: r[:3] for k, r in ranks.items()}}
write("v6_feature_importance.json", out, paths)
for kind in ("aetiology_rank_summary", "severity_rank_summary"):
    print(kind, out[kind])
