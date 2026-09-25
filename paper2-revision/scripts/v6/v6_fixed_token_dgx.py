#!/usr/bin/env python3
"""v6 fixed-token d-prime (§4.5 Table 6, §5.2) on the corrected master. Runs on the DGX.

Identical estimator to frozen_package/scripts/fixed_token_dprime.py (Experiment A) and
robustness_pass5.py (Experiment 1: common speaker set): per speaker and consonant contrast,
subsample exactly N tokens per class (skip if fewer), d-prime along that speaker's own
class-mean difference axis, averaged over 50 draws; composite over >= 3 contrasts.
Changes from the submitted run: labels come from the corrected master, and severity uses the
stated convention (six analysis groups; healthy controls = 0; others by severity label).
Usage: python v6_fixed_token_dgx.py <corrected_master.csv> <out.json>
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

MASTER = Path(sys.argv[1])
OUT = Path(sys.argv[2])
BASE = Path.home() / "dysarthria"
EMB_DIR = BASE / "results" / "track4" / "embeddings"
CONFIG_DIR = BASE / "config"
SEV = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
MAIN = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
CONTRASTS = ["nasal", "voicing", "sonorant", "strident", "manner"]
BUDGETS = [20, 50, 100, 200]
N_REPEATS = 50
LANG_MAP = {"en": "en", "nl": "nl", "es": "es", "fr": "fr", "zh": "zh", "it": "it", "de": "de",
            "pt": "pt", "hu": "hu", "ta": "ta", "sk": "de", "sw": "en"}


def sev_code(r):
    if r["aetiology"] not in MAIN:
        return None
    if r["is_control"] == "True":
        return 0
    return SEV.get(r["severity_label"])


def dprime(p, n):
    if len(p) < 2 or len(n) < 2:
        return np.nan
    d = p.mean(0) - n.mean(0)
    nr = np.linalg.norm(d)
    if nr < 1e-10:
        return 0.0
    d /= nr
    pp, pn = p @ d, n @ d
    s = np.sqrt((pp.var(ddof=1) + pn.var(ddof=1)) / 2)
    return 0.0 if s < 1e-10 else float((pp.mean() - pn.mean()) / s)


def classes(pf, c):
    mp = pf.get("consonant_features", {}).get(c, {})
    return set(mp.get("positive", [])), set(mp.get("negative", []))


rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
meta = {f"{r['dataset']}__{r['speaker_id']}": r for r in rows}
pfs = {}
for l in set(LANG_MAP.values()):
    p = CONFIG_DIR / f"phone_features_{l}.json"
    if p.exists():
        pfs[l] = json.load(open(p))

spk = []
for f in sorted(EMB_DIR.glob("*.npz")):
    r = meta.get(f.stem)
    if r is None:
        continue
    s = sev_code(r)
    pf = pfs.get(LANG_MAP.get(r["language"], "en"))
    if s is None or pf is None:
        continue
    z = np.load(f, allow_pickle=True)
    ph = list(z["phones"])
    idx = {}
    for c in CONTRASTS:
        pos, neg = classes(pf, c)
        idx[c] = ([i for i, x in enumerate(ph) if x in pos], [i for i, x in enumerate(ph) if x in neg])
    spk.append({"key": f.stem, "sev": s, "aet": r["aetiology"], "path": f, "idx": idx})
print("eligible speakers with embeddings:", len(spk), flush=True)


def composite(sp, budget, rng, emb):
    vals = []
    for c in CONTRASTS:
        pi, ni = sp["idx"][c]
        if len(pi) < budget or len(ni) < budget:
            continue
        ds = [dprime(emb[rng.choice(pi, budget, replace=False)], emb[rng.choice(ni, budget, replace=False)])
              for _ in range(N_REPEATS)]
        vals.append(np.mean(ds))
    return float(np.mean(vals)) if len(vals) >= 3 else None


def summarise(res):
    s = np.array([x[0] for x in res]); v = np.array([x[1] for x in res]); a = [x[2] for x in res]
    rho, p = stats.spearmanr(s, v)
    g = defaultdict(list)
    for ai, vi in zip(a, v):
        g[ai].append(vi)
    gv = [x for x in g.values() if len(x) >= 5]
    H, pk = stats.kruskal(*gv)
    N, k = sum(len(x) for x in gv), len(gv)
    return {"n": len(res), "rho": float(rho), "p": float(p), "eps2": float((H - k + 1) / (N - k)),
            "kw_p": float(pk), "severity_means": {lab: float(v[s == c].mean()) for lab, c in SEV.items() if (s == c).any()},
            "severity_n": {lab: int((s == c).sum()) for lab, c in SEV.items()}}


out = {"master_sha256": hashlib.sha256(open(MASTER, "rb").read()).hexdigest(),
       "n_npz_in_cache": len(list(EMB_DIR.glob("*.npz"))), "per_budget": {}, "common_set": {}}
rng = np.random.RandomState(42)
cache = {}
for b in BUDGETS:
    res = []
    for sp in spk:
        emb = np.load(sp["path"], allow_pickle=True)["embeddings"]
        c = composite(sp, b, rng, emb)
        if c is not None:
            res.append((sp["sev"], c, sp["aet"]))
    out["per_budget"][b] = summarise(res)
    print("budget", b, out["per_budget"][b], flush=True)

qual = [sp for sp in spk if sum(len(sp["idx"][c][0]) >= 200 and len(sp["idx"][c][1]) >= 200 for c in CONTRASTS) >= 3]
out["common_set"]["n_qualifying"] = len(qual)
for b in BUDGETS:
    res = []
    for sp in qual:
        emb = np.load(sp["path"], allow_pickle=True)["embeddings"]
        c = composite(sp, b, rng, emb)
        if c is not None:
            res.append((sp["sev"], c, sp["aet"]))
    out["common_set"][b] = summarise(res)
    print("common", b, out["common_set"][b], flush=True)
OUT.write_text(json.dumps(out, indent=1))
print("wrote", OUT)
