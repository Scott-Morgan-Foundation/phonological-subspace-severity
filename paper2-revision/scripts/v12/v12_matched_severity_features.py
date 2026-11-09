#!/usr/bin/env python3
"""v12 (CSL R2, Reviewer 2 point 1): individual phonological features across aetiologies at
matched severity, and whether profile SHAPE differs independently of overall severity.

Input: corrected/track4_master.csv (sha256 d140cb07..., the v9-v15 corrected state).
Speakers: PD, CP, ALS, DS, stroke with a recorded severity label. Stroke has 2 labelled
speakers and severe has only CP in quantity, so the strata analysed are mild and moderate.

A  Matched severity, all corpora: per stratum, per feature (13, as Table 3): Kruskal-Wallis over
   groups with >=5 valid values, epsilon-squared (H-k+1)/(N-k), Holm over the 13 features;
   pairwise Mann-Whitney + pooled-SD Cohen's d, Holm over features within each pair.
B  Matched severity AND corpus/language: SAP English only (PD is absent there), same tests.
C  Profile shape with overall level removed: each speaker's 5 consonant d' z-scored against all
   healthy controls, then centred on the speaker's own mean over the 5 (removes overall
   severity/level). Statistic = sum over strata of n-weighted between-group sum of squares of the
   centred group-mean profiles. Null: aetiology labels permuted within stratum (A) or within
   stratum in SAP only (B); 10,000 permutations, seed 0. Also the uncentred version, and the
   same statistic on the speaker mean level alone for contrast.
"""
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

REV = Path(__file__).resolve().parents[2]
MASTER = REV / "corrected" / "track4_master.csv"
OUT = REV / "results" / "v12" / "v12_matched_severity_features.json"

FEATS_13 = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime",
            "high_dprime", "low_dprime", "back_dprime", "round_dprime",
            "vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv"]
CONS5 = FEATS_13[:5]
AET = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}
STRATA = ["mild", "moderate"]
MIN_N = 5
N_PERM = 10000


def f(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def holm(ps):
    order = np.argsort(ps)
    m, adj, run = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / sp) if sp > 0 else 0.0


def feature_tests(rows):
    out = {}
    kw_p, kw_feats = [], []
    for ft in FEATS_13:
        g = defaultdict(list)
        for r in rows:
            v = f(r[ft])
            if v is not None:
                g[AET[r["aetiology"]]].append(v)
        valid = {a: x for a, x in g.items() if len(x) >= MIN_N}
        rec = {"n": {a: len(x) for a, x in valid.items()}}
        if len(valid) >= 2:
            H, p = stats.kruskal(*valid.values())
            N, k = sum(map(len, valid.values())), len(valid)
            rec.update(H=float(H), p=float(p), eps2=float(max((H - k + 1) / (N - k), 0)))
            kw_p.append(p)
            kw_feats.append(ft)
            rec["pairs"] = {}
            for a, b in combinations(sorted(valid), 2):
                U, pu = stats.mannwhitneyu(valid[a], valid[b], alternative="two-sided")
                rec["pairs"][f"{a}-{b}"] = {"d": cohen_d(valid[a], valid[b]), "p": float(pu)}
        out[ft] = rec
    for ft, pa in zip(kw_feats, holm(kw_p) if kw_p else []):
        out[ft]["p_holm"] = float(pa)
    pair_names = {p for r in out.values() for p in r.get("pairs", {})}
    for pn in pair_names:
        fts = [ft for ft in FEATS_13 if pn in out[ft].get("pairs", {})]
        for ft, pa in zip(fts, holm([out[ft]["pairs"][pn]["p"] for ft in fts])):
            out[ft]["pairs"][pn]["p_holm"] = float(pa)
    summary = {"n_features_tested": len(kw_p),
               "n_sig_holm_05": sum(1 for ft in kw_feats if out[ft]["p_holm"] < 0.05),
               "n_eps2_gt_0.14": sum(1 for ft in kw_feats if out[ft]["eps2"] > 0.14)}
    return {"features": out, "summary": summary}


def profile_test(rows, hc_mu, hc_sd, rng):
    data = []
    for r in rows:
        v = [f(r[c]) for c in CONS5]
        if any(x is None for x in v):
            continue
        z = (np.array(v) - hc_mu) / hc_sd
        data.append((r["severity_label"], AET[r["aetiology"]], z))
    strata = defaultdict(list)
    for s, a, z in data:
        strata[s].append((a, z))
    # keep groups with >=MIN_N within each stratum
    cells = {}
    for s, lst in strata.items():
        cnt = defaultdict(int)
        for a, _ in lst:
            cnt[a] += 1
        keep = [(a, z) for a, z in lst if cnt[a] >= MIN_N]
        if len({a for a, _ in keep}) >= 2:
            cells[s] = (np.array([a for a, _ in keep]), np.stack([z for _, z in keep]))

    def stat(mode, labels_by_stratum):
        tot = 0.0
        for s, (_, Z) in cells.items():
            lab = labels_by_stratum[s]
            if mode == "centred":
                X = Z - Z.mean(1, keepdims=True)
            elif mode == "raw":
                X = Z
            else:  # level only
                X = Z.mean(1, keepdims=True)
            gm = X.mean(0)
            for a in np.unique(lab):
                m = lab == a
                tot += m.sum() * np.sum((X[m].mean(0) - gm) ** 2)
        return tot

    obs_lab = {s: c[0] for s, c in cells.items()}
    res = {"cells": {s: {a: int((c[0] == a).sum()) for a in np.unique(c[0])} for s, c in cells.items()}}
    for mode in ("centred", "raw", "level"):
        obs = stat(mode, obs_lab)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            null[i] = stat(mode, {s: rng.permutation(l) for s, l in obs_lab.items()})
        res[mode] = {"stat": float(obs), "null_mean": float(null.mean()),
                     "p": float((1 + np.sum(null >= obs)) / (N_PERM + 1))}
    # centred group-mean profiles, for the table
    res["centred_profiles"] = {}
    for s, (lab, Z) in cells.items():
        X = Z - Z.mean(1, keepdims=True)
        res["centred_profiles"][s] = {a: [round(float(x), 3) for x in X[lab == a].mean(0)]
                                      for a in np.unique(lab)}
    res["profile_features"] = [c.replace("_dprime", "") for c in CONS5]
    return res


def main():
    rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
    hc = [r for r in rows if r["aetiology"] == "healthy"]
    hcz = np.array([[f(r[c]) for c in CONS5] for r in hc
                    if all(f(r[c]) is not None for c in CONS5)])
    hc_mu, hc_sd = hcz.mean(0), hcz.std(0, ddof=1)
    lab = [r for r in rows if r["aetiology"] in AET and r["severity_label"] in STRATA]
    sap = [r for r in lab if r["dataset"] == "SAP"]
    rng = np.random.RandomState(0)

    out = {"input": str(MASTER.name), "hc_n_for_z": int(len(hcz)),
           "composition": {}, "A_all_corpora": {}, "B_SAP_english": {}}
    for name, subset in (("A_all_corpora", lab), ("B_SAP_english", sap)):
        comp = defaultdict(lambda: defaultdict(int))
        for r in subset:
            comp[f"{r['severity_label']}/{AET[r['aetiology']]}"][f"{r['dataset']}/{r['language']}"] += 1
        out["composition"][name] = {k: dict(v) for k, v in sorted(comp.items())}
        for s in STRATA:
            out[name][s] = feature_tests([r for r in subset if r["severity_label"] == s])
        out[name]["profile_shape"] = profile_test(subset, hc_mu, hc_sd, rng)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    for name in ("A_all_corpora", "B_SAP_english"):
        print(f"\n=== {name} ===")
        for s in STRATA:
            ft = out[name][s]
            print(f" {s}: {ft['summary']}")
            for k, r in ft["features"].items():
                if "eps2" in r:
                    print(f"   {k:20s} n={r['n']} eps2={r['eps2']:.3f} p_holm={r['p_holm']:.2g}")
        ps = out[name]["profile_shape"]
        print(" profile cells:", ps["cells"])
        for m in ("centred", "raw", "level"):
            print(f"  {m}: p={ps[m]['p']:.4f} stat={ps[m]['stat']:.2f} null={ps[m]['null_mean']:.2f}")
        print("  centred profiles", ps["profile_features"], json.dumps(ps["centred_profiles"]))


if __name__ == "__main__":
    sys.exit(main())
