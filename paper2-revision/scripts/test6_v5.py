#!/usr/bin/env python
"""Test 6 (v5): cross-backbone stability on the corrected inputs.

Rules (Bernard, 2026-09-25):
- inputs: corrected/ (byte-identical to the 2026-09-15 corrected package)
- composite = mean of the 5 consonant d' over available features, only when >= 3 are valid
- duplicated (dataset, speaker_id) keys: keep the LAST row. In the five non-HuBERT-base
  files the 18 SLR65_Tamil speakers appear twice; the last copy was scored with Tamil
  directions pooled over SLR65 + SSNCE controls, as the Methods describe (see
  v5_investigation Tamil provenance report). The HuBERT-base master has no duplicates.
Outputs: results/v5/test6_v5.json
"""
import csv
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CORR = BASE / "corrected"
OUT = BASE / "results" / "v5" / "test6_v5.json"

SEED = 42
N_BOOT = 1_000
MIN_FEATURES = 3
FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime",
         "manner_dprime"]
AETS = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
        "down_syndrome": "DS", "stroke": "Stroke"}
SEV = ["control", "mild", "moderate", "severe"]
BB = {"hubert-base": "track4_master.csv",
      "hubert-large": "track4_results_hubert-large.csv",
      "wavlm": "track4_results_wavlm.csv",
      "wav2vec2": "track4_results_wav2vec2.csv",
      "xlsr": "track4_results_xlsr.csv",
      "mms": "track4_results_mms.csv"}


def fnum(v):
    try:
        x = float(v)
        return x if x == x else np.nan
    except (TypeError, ValueError):
        return np.nan


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(fn, excl_nl=False):
    rows, keys = {}, []
    for r in csv.DictReader(open(CORR / fn, encoding="utf-8")):
        if excl_nl and r["language"] == "nl":
            continue
        key = (r["dataset"], r["speaker_id"])
        keys.append(key)
        sev = "control" if r["is_control"].lower() == "true" else r["severity_label"]
        rows[key] = {"vec": np.array([fnum(r[c]) for c in FEATS]),
                     "aet": "HC" if r["aetiology"] == "healthy" else AETS.get(r["aetiology"]),
                     "hc": r["aetiology"] == "healthy",
                     "sev": sev if sev in SEV else None}
    dups = {f"{k[0]}|{k[1]}": c for k, c in Counter(keys).items() if c > 1}
    return rows, dups


def comp(v):
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) >= MIN_FEATURES else np.nan


def cos(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def pair_spearman(data):
    comps = {b: {k: comp(v["vec"]) for k, v in d.items()} for b, d in data.items()}
    res = {}
    for b1, b2 in combinations(data, 2):
        common = sorted(set(comps[b1]) & set(comps[b2]))
        p = [(comps[b1][k], comps[b2][k]) for k in common]
        p = [(x, y) for x, y in p if x == x and y == y]
        res[f"{b1}|{b2}"] = {"rho": float(spearmanr([a for a, _ in p], [b for _, b in p]).statistic),
                             "n": len(p)}
    return res


def profile_cos(data, rng):
    out = {}
    for aet in ["PD", "CP", "ALS", "DS", "Stroke", "HC"]:
        pairs = {}
        for b1, b2 in combinations(data, 2):
            s1 = {k: v["vec"] for k, v in data[b1].items() if v["aet"] == aet}
            s2 = {k: v["vec"] for k, v in data[b2].items() if v["aet"] == aet}
            common = sorted(set(s1) & set(s2))
            if len(common) < 5:
                continue
            a1 = np.array([s1[k] for k in common])
            a2 = np.array([s2[k] for k in common])
            obs = cos(np.nanmean(a1, 0), np.nanmean(a2, 0))
            draws = np.empty(N_BOOT)
            for i in range(N_BOOT):
                idx = rng.integers(0, len(common), len(common))
                draws[i] = cos(np.nanmean(a1[idx], 0), np.nanmean(a2[idx], 0))
            lo, hi = np.percentile(draws[~np.isnan(draws)], [2.5, 97.5])
            pairs[f"{b1}|{b2}"] = {"cos": obs, "ci95": [float(lo), float(hi)], "n_spk": len(common)}
        out[aet] = pairs
    return out


def kendall_w(data):
    def w(rm):
        m, n = rm.shape
        R = rm.sum(axis=0)
        return float(12 * ((R - R.mean()) ** 2).sum() / (m * m * (n ** 3 - n)))
    kw = {}
    for aet in ["PD", "CP", "ALS", "DS", "Stroke"]:
        ranks = []
        for b, d in data.items():
            hc = np.nanmean([v["vec"] for v in d.values() if v["hc"]], 0)
            dy = [v["vec"] for v in d.values() if v["aet"] == aet]
            if len(dy) < 5:
                continue
            ratio = np.nanmean(dy, 0) / np.where(hc == 0, 1, hc)
            ranks.append(np.argsort(np.argsort(ratio)) + 1)
        if len(ranks) == len(data):
            kw[aet] = w(np.array(ranks))
    return kw


def gradient(data, rng):
    out = {}
    for b, d in data.items():
        g = {s: np.array([x for x in (comp(v["vec"]) for v in d.values() if v["sev"] == s) if x == x])
             for s in SEV}
        means = {s: float(g[s].mean()) for s in SEV}
        margins = {}
        for s1, s2 in zip(SEV, SEV[1:]):
            draws = np.empty(N_BOOT)
            for i in range(N_BOOT):
                draws[i] = g[s1][rng.integers(0, len(g[s1]), len(g[s1]))].mean() - \
                           g[s2][rng.integers(0, len(g[s2]), len(g[s2]))].mean()
            lo, hi = np.percentile(draws, [2.5, 97.5])
            margins[f"{s1}-{s2}"] = {"margin": means[s1] - means[s2], "ci95": [float(lo), float(hi)]}
        out[b] = {"means": means, "n": {s: int(len(g[s])) for s in SEV},
                  "monotonic_point": means["control"] > means["mild"] > means["moderate"] > means["severe"],
                  "adjacent_margins": margins}
    return out


def main():
    rng = np.random.default_rng(SEED)
    loaded = {b: load(fn) for b, fn in BB.items()}
    data = {b: x[0] for b, x in loaded.items()}
    sp = pair_spearman(data)
    rhos = {k: v["rho"] for k, v in sp.items()}
    ns = [v["n"] for v in sp.values()]

    nl = {b: load(fn, excl_nl=True)[0] for b, fn in BB.items()}
    sp_nl = pair_spearman(nl)
    rhos_nl = {k: v["rho"] for k, v in sp_nl.items()}

    def pc_obs_min(dat):
        res = {}
        for aet in ["PD", "CP", "ALS", "HC"]:
            vals = []
            for b1, b2 in combinations(dat, 2):
                s1 = {k: v["vec"] for k, v in dat[b1].items() if v["aet"] == aet}
                s2 = {k: v["vec"] for k, v in dat[b2].items() if v["aet"] == aet}
                common = sorted(set(s1) & set(s2))
                if len(common) >= 5:
                    vals.append(cos(np.nanmean([s1[k] for k in common], 0), np.nanmean([s2[k] for k in common], 0)))
            res[aet] = round(min(vals), 4)
        return res
    pcmin_with, pcmin_without = pc_obs_min(data), pc_obs_min(nl)
    pc = profile_cos(data, rng)
    pc_min = {a: min(v["cos"] for v in p.values()) for a, p in pc.items() if p}
    kw = kendall_w(data)
    grad = gradient(data, rng)
    sevnum = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
    sev_rho = {}
    for b, d in data.items():
        xy = [(comp(v["vec"]), sevnum[v["sev"]]) for v in d.values() if v["aet"] is not None and v["sev"]]
        xy = [(x, y) for x, y in xy if x == x]
        r = spearmanr([x for x, _ in xy], [y for _, y in xy])
        sev_rho[b] = {"rho": round(float(r.statistic), 4), "p": float(r.pvalue), "n": len(xy)}

    out = {
        "rules": {"inputs": "corrected/", "min_features": MIN_FEATURES, "duplicate_keys": "keep last",
                  "seed": SEED, "n_boot": N_BOOT},
        "input_sha256": {fn: sha(CORR / fn) for fn in BB.values()},
        "duplicate_keys_per_file": {b: x[1] for b, x in loaded.items()},
        "per_pair": {k: {"rho": round(v["rho"], 4), "n": v["n"]} for k, v in sp.items()},
        "rho_min": round(min(rhos.values()), 4), "min_pair": min(rhos, key=rhos.get),
        "rho_max": round(max(rhos.values()), 4), "max_pair": max(rhos, key=rhos.get),
        "n_range": [min(ns), max(ns)],
        "dutch_exclusion": {"rho_min_with_nl": round(min(rhos.values()), 4),
                            "rho_min_without_nl": round(min(rhos_nl.values()), 4),
                            "min_pair_without_nl": min(rhos_nl, key=rhos_nl.get),
                            "per_pair_without_nl": {k: round(v, 4) for k, v in rhos_nl.items()},
                            "profile_cos_min_with_nl": pcmin_with,
                            "profile_cos_min_without_nl": pcmin_without,
                            "n_range_without_nl": [min(v["n"] for v in sp_nl.values()),
                                                   max(v["n"] for v in sp_nl.values())]},
        "profile_cosine_min_per_aetiology": {a: round(v, 4) for a, v in pc_min.items()},
        "profile_cosine_min_overall": round(min(pc_min.values()), 4),
        "profile_cosines": {a: {k: {"cos": round(v["cos"], 4), "ci95": [round(x, 4) for x in v["ci95"]],
                                    "n_spk": v["n_spk"]} for k, v in p.items()} for a, p in pc.items()},
        "severity_rho_per_backbone": sev_rho,
        "kendalls_w": {a: round(v, 4) for a, v in kw.items()},
        "kendalls_w_range": [round(min(kw.values()), 4), round(max(kw.values()), 4)],
        "severity_gradient": {b: {"means": {s: round(m, 4) for s, m in g["means"].items()}, "n": g["n"],
                                  "monotonic_point": g["monotonic_point"],
                                  "adjacent_margins": {k: {"margin": round(v["margin"], 4),
                                                           "ci95": [round(x, 4) for x in v["ci95"]]}
                                                       for k, v in g["adjacent_margins"].items()}}
                              for b, g in grad.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("rho_min", "min_pair", "rho_max", "max_pair", "n_range",
                                          "dutch_exclusion", "profile_cosine_min_per_aetiology",
                                          "kendalls_w", "duplicate_keys_per_file")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
