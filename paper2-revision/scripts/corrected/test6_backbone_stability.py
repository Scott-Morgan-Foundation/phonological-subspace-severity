#!/usr/bin/env python
"""Test 6 — cross-backbone stability WITH uncertainty (ANALYSIS_PLAN §7).

Six backbones: hubert-base (master) + hubert-large, mms, wav2vec2, wavlm, xlsr.
1. Severity gradient per backbone (control/mild/moderate/severe composite means)
   with speaker-bootstrap 95% CIs on each ADJACENT margin + monotonicity verdict.
2. Aetiology 5-feature profiles: pairwise backbone cosines per aetiology with
   speaker-bootstrap CIs (checks the submitted ">0.96 across all backbone pairs").
3. Kendall's W across backbones on the feature-collapse ordering per aetiology
   (HC-normalised ratios).
4. Per-speaker composite Spearman between backbone pairs on common speakers
   (checks the submitted "rho > 0.77").
Seed 42, 1,000 bootstrap draws. Reads frozen inputs only.
Outputs: results/test6_backbones.json + results/test6_report.md
"""
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
RES = BASE / "corrected"
OUT_JSON = BASE / "results" / "test6_backbones_corrected.json"
OUT_MD = BASE / "results" / "test6_report_corrected.md"

SEED = 42
N_BOOT = 1_000
FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime",
         "manner_dprime"]
AETS = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
        "down_syndrome": "DS", "stroke": "Stroke"}
SEV = ["control", "mild", "moderate", "severe"]
BACKBONES = {"hubert-base": "track4_master.csv",
             "hubert-large": "track4_results_hubert-large.csv",
             "wavlm": "track4_results_wavlm.csv",
             "wav2vec2": "track4_results_wav2vec2.csv",
             "xlsr": "track4_results_xlsr.csv",
             "mms": "track4_results_mms.csv"}


def fnum(v):
    try:
        f = float(v)
        return f if f == f else np.nan
    except (TypeError, ValueError):
        return np.nan


def load(fname):
    rows = {}
    with open(RES / fname, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["speaker_id"])
            vec = np.array([fnum(r[c]) for c in FEATS])
            sev = "control" if r["is_control"].lower() == "true" else r["severity_label"]
            rows[key] = {"vec": vec, "aet": ("HC" if r["aetiology"] == "healthy"
                                             else AETS.get(r["aetiology"])),
                         "sev": sev if sev in SEV else None,
                         "hc": r["aetiology"] == "healthy"}
    return rows


def comp(vec):
    v = vec[~np.isnan(vec)]
    return float(v.mean()) if len(v) else np.nan


def cos(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    rng = np.random.default_rng(SEED)
    data = {b: load(f) for b, f in BACKBONES.items()}
    names = list(BACKBONES)
    out = {"seed": SEED, "n_boot": N_BOOT}

    # 1. severity gradient + adjacent-margin CIs
    grad = {}
    for b in names:
        groups = {s: [comp(v["vec"]) for v in data[b].values() if v["sev"] == s] for s in SEV}
        groups = {s: np.array([x for x in g if x == x]) for s, g in groups.items()}
        means = {s: float(g.mean()) for s, g in groups.items()}
        margins = {}
        for s1, s2 in zip(SEV, SEV[1:]):
            g1, g2 = groups[s1], groups[s2]
            d_obs = g1.mean() - g2.mean()
            draws = np.empty(N_BOOT)
            for i in range(N_BOOT):
                draws[i] = g1[rng.integers(0, len(g1), len(g1))].mean() - \
                           g2[rng.integers(0, len(g2), len(g2))].mean()
            lo, hi = np.percentile(draws, [2.5, 97.5])
            margins[f"{s1}-{s2}"] = {"margin": round(float(d_obs), 4),
                                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                     "resolved": bool(lo > 0)}
        grad[b] = {"means": {s: round(m, 4) for s, m in means.items()},
                   "n": {s: int(len(groups[s])) for s in SEV},
                   "monotonic_point": bool(means["control"] > means["mild"] >
                                           means["moderate"] > means["severe"]),
                   "adjacent_margins": margins}
    out["severity_gradient"] = grad

    # 2. aetiology profile cosines across backbone pairs, with bootstrap CIs
    prof_cos = {}
    for aet in ["PD", "CP", "ALS", "DS", "Stroke", "HC"]:
        pairs = {}
        for b1, b2 in combinations(names, 2):
            s1 = {k: v["vec"] for k, v in data[b1].items() if v["aet"] == aet}
            s2 = {k: v["vec"] for k, v in data[b2].items() if v["aet"] == aet}
            common = sorted(set(s1) & set(s2))
            if len(common) < 5:
                continue
            m1 = np.nanmean([s1[k] for k in common], 0)
            m2 = np.nanmean([s2[k] for k in common], 0)
            obs = cos(m1, m2)
            draws = np.empty(N_BOOT)
            arr1 = np.array([s1[k] for k in common])
            arr2 = np.array([s2[k] for k in common])
            for i in range(N_BOOT):
                idx = rng.integers(0, len(common), len(common))
                draws[i] = cos(np.nanmean(arr1[idx], 0), np.nanmean(arr2[idx], 0))
            ok = draws[~np.isnan(draws)]
            lo, hi = np.percentile(ok, [2.5, 97.5])
            pairs[f"{b1}|{b2}"] = {"cos": round(obs, 4),
                                   "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                   "n_spk": len(common)}
        prof_cos[aet] = pairs
    out["profile_cosines"] = prof_cos

    # 3. Kendall's W on feature-collapse ordering (HC-normalised ratios) per aetiology
    def kendall_w(rank_matrix):
        m, n = rank_matrix.shape          # m judges (backbones), n items (features)
        R = rank_matrix.sum(axis=0)
        S = ((R - R.mean()) ** 2).sum()
        return float(12 * S / (m * m * (n ** 3 - n)))
    kw = {}
    for aet in ["PD", "CP", "ALS", "DS", "Stroke"]:
        ranks = []
        for b in names:
            hc = np.nanmean([v["vec"] for v in data[b].values() if v["hc"]], 0)
            dy = [v["vec"] for v in data[b].values() if v["aet"] == aet]
            if len(dy) < 5:
                continue
            ratio = np.nanmean(dy, 0) / np.where(hc == 0, 1, hc)
            ranks.append(np.argsort(np.argsort(ratio)) + 1)
        if len(ranks) == len(names):
            kw[aet] = round(kendall_w(np.array(ranks)), 4)
    out["kendalls_w_feature_ordering"] = kw

    # 4. per-speaker composite Spearman between backbone pairs
    sp = {}
    for b1, b2 in combinations(names, 2):
        common = sorted(set(data[b1]) & set(data[b2]))
        a = [comp(data[b1][k]["vec"]) for k in common]
        c = [comp(data[b2][k]["vec"]) for k in common]
        pair = [(x, y) for x, y in zip(a, c) if x == x and y == y]
        sp[f"{b1}|{b2}"] = round(float(spearmanr([p[0] for p in pair],
                                                 [p[1] for p in pair]).statistic), 4)
    out["speaker_composite_spearman"] = sp

    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")

    lines = ["# Test 6 — Cross-backbone stability with uncertainty", "",
             f"Seed {SEED}, {N_BOOT} bootstrap draws. Full numbers: `test6_backbones.json`.", ""]
    lines.append("## Severity gradient (composite of 5 consonant d')")
    lines.append("| Backbone | control | mild | moderate | severe | monotonic | unresolved adjacent margins |")
    lines.append("|---|---|---|---|---|---|---|")
    for b in names:
        g = grad[b]
        unres = [k for k, v in g["adjacent_margins"].items() if not v["resolved"]]
        lines.append(f"| {b} | {g['means']['control']} | {g['means']['mild']} | "
                     f"{g['means']['moderate']} | {g['means']['severe']} | "
                     f"{'yes' if g['monotonic_point'] else 'NO'} | "
                     f"{', '.join(unres) if unres else '—'} |")
    lines.append("")
    lines.append("## Aetiology profile cosines across backbone pairs (min per aetiology)")
    for aet, pairs in prof_cos.items():
        if not pairs:
            continue
        mn = min(pairs.items(), key=lambda kv: kv[1]["cos"])
        below = [f"{k} {v['cos']}" for k, v in pairs.items() if v["cos"] < 0.96]
        lines.append(f"- {aet}: min {mn[1]['cos']} CI {mn[1]['ci95']} ({mn[0]}); "
                     f"pairs < 0.96: {', '.join(below) if below else 'none'}")
    lines.append("")
    lines.append(f"## Kendall's W on feature-collapse ordering: {kw}")
    lines.append("")
    mn = min(sp.items(), key=lambda kv: kv[1])
    lines.append(f"## Per-speaker composite Spearman: min {mn[1]} ({mn[0]}); "
                 f"all pairs: {sp}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
