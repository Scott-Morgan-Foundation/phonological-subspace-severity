"""Phase A Q2e-2h: Fig 4 adjacent-severity intervals under alternative binnings (>=3 rule, corrected inputs).

A  current regen_fig4_v4.py: bin = severity_label, every raw row (duplicate keys counted twice)
B  A + one row per (dataset, speaker) key, keep-last
C  B + bin = 'control' when is_control == True (COPAS healthy controls with DIA mild/moderate -> control)
D  C + exclude non-dysarthric COPAS pathologies (cleft_palate, laryngectomy, voice_disorder, unknown) from mild/moderate/severe
Bootstrap: independent speaker resampling within each group, seed 42, 1,000 draws, percentile 95% CI of
mean(group_i) - mean(group_i+1).
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
CORR = BASE / "corrected"
FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
SEV = ["control", "mild", "moderate", "severe"]
BB = {"hubert-base": "track4_master.csv", "hubert-large": "track4_results_hubert-large.csv",
      "wavlm": "track4_results_wavlm.csv", "wav2vec2": "track4_results_wav2vec2.csv",
      "xlsr": "track4_results_xlsr.csv", "mms": "track4_results_mms.csv"}
NONDYS = {"cleft_palate", "laryngectomy", "voice_disorder", "unknown"}
SEED, NB = 42, 1000


def f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def comp(r):
    v = [f(r[c]) for c in FEATS]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if len(v) >= 3 else None


def groups(rows, opt):
    if opt != "A":
        d = {}
        for r in rows:
            d[(r["dataset"], r["speaker_id"])] = r
        rows = list(d.values())
    g = defaultdict(list)
    for r in rows:
        s = r["severity_label"]
        if opt in ("C", "D") and r["is_control"].lower() == "true":
            s = "control"
        if opt == "D" and s in ("mild", "moderate", "severe") and r["dataset"] == "COPAS" and r["aetiology"] in NONDYS:
            continue
        if s not in SEV:
            continue
        c = comp(r)
        if c is not None:
            g[s].append(c)
    return {s: np.array(g[s]) for s in SEV}


out = {"seed": SEED, "n_boot": NB, "rule": ">=3 of 5 consonant features", "options": __doc__}
for opt in "ABCD":
    rng = np.random.default_rng(SEED)
    res = {}
    for b, fn in BB.items():
        rows = list(csv.DictReader(open(CORR / fn, encoding="utf-8")))
        g = groups(rows, opt)
        means = {s: round(float(g[s].mean()), 4) for s in SEV}
        margins = {}
        for s1, s2 in zip(SEV, SEV[1:]):
            a, c = g[s1], g[s2]
            dr = np.array([a[rng.integers(0, len(a), len(a))].mean() - c[rng.integers(0, len(c), len(c))].mean() for _ in range(NB)])
            lo, hi = np.percentile(dr, [2.5, 97.5])
            margins[f"{s1}-{s2}"] = {"margin": round(float(a.mean() - c.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "excludes_zero": bool(lo > 0 or hi < 0)}
        res[b] = {"n": {s: int(len(g[s])) for s in SEV}, "means": means,
                  "monotonic_point": bool(means["control"] > means["mild"] > means["moderate"] > means["severe"]),
                  "adjacent": margins}
    out[f"option_{opt}"] = res
(Path(__file__).parent / "fig4_adjacent_intervals.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for opt in "ABCD":
    print("== option", opt)
    for b, r in out[f"option_{opt}"].items():
        mm = r["adjacent"]["mild-moderate"]
        print(f"  {b:13s} n={r['n']} mono={r['monotonic_point']} mild-mod={mm['margin']} {mm['ci95']} excl0={mm['excludes_zero']}  ctrl-mild={r['adjacent']['control-mild']['margin']} mod-sev={r['adjacent']['moderate-severe']['margin']} {r['adjacent']['moderate-severe']['ci95']}")
