#!/usr/bin/env python3
"""Figure 4 (severity gradient across 6 SSL backbones), Paper 2 v5. Bernard's decision D3 = option D.

Construction, identical for all six panels:
  - inputs: corrected/ (byte-identical to the 2026-09-15 verification package)
  - unit: one row per unique (dataset, speaker_id) key
  - duplicate keys: keep the LAST row (Bernard D1, 2026-09-25). Affects the 18 SLR65_Tamil keys in the
    five non-HuBERT-base files; the last copy was scored with Tamil directions pooled over SLR65 + SSNCE
    controls, as the Methods describe
  - composite: mean of the five consonant d-primes, >= 3 valid
  - bin: severity_label, except is_control == True -> control
  - excluded from the severity bins: COPAS speakers with non-dysarthric aetiology
    (cleft_palate, laryngectomy, voice_disorder, unknown)
Error bars: 95% percentile bootstrap CI of each group mean (1,000 speaker resamples, seed 42).
Adjacent-severity intervals: percentile 95% CI of the difference in group means, same bootstrap,
saved to results/v5/fig4_adjacent_intervals.json (these are the numbers the caption relies on).
Figure drawn at its placed size (\\textwidth = 5.40 in) so native font size = printed font size.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
CORR = BASE / "corrected"
OUT_PDF = BASE / "figures" / "fig4_severity_v5.pdf"
OUT_PNG = BASE / "figures" / "fig4_severity_v5.png"
OUT_JSON = BASE / "results" / "v5" / "fig4_adjacent_intervals.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
MIN_FEATS = 3
SEV = ["control", "mild", "moderate", "severe"]
SEV_LABELS = ["Control", "Mild", "Moderate", "Severe"]
SEV_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
NONDYS_COPAS = {"cleft_palate", "laryngectomy", "voice_disorder", "unknown"}
DUP_RULE = "keep_last"
SEED, N_BOOT = 42, 1000
PLACED_W = 5.40

BACKBONES = [
    ("HuBERT-base", "track4_master.csv"),
    ("HuBERT-large", "track4_results_hubert-large.csv"),
    ("WavLM", "track4_results_wavlm.csv"),
    ("wav2vec2", "track4_results_wav2vec2.csv"),
    ("XLS-R", "track4_results_xlsr.csv"),
    ("MMS", "track4_results_mms.csv"),
]


def fv(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def composite(r):
    v = [fv(r.get(c)) for c in FEATS]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if len(v) >= MIN_FEATS else None


def load_groups(fn):
    rows = list(csv.DictReader(open(CORR / fn, encoding="utf-8")))
    counts = Counter((r["dataset"], r["speaker_id"]) for r in rows)
    dup_keys = {k for k, c in counts.items() if c > 1}
    last = {}
    for r in rows:
        last[(r["dataset"], r["speaker_id"])] = r
    rows = list(last.values())
    g = defaultdict(list)
    excluded_nondys = 0
    for r in rows:
        s = r["severity_label"]
        if str(r.get("is_control", "")).lower() == "true":
            s = "control"
        if s in ("mild", "moderate", "severe") and r["dataset"] == "COPAS" and r["aetiology"] in NONDYS_COPAS:
            excluded_nondys += 1
            continue
        if s not in SEV:
            continue
        c = composite(r)
        if c is not None:
            g[s].append(c)
    return {s: np.array(g[s]) for s in SEV}, len(dup_keys), excluded_nondys


rng = np.random.default_rng(SEED)
res = {}
for name, fn in BACKBONES:
    g, ndup, nex = load_groups(fn)
    boots = {s: np.array([g[s][rng.integers(0, len(g[s]), len(g[s]))].mean() for _ in range(N_BOOT)]) for s in SEV}
    means = {s: float(g[s].mean()) for s in SEV}
    ci = {s: [float(x) for x in np.percentile(boots[s], [2.5, 97.5])] for s in SEV}
    adj = {}
    for a, b in zip(SEV, SEV[1:]):
        d = boots[a] - boots[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        adj[f"{a}-{b}"] = {"margin": round(means[a] - means[b], 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                           "excludes_zero": bool(lo > 0 or hi < 0)}
    res[name] = {"file": fn, "n": {s: int(len(g[s])) for s in SEV}, "means": {s: round(means[s], 4) for s in SEV},
                 "mean_ci95": {s: [round(x, 4) for x in ci[s]] for s in SEV},
                 "monotonic_point": bool(means["control"] > means["mild"] > means["moderate"] > means["severe"]),
                 "adjacent": adj, "duplicated_keys_resolved_keep_last": ndup, "copas_nondysarthric_rows_excluded": nex}

OUT_JSON.write_text(json.dumps({
    "script": "scripts/regen_fig4_v5.py", "inputs": "corrected/", "seed": SEED, "n_boot": N_BOOT,
    "unit": "unique (dataset, speaker_id) key", "duplicate_rule": DUP_RULE,
    "binning": "severity_label; is_control==True -> control; COPAS non-dysarthric aetiologies excluded from mild/moderate/severe",
    "composite": ">=3 of 5 consonant d-primes", "ci_method": "percentile 95% of bootstrap difference in group means (speaker resampling within group)",
    "backbones": res}, indent=1), encoding="utf-8")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8,
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
fig, axes = plt.subplots(2, 3, figsize=(PLACED_W, 3.9), sharey=True)
ymax = max(max(r["mean_ci95"][s][1] for s in SEV) for r in res.values()) * 1.18
for ax, (name, _) in zip(axes.flat, BACKBONES):
    r = res[name]
    m = [r["means"][s] for s in SEV]
    lo = [r["means"][s] - r["mean_ci95"][s][0] for s in SEV]
    hi = [r["mean_ci95"][s][1] - r["means"][s] for s in SEV]
    ax.bar(range(4), m, color=SEV_COLORS, yerr=[lo, hi], capsize=2,
           error_kw={"linewidth": 0.7, "ecolor": "#333333"}, edgecolor="#333333", linewidth=0.4)
    for i, s in enumerate(SEV):
        ax.text(i, r["mean_ci95"][s][1] + ymax * 0.02, f"n={r['n'][s]}", ha="center", va="bottom", fontsize=6, color="#444444")
    ax.set_xticks(range(4))
    ax.set_xticklabels(SEV_LABELS, rotation=30, ha="right")
    ax.set_title(name, fontweight="bold")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
for i in (0, 3):
    axes.flat[i].set_ylabel("Composite consonant d′")
fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.12, wspace=0.12, hspace=0.55)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
for name, r in res.items():
    mm = r["adjacent"]["mild-moderate"]
    print(f"{name:12s} n={r['n']} mono={r['monotonic_point']} dup_keep_last={r['duplicated_keys_resolved_keep_last']} "
          f"mild-mod={mm['margin']} {mm['ci95']} excl0={mm['excludes_zero']}")
print("wrote", OUT_PDF, OUT_JSON)
