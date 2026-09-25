#!/usr/bin/env python3
"""Figure 2 (pairwise aetiology radar profiles), Paper 2 v5.

Sample: every HuBERT-base speaker in corrected/track4_master.csv with aetiology in the six analysis
groups (HC, PD, CP, ALS, DS, Stroke), all severity levels (not the moderate-only subset). Each feature
is the group mean over speakers with a valid value for that feature, divided by the HC mean (inverted
for pause rate and vowel-duration CV so that outward = closer to healthy).
Drawn at its placed size (\\textwidth = 5.40 in, no bbox_inches='tight'), so native font size = printed
size. One shared legend above the panels; axis labels placed with angle-dependent alignment so that
adjacent labels at the bottom of each radar do not overlap.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

BASE = Path(__file__).resolve().parent.parent
MASTER = BASE / "corrected" / "track4_master.csv"
OUT_PDF = BASE / "figures" / "fig2_radars_v5.pdf"
OUT_PNG = BASE / "figures" / "fig2_radars_v5.png"
OUT_JSON = BASE / "results" / "v5" / "fig2_group_means.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
PLACED_W, H = 5.40, 6.2

FEATS = [("nasal_dprime", "Nasality", 1), ("voicing_dprime", "Voicing", 1),
         ("strident_dprime", "Stridency", 1), ("sonorant_dprime", "Sonorance", 1),
         ("round_dprime", "Rounding", 1), ("back_dprime", "Backness", 1),
         ("low_dprime", "Lowness", 1), ("high_dprime", "Height", 1),
         ("manner_dprime", "Manner", 1), ("vowel_triangle_area", "Vowel triangle", 1),
         ("speech_rate", "Speech rate", 1), ("pause_rate", "Pause rate", -1),
         ("vowel_duration_cv", "Vowel dur. CV", -1)]
AET = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke", "healthy": "HC"}
NAMES = {"PD": "Parkinson's disease", "CP": "Cerebral palsy", "ALS": "ALS", "DS": "Down syndrome",
         "Stroke": "Stroke", "HC": "Healthy controls (= 1.0)"}
COLORS = {"PD": "#7B2FBE", "CP": "#E02020", "ALS": "#1A2B6D", "DS": "#F5820B", "Stroke": "#8B4A2B", "HC": "#5BB8D8"}
NAVY = "#0B1F5C"
PAIRS = [("PD", "CP"), ("ALS", "CP"), ("PD", "ALS"), ("DS", "Stroke"), ("PD", "DS"), ("ALS", "Stroke")]


def f(v):
    try:
        x = float(v)
        return x if x == x else np.nan
    except (TypeError, ValueError):
        return np.nan


groups = {}
for r in csv.DictReader(open(MASTER, encoding="utf-8")):
    g = AET.get(r["aetiology"])
    if g:
        groups.setdefault(g, []).append([f(r[c]) for c, _, _ in FEATS])
arr = {g: np.array(v, float) for g, v in groups.items()}
means = {g: np.nanmean(a, 0) for g, a in arr.items()}
hc = means["HC"]
ratio = {g: np.array([(m[i] / hc[i]) if s == 1 else (hc[i] / m[i]) for i, (_, _, s) in enumerate(FEATS)]) for g, m in means.items()}

OUT_JSON.write_text(json.dumps({
    "script": "scripts/regen_fig2_v5.py", "input": "corrected/track4_master.csv (HuBERT-base)",
    "sample": "all speakers in six analysis groups, all severity levels",
    "n_speakers": {g: int(len(a)) for g, a in arr.items()},
    "n_valid_per_feature": {g: {FEATS[i][0]: int(np.isfinite(a[:, i]).sum()) for i in range(len(FEATS))} for g, a in arr.items()},
    "ratio_to_hc": {g: {FEATS[i][0]: round(float(v[i]), 4) for i in range(len(FEATS))} for g, v in ratio.items()},
}, indent=1), encoding="utf-8")

N = len(FEATS)
ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
ang_c = np.concatenate([ang, ang[:1]])
fig, axes = plt.subplots(3, 2, figsize=(PLACED_W, H), subplot_kw={"polar": True})
fig.subplots_adjust(left=0.16, right=0.84, top=0.885, bottom=0.04, wspace=1.05, hspace=0.62)
RMAX = max(1.25, float(max(v.max() for v in ratio.values())) * 1.05)
for ax, (a, b) in zip(axes.flat, PAIRS):
    one = np.ones(N + 1)
    ax.plot(ang_c, one, color=COLORS["HC"], lw=0.9)
    ax.fill(ang_c, one, color=COLORS["HC"], alpha=0.12)
    for g in (a, b):
        v = np.concatenate([ratio[g], ratio[g][:1]])
        ax.plot(ang_c, v, color=COLORS[g], lw=1.1, marker="o", ms=1.8)
        ax.fill(ang_c, v, color=COLORS[g], alpha=0.15)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    ax.set_ylim(0, RMAX)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["", "1.0"], fontsize=6, color="#555555")
    ax.set_rlabel_position(-360 / N / 2)
    for th, (_, lab, _) in zip(ang, FEATS):
        x = np.sin(th)  # screen x with clockwise-from-top layout
        ha = "center" if abs(x) < 0.15 else ("left" if x > 0 else "right")
        y = np.cos(th)
        va = "bottom" if y > 0.6 else ("top" if y < -0.6 else "center")
        ax.text(th, RMAX * 1.10, lab, fontsize=6.5, ha=ha, va=va, color="#111111")
    ax.set_title(f"{a} vs {b}", fontsize=8, fontweight="bold", color=NAVY, pad=16)
handles = [Line2D([0], [0], color=COLORS[g], lw=2, label=NAMES[g]) for g in ("HC", "PD", "CP", "ALS", "DS", "Stroke")]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print("n speakers:", {g: len(a) for g, a in arr.items()})
print("wrote", OUT_PDF)
