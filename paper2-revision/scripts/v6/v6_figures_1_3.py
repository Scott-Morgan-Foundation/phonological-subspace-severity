"""v6: regenerate Figure 1 (aetiology deviation heatmap) and Figure 3 (HC-normalised PD
profiles by language) from the corrected master, drawn at their placed size so that the
printed text size equals the font size set here (placed at \\textwidth = 5.4 in).

Figure 1: Cohen's d (HC - aetiology; pooled SD) for 13 features x 5 aetiologies; a cell is
left blank (light grey) when the aetiology has fewer than 5 speakers with a valid value.
Figure 3: for each language with >= 3 PD speakers, PD mean / same-language HC mean for the
9 d-prime features (1.0 = healthy); language HC reference requires >= 3 HC speakers,
otherwise the pooled HC mean is used (as in the submitted figure script).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common import MASTER, REV, write

FIG = REV / "figures"
W = 5.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
m = pd.read_csv(MASTER)

FEATURES = [("Nasality", "nasal_dprime", "Consonant"), ("Voicing", "voicing_dprime", "Consonant"),
            ("Sonorance", "sonorant_dprime", "Consonant"), ("Stridency", "strident_dprime", "Consonant"),
            ("Manner", "manner_dprime", "Consonant"), ("Height", "high_dprime", "Vowel"),
            ("Lowness", "low_dprime", "Vowel"), ("Backness", "back_dprime", "Vowel"),
            ("Rounding", "round_dprime", "Vowel"), ("Vowel triangle", "vowel_triangle_area", "Structural"),
            ("Speech rate", "speech_rate", "Prosodic"), ("Pause rate", "pause_rate", "Prosodic"),
            ("Vowel dur. CV", "vowel_duration_cv", "Prosodic")]
AETS = [("PD", "parkinsons"), ("CP", "cerebral_palsy"), ("ALS", "als"), ("DS", "down_syndrome"),
        ("Stroke", "stroke")]
hc = m[m.aetiology == "healthy"]


def cd(a, b):
    a, b = a.dropna(), b.dropna()
    s = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s)


M = np.full((len(FEATURES), len(AETS)), np.nan)
N = np.zeros_like(M, dtype=int)
for i, (_, col, _) in enumerate(FEATURES):
    for j, (_, a) in enumerate(AETS):
        x = m[m.aetiology == a][col]
        N[i, j] = int(x.notna().sum())
        if N[i, j] >= 5:
            M[i, j] = cd(hc[col], x)

fig = plt.figure(figsize=(W, 5.6))
ax = fig.add_axes([0.26, 0.05, 0.56, 0.90])
cmap = plt.get_cmap("YlOrRd").copy()
cmap.set_bad("#d9d9d9")
im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=mcolors.Normalize(0, 2.5), aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isnan(M[i, j]):
            ax.text(j, i, f"n={N[i, j]}", ha="center", va="center", fontsize=6.5, color="#444444")
        else:
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 1.4 else "black")
ax.set_xticks(range(len(AETS)))
ax.set_xticklabels([a for a, _ in AETS], fontsize=7.5, fontweight="bold")
ax.xaxis.tick_top()
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels([f for f, _, _ in FEATURES], fontsize=7)
ax.tick_params(length=0)
prev = FEATURES[0][2]
groups = {}
for k, (_, _, g) in enumerate(FEATURES):
    groups.setdefault(g, []).append(k)
    if g != prev:
        ax.axhline(k - 0.5, color="white", lw=2)
        prev = g
for g, ks in groups.items():   # group labels in their own column, clear of the row labels
    y0, y1 = ks[0] - 0.4, ks[-1] + 0.4
    fig_x = 0.035
    yt = ax.transData.transform([(0, y0), (0, y1)])
    yf = fig.transFigure.inverted().transform(yt)[:, 1]
    fig.add_artist(plt.Line2D([fig_x + 0.012, fig_x + 0.012], [yf[0], yf[1]], color="gray", lw=0.8,
                              transform=fig.transFigure))
    fig.text(fig_x, (yf[0] + yf[1]) / 2, g, rotation=90, ha="center", va="center", fontsize=6.5,
             style="italic", color="gray")
cax = fig.add_axes([0.85, 0.25, 0.025, 0.5])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Cohen's d (HC minus aetiology)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
fig.savefig(FIG / "fig1_heatmap_v6.pdf")
fig.savefig(FIG / "fig1_heatmap_v6.png", dpi=300)
plt.close(fig)

# Figure 3
FE9 = [f for f in FEATURES[:9]]
pdm = m[m.aetiology == "parkinsons"]
langs = [l for l, x in pdm.groupby("language") if len(x) >= 3 and x[[c for _, c, _ in FE9]].mean().notna().sum() >= 5]
allhc = hc[[c for _, c, _ in FE9]].mean()
ratios, ns, refs = {}, {}, {}
for l in langs:
    p = pdm[pdm.language == l][[c for _, c, _ in FE9]].mean()
    h = hc[hc.language == l]
    ref = h[[c for _, c, _ in FE9]].mean() if len(h) >= 3 else allhc
    refs[l] = "language HC" if len(h) >= 3 else "pooled HC"
    ratios[l] = (p / ref).to_dict()
    ns[l] = int(len(pdm[pdm.language == l]))
NAMES = {"en": "English", "es": "Spanish", "sk": "Slovak", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
         "de": "German", "fr": "French", "hu": "Hungarian", "sw": "Swahili", "zh": "Mandarin", "ta": "Tamil"}
COL = {"en": "#E41A1C", "es": "#377EB8", "sk": "#4DAF4A", "it": "#984EA3", "pt": "#FF7F00", "nl": "#8B4513",
       "de": "#A65628", "fr": "#F781BF", "hu": "#999999", "sw": "#FFD700", "zh": "#666666", "ta": "#00CED1"}
fig, ax = plt.subplots(figsize=(W, 2.8))
x = np.arange(9)
w = 0.8 / len(langs)
for i, l in enumerate(langs):
    v = np.array([ratios[l][c] for _, c, _ in FE9], float)
    ok = ~np.isnan(v)
    ax.bar(x[ok] + (i - len(langs) / 2 + 0.5) * w, v[ok], w, color=COL.get(l, "#333"), alpha=0.85,
           edgecolor="white", lw=0.3, label=f"{NAMES.get(l, l)} (n={ns[l]})")
ax.axhline(1.0, color="gray", ls="--", lw=0.8, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([f for f, _, _ in FE9], fontsize=7, rotation=25, ha="right", rotation_mode="anchor")
ax.set_ylabel("PD mean / HC mean", fontsize=7)
ax.tick_params(labelsize=7)
ax.set_ylim(0, 1.35)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
fig.savefig(FIG / "fig3_pd_profiles_v6.pdf")
fig.savefig(FIG / "fig3_pd_profiles_v6.png", dpi=300)
plt.close(fig)

write("v6_figures_1_3.json", {
    "definition": __doc__,
    "fig1": {"features": [f for f, _, _ in FEATURES], "aetiologies": [a for a, _ in AETS],
             "cohens_d": {FEATURES[i][0]: {AETS[j][0]: (None if np.isnan(M[i, j]) else float(M[i, j]))
                                           for j in range(len(AETS))} for i in range(len(FEATURES))},
             "n": {FEATURES[i][0]: {AETS[j][0]: int(N[i, j]) for j in range(len(AETS))} for i in range(len(FEATURES))},
             "hc_n_by_feature": {f: int(hc[c].notna().sum()) for f, c, _ in FEATURES},
             "blank_cells": [f"{FEATURES[i][0]} x {AETS[j][0]} (n={N[i, j]})" for i in range(len(FEATURES))
                             for j in range(len(AETS)) if np.isnan(M[i, j])]},
    "fig3": {"languages": langs, "n_pd": ns, "reference": refs, "ratios": ratios},
}, [MASTER])
print("blank", [f"{FEATURES[i][0]} x {AETS[j][0]} n={N[i, j]}" for i in range(len(FEATURES)) for j in range(len(AETS)) if np.isnan(M[i, j])])
print(langs, ns, refs)
print({FEATURES[i][0]: [round(M[i, j], 2) for j in range(5)] for i in range(5)})
