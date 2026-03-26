#!/usr/bin/env python3
"""Generate Track 4 paper figures."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import warnings, os
warnings.filterwarnings("ignore")

df = pd.read_csv("results/track4/track4_results_merged.csv")
FIGS = "results/track4/figures"
os.makedirs(FIGS, exist_ok=True)

SEV_ORDER = ["control", "mild", "moderate", "severe"]
SEV_COLORS = {"control": "#2ecc71", "mild": "#f1c40f", "moderate": "#e67e22", "severe": "#e74c3c"}
FEATURES = ["nasal_dprime", "voicing_dprime", "strident_dprime", "sonorant_dprime", "manner_dprime"]

# Fig 1: Severity degradation slope plot per corpus
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
fig.suptitle("Consonant d-prime Degradation by Severity (per corpus)", fontsize=14, fontweight="bold", y=1.02)
for ax, feat in zip(axes, FEATURES):
    for ds in ["MDSC", "TORGO", "UASPEECH", "Neurovoz", "COPAS", "SAP", "PC-GITA", "YouTube_French"]:
        sub = df[df.dataset == ds]
        means, sev_idx = [], []
        for i, sev in enumerate(SEV_ORDER):
            vals = sub[sub.severity_label == sev][feat].dropna()
            if len(vals) >= 2:
                means.append(vals.mean())
                sev_idx.append(i)
        if len(means) >= 2:
            ax.plot(sev_idx, means, "o-", label=ds, linewidth=1.5, markersize=4, alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Ctrl", "Mild", "Mod", "Sev"], fontsize=9)
    ax.set_title(feat.replace("_dprime", "").title(), fontsize=11)
    ax.set_ylabel("d'")
    ax.grid(True, alpha=0.3)
axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
fig.savefig(f"{FIGS}/fig1_slope_degradation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 1: slope plot saved")

# Fig 2: Per-aetiology radar charts
feat_names = ["nasal", "voicing", "strident", "sonorant", "manner", "high", "low", "back", "round"]
feat_cols = [f + "_dprime" for f in feat_names]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
fig.suptitle("Phonological Fingerprints by Aetiology", fontsize=14, fontweight="bold")
for ax, aet in zip(axes, ["CEREBRAL_PALSY", "PARKINSONS", "ALS"]):
    sub = df[df.aetiology == aet]
    ctrl = df[df.severity_label == "control"]
    angles = np.linspace(0, 2 * np.pi, len(feat_cols), endpoint=False).tolist()
    angles.append(angles[0])
    for sev in SEV_ORDER:
        data = ctrl if sev == "control" else sub[sub.severity_label == sev]
        if len(data) < 2:
            continue
        means = [data[f].dropna().mean() if len(data[f].dropna()) > 0 else 0 for f in feat_cols]
        means.append(means[0])
        color = SEV_COLORS[sev]
        ax.plot(angles, means, "o-", linewidth=2, label=f"{sev} (n={len(data)})", color=color)
        ax.fill(angles, means, alpha=0.1, color=color)
    ax.set_thetagrids(np.degrees(angles[:-1]), feat_names, fontsize=8)
    ax.set_title(aet.replace("_", " ").title(), fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
plt.tight_layout()
fig.savefig(f"{FIGS}/fig2_radar_aetiology.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 2: radar charts saved")

# Fig 3: Within-corpus Spearman rho heatmap
corpora = ["MDSC", "UASPEECH", "TORGO", "Neurovoz", "COPAS", "SAP", "PC-GITA"]
feat_short = ["Nasal", "Voicing", "Strident", "Sonorant", "Manner"]
rho_matrix = np.full((len(corpora), len(FEATURES)), np.nan)
for i, ds in enumerate(corpora):
    sub = df[(df.dataset == ds) & (df.severity_numeric >= 0)]
    for j, feat in enumerate(FEATURES):
        valid = sub[[feat, "severity_numeric"]].dropna()
        if len(valid) >= 5:
            rho, p = stats.spearmanr(valid.severity_numeric, valid[feat])
            rho_matrix[i, j] = rho
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=0, aspect="auto")
ax.set_xticks(range(len(feat_short)))
ax.set_xticklabels(feat_short, fontsize=10)
ax.set_yticks(range(len(corpora)))
ax.set_yticklabels(corpora, fontsize=10)
for i in range(len(corpora)):
    for j in range(len(FEATURES)):
        val = rho_matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(val) > 0.5 else "black")
plt.colorbar(im, ax=ax, label="Spearman rho")
ax.set_title("Within-Corpus Spearman Correlation (Feature vs Severity)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{FIGS}/fig3_heatmap_rho.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 3: heatmap saved")

# Fig 4: Token count vs d-prime bubble chart
fig, ax = plt.subplots(figsize=(10, 7))
lang_colors = {"en": "#e74c3c", "es": "#3498db", "fr": "#2ecc71", "nl": "#f39c12", "zh": "#9b59b6"}
for ds in df.dataset.unique():
    sub = df[(df.dataset == ds) & (df.severity_label == "control")]
    if len(sub) < 2:
        continue
    x = sub.n_phones.median()
    y = sub.nasal_dprime.dropna().mean()
    n = len(sub)
    lang = sub.language.iloc[0]
    ax.scatter(x, y, s=n * 5, c=lang_colors.get(lang, "gray"), alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.annotate(ds, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("Median Phone Tokens per Speaker", fontsize=11)
ax.set_ylabel("Mean Nasality d-prime (HC only)", fontsize=11)
ax.set_title("Token Count vs d-prime (Bubble size = N speakers)", fontsize=13, fontweight="bold")
for lang, color in lang_colors.items():
    ax.scatter([], [], c=color, s=50, label=lang)
ax.legend(title="Language", loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{FIGS}/fig4_bubble_tokens.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 4: bubble chart saved")

# Fig 5: Cross-lingual degradation heatmap (nasality)
languages = ["en", "es", "nl", "zh", "fr"]
lang_names = ["English", "Spanish", "Dutch", "Mandarin", "French"]
heatmap_data = np.full((4, len(languages)), np.nan)
for j, lang in enumerate(languages):
    for i, sev in enumerate(SEV_ORDER):
        vals = df[(df.language == lang) & (df.severity_label == sev)]["nasal_dprime"].dropna()
        if len(vals) > 0:
            heatmap_data[i, j] = vals.mean()
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(lang_names)))
ax.set_xticklabels(lang_names, fontsize=11)
ax.set_yticks(range(4))
ax.set_yticklabels(["Control", "Mild", "Moderate", "Severe"], fontsize=11)
for i in range(4):
    for j in range(len(languages)):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax, label="Nasality d'")
ax.set_title("Cross-Lingual Nasality d-prime by Severity", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{FIGS}/fig5_crosslingual_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 5: cross-lingual heatmap saved")

print(f"\nAll 5 figures saved to {FIGS}/")
