#!/usr/bin/env python3
"""
Publication-quality figures for aetiology discrimination analysis.

Figures:
  1. Radar plot: aetiology profiles across 13 features
  2. Heatmap: pairwise Cohen's d between aetiologies (consonant d-primes)
  3. Bar chart: eta2 effect sizes per feature
  4. Box plots: top 4 features by aetiology
  5. Heatmap: HC vs each aetiology, all 13 features
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from scipy import stats

BASE = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
MASTER = BASE / "results" / "track4" / "track4_master.csv"
OUTDIR = BASE / "results" / "track4" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# SMF brand colors
SMF_ORANGE = "#F04E23"
SMF_NAVY = "#0B1F5C"
SMF_BLUE = "#4DB8C8"
SMF_WHITE = "#FFFFFF"

# Aetiology colors — distinct, colorblind-friendly
AET_COLORS = {
    "healthy": "#4DAF4A",       # green
    "parkinsons": "#377EB8",    # blue
    "cerebral_palsy": "#E41A1C",# red
    "als": "#FF7F00",           # orange
    "down_syndrome": "#984EA3", # purple
    "stroke": "#A65628",        # brown
}

AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}

AETIOLOGIES = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
DYS_AETS = ["parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]

FEATS_13 = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime",
    "high_dprime", "low_dprime", "back_dprime", "round_dprime",
    "vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv",
]

FEAT_LABELS = [
    "Nasality", "Voicing", "Sonorance", "Stridency", "Manner",
    "Height", "Lowness", "Backness", "Rounding",
    "Vowel Triangle", "Speech Rate", "Pause Rate", "Vowel Dur. CV",
]


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except (ValueError, TypeError):
        return None


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return (mx - my) / sp if sp > 0 else 0.0


def load_data():
    with open(MASTER, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_aet = defaultdict(list)
    for r in rows:
        aet = r.get("aetiology", "unknown")
        if aet in AETIOLOGIES:
            by_aet[aet].append(r)

    # Extract feature values per aetiology
    feat_data = {}
    for feat in FEATS_13:
        groups = {}
        for aet in AETIOLOGIES:
            vals = [safe_float(r.get(feat)) for r in by_aet[aet]]
            vals = [v for v in vals if v is not None]
            groups[aet] = np.array(vals)
        feat_data[feat] = groups

    return by_aet, feat_data


def fig1_radar(feat_data):
    """Radar plot of aetiology profiles (z-scored)."""
    # Use 9 d-prime features (not prosodic/VTA for radar — different scales)
    radar_feats = FEATS_13[:9]
    radar_labels = FEAT_LABELS[:9]
    n_feats = len(radar_feats)

    # Compute means and z-score across aetiologies
    means = {}
    for aet in AETIOLOGIES:
        m = []
        for feat in radar_feats:
            vals = feat_data[feat].get(aet, np.array([]))
            m.append(np.mean(vals) if len(vals) > 0 else 0)
        means[aet] = np.array(m)

    # Z-score relative to HC
    hc_mean = means["healthy"]
    hc_std = np.array([np.std(feat_data[f]["healthy"]) for f in radar_feats])
    hc_std[hc_std == 0] = 1

    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for aet in AETIOLOGIES:
        z_vals = (means[aet] - hc_mean) / hc_std
        values = z_vals.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, markersize=6,
                color=AET_COLORS[aet], label=f"{AET_SHORT[aet]} (n={len(feat_data[radar_feats[0]][aet])})")
        ax.fill(angles, values, alpha=0.05, color=AET_COLORS[aet])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=12, fontweight='bold')
    ax.set_title("Aetiology Phonological Profiles\n(z-scored relative to HC mean/SD)",
                 fontsize=16, fontweight='bold', pad=30, color=SMF_NAVY)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11, framealpha=0.9)

    # Add reference line at 0 (HC level)
    ref = [0] * (n_feats + 1)
    ax.plot(angles, ref, '--', linewidth=1.5, color='gray', alpha=0.5, label='_nolegend_')

    ax.set_ylim(-3.5, 1)
    ax.set_yticks([-3, -2, -1, 0])
    ax.set_yticklabels(['-3 SD', '-2 SD', '-1 SD', 'HC'], fontsize=9)

    plt.tight_layout()
    path = OUTDIR / "aetiology_radar_13feat.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig2_pairwise_heatmap(feat_data):
    """Heatmap of pairwise Cohen's d (composite consonant d-prime)."""
    consonant_feats = FEATS_13[:5]

    # Composite per speaker per aetiology
    comp_by_aet = {}
    for aet in AETIOLOGIES:
        scores = []
        n = len(feat_data[consonant_feats[0]][aet])
        for i in range(n):
            vals = []
            for feat in consonant_feats:
                arr = feat_data[feat][aet]
                if i < len(arr):
                    vals.append(arr[i])
            if len(vals) >= 3:
                scores.append(np.mean(vals))
        comp_by_aet[aet] = np.array(scores)

    n_aet = len(AETIOLOGIES)
    d_matrix = np.zeros((n_aet, n_aet))
    labels = [AET_SHORT[a] for a in AETIOLOGIES]

    for i, a1 in enumerate(AETIOLOGIES):
        for j, a2 in enumerate(AETIOLOGIES):
            if i == j:
                d_matrix[i, j] = 0
            else:
                d_matrix[i, j] = cohens_d(comp_by_aet[a1], comp_by_aet[a2])

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(d_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(n_aet))
    ax.set_yticks(range(n_aet))
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=13, fontweight='bold')

    # Annotate cells
    for i in range(n_aet):
        for j in range(n_aet):
            val = d_matrix[i, j]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax.text(j, i, f"{val:+.2f}", ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d (row - column)", fontsize=12)

    ax.set_title("Pairwise Cohen's d\n(Composite Consonant d-prime)",
                 fontsize=16, fontweight='bold', color=SMF_NAVY)
    ax.set_xlabel("", fontsize=1)

    plt.tight_layout()
    path = OUTDIR / "aetiology_pairwise_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig3_eta2_bar(feat_data):
    """Horizontal bar chart of eta2 per feature."""
    eta2s = []
    for feat in FEATS_13:
        group_lists = [feat_data[feat][a] for a in AETIOLOGIES if len(feat_data[feat][a]) >= 5]
        if len(group_lists) < 2:
            eta2s.append(0)
            continue
        H, p = stats.kruskal(*group_lists)
        N = sum(len(g) for g in group_lists)
        k = len(group_lists)
        eta_sq = max((H - k + 1) / (N - k), 0) if N > k else 0
        eta2s.append(eta_sq)

    # Sort by eta2
    sorted_idx = np.argsort(eta2s)
    sorted_labels = [FEAT_LABELS[i] for i in sorted_idx]
    sorted_vals = [eta2s[i] for i in sorted_idx]

    # Color by effect size
    colors = []
    for v in sorted_vals:
        if v >= 0.14:
            colors.append(SMF_ORANGE)
        elif v >= 0.06:
            colors.append(SMF_BLUE)
        else:
            colors.append('#AAAAAA')

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(sorted_labels)), sorted_vals, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=12)
    ax.set_xlabel("eta-squared (effect size)", fontsize=13, fontweight='bold')
    ax.set_title("Aetiology Discrimination Power per Feature\n(Kruskal-Wallis eta-squared, 6 groups, n=2,998)",
                 fontsize=15, fontweight='bold', color=SMF_NAVY)

    # Reference lines
    ax.axvline(x=0.06, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.14, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.065, len(sorted_labels) - 0.5, 'medium', fontsize=9, color='gray')
    ax.text(0.145, len(sorted_labels) - 0.5, 'large', fontsize=9, color='gray')

    # Value labels
    for bar, val in zip(bars, sorted_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    # Legend
    patches = [
        mpatches.Patch(color=SMF_ORANGE, label='Large (>0.14)'),
        mpatches.Patch(color=SMF_BLUE, label='Medium (0.06-0.14)'),
        mpatches.Patch(color='#AAAAAA', label='Small (<0.06)'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=11)

    ax.set_xlim(0, max(sorted_vals) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTDIR / "aetiology_eta2_bar.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig4_boxplots(feat_data):
    """Box plots of top 4 features by aetiology."""
    top_feats = ["high_dprime", "round_dprime", "strident_dprime", "nasal_dprime"]
    top_labels = ["Height d'", "Rounding d'", "Stridency d'", "Nasality d'"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (feat, label) in enumerate(zip(top_feats, top_labels)):
        ax = axes[idx]
        data = []
        positions = []
        tick_labels = []
        colors_list = []

        for i, aet in enumerate(AETIOLOGIES):
            vals = feat_data[feat].get(aet, np.array([]))
            if len(vals) > 0:
                data.append(vals)
                positions.append(i)
                tick_labels.append(f"{AET_SHORT[aet]}\n(n={len(vals)})")
                colors_list.append(AET_COLORS[aet])

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=10, fontweight='bold')
        ax.set_ylabel("d-prime", fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold', color=SMF_NAVY)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Phonological Contrast Preservation by Aetiology",
                 fontsize=17, fontweight='bold', color=SMF_NAVY, y=1.01)
    plt.tight_layout()
    path = OUTDIR / "aetiology_boxplots_top4.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig5_hc_vs_aet_heatmap(feat_data):
    """Heatmap of Cohen's d: HC vs each dysarthric aetiology, all 13 features."""
    d_matrix = np.zeros((len(FEATS_13), len(DYS_AETS)))

    for i, feat in enumerate(FEATS_13):
        hc = feat_data[feat].get("healthy", np.array([]))
        for j, aet in enumerate(DYS_AETS):
            vals = feat_data[feat].get(aet, np.array([]))
            if len(hc) >= 5 and len(vals) >= 5:
                d_matrix[i, j] = cohens_d(hc, vals)
            else:
                d_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(d_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2.5)

    ax.set_xticks(range(len(DYS_AETS)))
    ax.set_yticks(range(len(FEATS_13)))
    ax.set_xticklabels([AET_SHORT[a] for a in DYS_AETS], fontsize=13, fontweight='bold')
    ax.set_yticklabels(FEAT_LABELS, fontsize=11)

    for i in range(len(FEATS_13)):
        for j in range(len(DYS_AETS)):
            val = d_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 1.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d (HC - aetiology)", fontsize=12)

    # Separator lines between feature groups
    ax.axhline(y=4.5, color='white', linewidth=2)  # consonant/vowel boundary
    ax.axhline(y=8.5, color='white', linewidth=2)  # vowel/structural boundary
    ax.axhline(y=9.5, color='white', linewidth=2)  # structural/prosodic boundary

    # Group labels
    ax.text(-0.8, 2, 'Consonant', rotation=90, va='center', fontsize=10, fontstyle='italic', color='gray')
    ax.text(-0.8, 6.5, 'Vowel', rotation=90, va='center', fontsize=10, fontstyle='italic', color='gray')
    ax.text(-0.8, 9, 'Struct.', rotation=90, va='center', fontsize=10, fontstyle='italic', color='gray')
    ax.text(-0.8, 11, 'Prosodic', rotation=90, va='center', fontsize=10, fontstyle='italic', color='gray')

    ax.set_title("Deviation from Healthy Controls by Aetiology\n(Cohen's d, all 13 features)",
                 fontsize=15, fontweight='bold', color=SMF_NAVY)

    plt.tight_layout()
    path = OUTDIR / "aetiology_hc_deviation_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 80)
    print("Generating aetiology discrimination figures")
    print("=" * 80)

    by_aet, feat_data = load_data()
    print(f"Loaded {sum(len(v) for v in by_aet.values())} speakers across {len(by_aet)} aetiologies\n")

    fig1_radar(feat_data)
    fig2_pairwise_heatmap(feat_data)
    fig3_eta2_bar(feat_data)
    fig4_boxplots(feat_data)
    fig5_hc_vs_aet_heatmap(feat_data)

    print(f"\nAll figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
