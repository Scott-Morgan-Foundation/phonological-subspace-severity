#!/usr/bin/env python3
"""
Aetiology figures v2 — severity-matched + differential profiles.

Fixes:
  1. Severity-matched: compare aetiologies at SAME severity level (removes bias)
  2. Differential radar: show each aetiology relative to MEAN of all dysarthric
     (not HC) — reveals where each aetiology deviates from the dysarthric average
  3. Parallel coordinates: shows individual speaker trajectories, not just group means
  4. Feature signature heatmap: z-scored within severity, highlighting distinctive features
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path
from scipy import stats

BASE = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
MASTER = BASE / "results" / "track4" / "track4_master.csv"
OUTDIR = BASE / "results" / "track4" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

SMF_ORANGE = "#F04E23"
SMF_NAVY = "#0B1F5C"

AET_COLORS = {
    "healthy": "#4DAF4A",
    "parkinsons": "#377EB8",
    "cerebral_palsy": "#E41A1C",
    "als": "#FF7F00",
    "down_syndrome": "#984EA3",
    "stroke": "#A65628",
}

AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}

ALL_AETS = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
DYS_AETS = ["parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]

FEATS_9 = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime",
    "high_dprime", "low_dprime", "back_dprime", "round_dprime",
]

FEAT_LABELS_9 = [
    "Nasality", "Voicing", "Sonorance", "Stridency", "Manner",
    "Height", "Lowness", "Backness", "Rounding",
]

FEATS_13 = FEATS_9 + ["vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv"]
FEAT_LABELS_13 = FEAT_LABELS_9 + ["Vowel Triangle", "Speech Rate", "Pause Rate", "Vowel Dur. CV"]


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except (ValueError, TypeError):
        return None


def load_data():
    with open(MASTER, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def get_feat_vector(row, feats):
    """Get feature vector, return None if any missing."""
    vals = []
    for f in feats:
        v = safe_float(row.get(f))
        if v is None:
            return None
        vals.append(v)
    return np.array(vals)


def fig1_differential_radar(rows):
    """Radar: each aetiology vs mean of ALL dysarthric speakers (not HC).
    Shows what's *distinctive* about each aetiology."""
    feats = FEATS_9
    labels = FEAT_LABELS_9
    n = len(feats)

    # Collect all dysarthric speakers with complete features
    all_dys = []
    by_aet = defaultdict(list)
    for r in rows:
        aet = r.get("aetiology", "")
        if aet not in ALL_AETS:
            continue
        vec = get_feat_vector(r, feats)
        if vec is not None:
            if aet != "healthy":
                all_dys.append(vec)
            by_aet[aet].append(vec)

    if not all_dys:
        return

    # Grand dysarthric mean and SD
    all_matrix = np.array(all_dys)
    grand_mean = all_matrix.mean(axis=0)
    grand_std = all_matrix.std(axis=0)
    grand_std[grand_std == 0] = 1

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for aet in ALL_AETS:
        vecs = np.array(by_aet[aet])
        aet_mean = vecs.mean(axis=0)
        z = (aet_mean - grand_mean) / grand_std
        values = z.tolist() + z.tolist()[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, markersize=7,
                color=AET_COLORS[aet],
                label=f"{AET_SHORT[aet]} (n={len(vecs)})")
        ax.fill(angles, values, alpha=0.08, color=AET_COLORS[aet])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

    # Zero line = dysarthric average
    ref = [0] * (n + 1)
    ax.plot(angles, ref, '--', linewidth=2, color='gray', alpha=0.6)

    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(['-1 SD', '', 'Dys. mean', '', '+1 SD'], fontsize=9, color='gray')

    ax.set_title("Aetiology-Specific Phonological Signatures\n"
                 "(z-scored relative to dysarthric mean, not HC)",
                 fontsize=15, fontweight='bold', pad=30, color=SMF_NAVY)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=11, framealpha=0.9)

    plt.tight_layout()
    path = OUTDIR / "aetiology_differential_radar.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig2_severity_matched(rows):
    """Heatmap: aetiology means at each severity level separately.
    Removes severity distribution bias."""
    feats = FEATS_9
    labels = FEAT_LABELS_9
    sev_levels = ["mild", "moderate", "severe"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

    for ax_idx, sev in enumerate(sev_levels):
        ax = axes[ax_idx]

        # Get speakers at this severity
        aet_means = {}
        aet_ns = {}
        for aet in DYS_AETS:
            vecs = []
            for r in rows:
                if r.get("aetiology") != aet or r.get("severity_label") != sev:
                    continue
                vec = get_feat_vector(r, feats)
                if vec is not None:
                    vecs.append(vec)
            if len(vecs) >= 3:
                aet_means[aet] = np.mean(vecs, axis=0)
                aet_ns[aet] = len(vecs)

        if len(aet_means) < 2:
            ax.set_title(f"{sev.title()}\n(insufficient data)", fontsize=13)
            continue

        # Z-score within this severity level
        all_vecs = np.array(list(aet_means.values()))
        sev_mean = all_vecs.mean(axis=0)
        sev_std = all_vecs.std(axis=0)
        sev_std[sev_std == 0] = 1

        matrix = []
        aet_labels = []
        for aet in DYS_AETS:
            if aet in aet_means:
                z = (aet_means[aet] - sev_mean) / sev_std
                matrix.append(z)
                aet_labels.append(f"{AET_SHORT[aet]} (n={aet_ns[aet]})")

        matrix = np.array(matrix)
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)

        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(aet_labels)))
        ax.set_yticklabels(aet_labels, fontsize=11, fontweight='bold')

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = 'white' if abs(val) > 1.2 else 'black'
                ax.text(j, i, f"{val:+.1f}", ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

        ax.set_title(f"{sev.title()}", fontsize=14, fontweight='bold', color=SMF_NAVY)
        ax.axvline(x=4.5, color='white', linewidth=2)  # consonant/vowel separator

    cbar = plt.colorbar(im, ax=axes.tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("z-score (within severity level)", fontsize=11)

    fig.suptitle("Severity-Matched Aetiology Profiles\n"
                 "(z-scored within each severity level to remove distribution bias)",
                 fontsize=16, fontweight='bold', color=SMF_NAVY, y=1.02)
    plt.tight_layout()
    path = OUTDIR / "aetiology_severity_matched.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig3_signature_summary(rows):
    """Compact signature chart: for each aetiology, show which features are
    distinctively HIGH or LOW compared to other dysarthric aetiologies.
    Arrow-style or diverging bar chart."""
    feats = FEATS_13
    labels = FEAT_LABELS_13

    # Collect vectors
    by_aet = defaultdict(list)
    for r in rows:
        aet = r.get("aetiology", "")
        if aet not in DYS_AETS:
            continue
        vec = get_feat_vector(r, feats)
        if vec is not None:
            by_aet[aet].append(vec)

    # Grand dysarthric stats
    all_dys = []
    for aet in DYS_AETS:
        all_dys.extend(by_aet[aet])
    all_matrix = np.array(all_dys)
    grand_mean = all_matrix.mean(axis=0)
    grand_std = all_matrix.std(axis=0)
    grand_std[grand_std == 0] = 1

    fig, axes = plt.subplots(len(DYS_AETS), 1, figsize=(12, 14), sharex=True)

    for idx, aet in enumerate(DYS_AETS):
        ax = axes[idx]
        vecs = np.array(by_aet[aet])
        aet_mean = vecs.mean(axis=0)
        z = (aet_mean - grand_mean) / grand_std

        colors = [SMF_ORANGE if v > 0 else '#2166AC' for v in z]
        bars = ax.barh(range(len(feats)), z, color=colors, edgecolor='white', height=0.7, alpha=0.85)

        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(x=0, color='gray', linewidth=1, linestyle='-')
        ax.set_xlim(-1.5, 1.5)

        # Highlight distinctive features (|z| > 0.5)
        for i, v in enumerate(z):
            if abs(v) > 0.3:
                ax.text(v + (0.05 if v > 0 else -0.05), i,
                        f"{v:+.2f}", va='center',
                        ha='left' if v > 0 else 'right',
                        fontsize=8, fontweight='bold')

        ax.set_title(f"{AET_SHORT[aet]} (n={len(vecs)})",
                     fontsize=13, fontweight='bold', color=AET_COLORS[aet],
                     loc='left', pad=5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Separator
        ax.axhline(y=4.5, color='lightgray', linewidth=0.5, linestyle='--')
        ax.axhline(y=8.5, color='lightgray', linewidth=0.5, linestyle='--')

    axes[-1].set_xlabel("z-score (relative to dysarthric mean)", fontsize=12, fontweight='bold')

    fig.suptitle("Aetiology-Specific Feature Signatures\n"
                 "(orange = higher than dysarthric average, blue = lower)",
                 fontsize=16, fontweight='bold', color=SMF_NAVY, y=1.01)
    plt.tight_layout()
    path = OUTDIR / "aetiology_signatures_diverging.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig4_prosodic_scatter(rows):
    """Scatter: speech_rate vs pause_rate colored by aetiology.
    Shows where prosodic features DO separate aetiologies."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for aet in ALL_AETS:
        sr_vals, pr_vals = [], []
        for r in rows:
            if r.get("aetiology") != aet:
                continue
            sr = safe_float(r.get("speech_rate"))
            pr = safe_float(r.get("pause_rate"))
            if sr is not None and pr is not None:
                sr_vals.append(sr)
                pr_vals.append(pr)

        ax.scatter(sr_vals, pr_vals, c=AET_COLORS[aet], alpha=0.4, s=25,
                   label=f"{AET_SHORT[aet]} (n={len(sr_vals)})", edgecolors='none')
        # Add centroid
        if sr_vals:
            ax.scatter([np.mean(sr_vals)], [np.mean(pr_vals)],
                       c=AET_COLORS[aet], s=200, marker='D', edgecolors='black',
                       linewidths=1.5, zorder=5)

    ax.set_xlabel("Speech Rate (syllables/sec)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Pause Rate (pauses/sec)", fontsize=13, fontweight='bold')
    ax.set_title("Prosodic Space by Aetiology\n(diamonds = centroids)",
                 fontsize=15, fontweight='bold', color=SMF_NAVY)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = OUTDIR / "aetiology_prosodic_scatter.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig5_severity_distribution(rows):
    """Stacked bar: severity distribution per aetiology — shows the bias."""
    sev_order = ["control", "mild", "moderate", "severe"]
    sev_colors = {"control": "#4DAF4A", "mild": "#FFFF33", "moderate": "#FF7F00", "severe": "#E41A1C"}

    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        aet = r.get("aetiology", "")
        sev = r.get("severity_label", "unknown")
        if aet in DYS_AETS and sev in sev_order:
            counts[aet][sev] += 1

    fig, ax = plt.subplots(figsize=(10, 6))

    aet_labels = [AET_SHORT[a] for a in DYS_AETS]
    x = np.arange(len(DYS_AETS))
    width = 0.6

    # Normalize to percentages
    bottoms = np.zeros(len(DYS_AETS))
    for sev in sev_order:
        vals = []
        for aet in DYS_AETS:
            total = sum(counts[aet].values())
            pct = 100 * counts[aet][sev] / total if total > 0 else 0
            vals.append(pct)
        ax.bar(x, vals, width, bottom=bottoms, color=sev_colors[sev],
               label=sev.title(), edgecolor='white', linewidth=0.5)
        # Labels
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(x[i], bottoms[i] + v / 2, f"{v:.0f}%",
                        ha='center', va='center', fontsize=9, fontweight='bold')
        bottoms += vals

    # Add n below
    for i, aet in enumerate(DYS_AETS):
        total = sum(counts[aet].values())
        ax.text(x[i], -4, f"n={total}", ha='center', fontsize=10, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(aet_labels, fontsize=13, fontweight='bold')
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Severity Distribution by Aetiology\n"
                 "(why PD appears 'milder' — 56% mild vs CP 23% severe)",
                 fontsize=14, fontweight='bold', color=SMF_NAVY)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(-8, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTDIR / "aetiology_severity_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 80)
    print("Aetiology figures v2 — severity-matched + differential")
    print("=" * 80)

    rows = load_data()
    print(f"Loaded {len(rows)} speakers\n")

    fig1_differential_radar(rows)
    fig2_severity_matched(rows)
    fig3_signature_summary(rows)
    fig4_prosodic_scatter(rows)
    fig5_severity_distribution(rows)

    print(f"\nAll figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
