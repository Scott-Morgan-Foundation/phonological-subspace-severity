#!/usr/bin/env python3
"""
Cross-lingual aetiology consistency figures.

Fig 1: Overlaid radar per aetiology — one line per language, HC-normalized
Fig 2: Cosine similarity heatmaps
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from scipy.spatial.distance import cosine

BASE = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
MASTER = BASE / "results" / "track4" / "track4_master.csv"
OUTDIR = BASE / "results" / "track4" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

SMF_NAVY = "#0B1F5C"
SMF_ORANGE = "#F04E23"

FEATS = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime",
    "high_dprime", "low_dprime", "back_dprime", "round_dprime",
]
FEAT_LABELS = ["Nasality", "Voicing", "Sonorance", "Stridency", "Manner",
               "Height", "Lowness", "Backness", "Rounding"]

# Distinct colors per language
LANG_COLORS = {
    "en": "#E41A1C", "es": "#377EB8", "sk": "#4DAF4A", "it": "#984EA3",
    "pt": "#FF7F00", "de": "#A65628", "fr": "#F781BF", "hu": "#999999",
    "sw": "#FFD700", "zh": "#E41A1C", "ta": "#00CED1", "nl": "#8B4513",
}

LANG_NAMES = {
    "en": "English", "es": "Spanish", "sk": "Slovak", "it": "Italian",
    "pt": "Portuguese", "de": "German", "fr": "French", "hu": "Hungarian",
    "sw": "Swahili", "zh": "Mandarin", "ta": "Tamil", "nl": "Dutch",
}

AET_SHORT = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS", "healthy": "HC"}


def safe_float(v):
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except:
        return None


def mean_profile(subset):
    means = []
    for f in FEATS:
        vals = [safe_float(r.get(f)) for r in subset]
        vals = [v for v in vals if v is not None]
        means.append(np.mean(vals) if vals else np.nan)
    return np.array(means)


def load_data():
    with open(MASTER, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_aet_lang = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_aet_lang[r["aetiology"]][r["language"]].append(r)
    return rows, by_aet_lang


def fig1_overlaid_radars(rows, by_aet_lang):
    """One radar per aetiology, overlaid lines per language. HC-normalized per language."""
    aets = ["parkinsons", "cerebral_palsy", "als"]
    n_feats = len(FEATS)
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), subplot_kw=dict(polar=True))

    for ax_idx, aet in enumerate(aets):
        ax = axes[ax_idx]
        short = AET_SHORT[aet]

        langs = sorted(by_aet_lang[aet].keys())
        # Only languages with n>=3 for aetiology, n>=1 for HC, and at least 5 valid features
        usable_langs = []
        for lang in langs:
            n_dys = len(by_aet_lang[aet][lang])
            n_hc = len(by_aet_lang["healthy"].get(lang, []))
            p = mean_profile(by_aet_lang[aet][lang])
            if n_dys >= 3 and n_hc >= 1 and np.sum(~np.isnan(p)) >= 5:
                usable_langs.append(lang)

        # HC reference ring (grand HC mean)
        all_hc = []
        for lang_rows in by_aet_lang["healthy"].values():
            all_hc.extend(lang_rows)
        hc_grand = mean_profile(all_hc)
        hc_safe = hc_grand.copy()
        hc_safe[hc_safe == 0] = 1
        hc_normed = (hc_grand / hc_safe).tolist() + [(hc_grand / hc_safe)[0]]
        ax.plot(angles, hc_normed, '--', linewidth=1.5, color='lightgray', alpha=0.8, zorder=1)
        ax.fill(angles, hc_normed, alpha=0.05, color='lightgray')

        for lang in usable_langs:
            n = len(by_aet_lang[aet][lang])
            p = mean_profile(by_aet_lang[aet][lang])

            # HC-normalize using language-specific HC if available, else grand HC
            hc_lang = by_aet_lang["healthy"].get(lang, [])
            if len(hc_lang) >= 3:
                hc_ref = mean_profile(hc_lang)
                hc_ref_safe = hc_ref.copy()
                hc_ref_safe[hc_ref_safe == 0] = 1
                normed = p / hc_ref_safe
            else:
                normed = p / hc_safe

            vals = normed.tolist() + [normed[0]]
            # Replace NaN with None to break line
            vals = [v if not np.isnan(v) else None for v in vals]

            color = LANG_COLORS.get(lang, "#333333")
            lw = 2.5 if n >= 10 else 1.5
            ls = '-' if n >= 3 else ':'
            ms = 6 if n >= 10 else 4
            label = f"{LANG_NAMES.get(lang, lang)} (n={n})"

            # Plot — handle None gaps
            valid_angles = [a for a, v in zip(angles, vals) if v is not None]
            valid_vals = [v for v in vals if v is not None]
            if valid_vals:
                ax.plot(valid_angles, valid_vals, 'o-', linewidth=lw, linestyle=ls,
                        markersize=ms, color=color, label=label, zorder=3, alpha=0.85)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FEAT_LABELS, fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['', '0.4', '', '0.8', '1.0'], fontsize=7, color='gray')

        n_langs = len(usable_langs)
        ax.set_title(f"{short} across {n_langs} languages",
                     fontsize=14, fontweight='bold', color=SMF_NAVY, pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=8, framealpha=0.9)

    fig.suptitle("Cross-Lingual Aetiology Profiles\n"
                 "(normalised to language-specific HC = 1.0, dashed gray = HC reference)",
                 fontsize=16, fontweight='bold', color=SMF_NAVY, y=1.04)
    plt.tight_layout()
    path = OUTDIR / "cross_lingual_radars.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig2_cosine_heatmaps(rows, by_aet_lang):
    """Cosine similarity heatmaps per aetiology."""

    def cosine_sim(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 3:
            return np.nan
        return 1 - cosine(a[mask], b[mask])

    aets = ["healthy", "parkinsons", "cerebral_palsy", "als"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for ax_idx, aet in enumerate(aets):
        ax = axes[ax_idx]
        short = AET_SHORT[aet]

        langs = sorted([l for l, rs in by_aet_lang[aet].items()
                        if np.sum(~np.isnan(mean_profile(rs))) >= 5])

        if len(langs) < 2:
            ax.set_title(f"{short}\n(< 2 languages)", fontsize=12)
            ax.axis('off')
            continue

        profiles = {l: mean_profile(by_aet_lang[aet][l]) for l in langs}
        n_langs = len(langs)
        matrix = np.ones((n_langs, n_langs))

        for i in range(n_langs):
            for j in range(n_langs):
                if i != j:
                    matrix[i, j] = cosine_sim(profiles[langs[i]], profiles[langs[j]])

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0, aspect='auto')

        lang_labels = [f"{LANG_NAMES.get(l,l)[:3]}\n({len(by_aet_lang[aet][l])})" for l in langs]
        ax.set_xticks(range(n_langs))
        ax.set_yticks(range(n_langs))
        ax.set_xticklabels(lang_labels, fontsize=7)
        ax.set_yticklabels(lang_labels, fontsize=7)

        for i in range(n_langs):
            for j in range(n_langs):
                val = matrix[i, j]
                if i == j:
                    continue
                color = 'white' if val < 0.92 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=6, fontweight='bold', color=color)

        mean_sim = np.nanmean(matrix[np.triu_indices(n_langs, k=1)])
        ax.set_title(f"{short} ({n_langs} langs)\nmean cos = {mean_sim:.3f}",
                     fontsize=12, fontweight='bold', color=SMF_NAVY)

    plt.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02, label="Cosine similarity")
    fig.suptitle("Cross-Lingual Profile Consistency (9 d-prime features)",
                 fontsize=15, fontweight='bold', color=SMF_NAVY, y=1.02)
    plt.tight_layout()
    path = OUTDIR / "cross_lingual_cosine_heatmaps.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig3_profile_bars(rows, by_aet_lang):
    """Grouped bar chart: HC-normalized profiles per language for PD and CP."""
    n_feats = len(FEATS)

    for aet, min_n in [("parkinsons", 3), ("cerebral_palsy", 3)]:
        short = AET_SHORT[aet]
        langs = sorted([l for l, rs in by_aet_lang[aet].items()
                        if len(rs) >= min_n and np.sum(~np.isnan(mean_profile(rs))) >= 5])
        if len(langs) < 2:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(n_feats)
        width = 0.8 / len(langs)

        for i, lang in enumerate(langs):
            n = len(by_aet_lang[aet][lang])
            p = mean_profile(by_aet_lang[aet][lang])

            # HC-normalize
            hc_lang = by_aet_lang["healthy"].get(lang, [])
            if len(hc_lang) >= 3:
                hc_ref = mean_profile(hc_lang)
                hc_ref[hc_ref == 0] = 1
                normed = p / hc_ref
            else:
                all_hc = []
                for lr in by_aet_lang["healthy"].values():
                    all_hc.extend(lr)
                hc_ref = mean_profile(all_hc)
                hc_ref[hc_ref == 0] = 1
                normed = p / hc_ref

            # Replace NaN with 0 for plotting, but mark as missing
            normed_clean = np.where(np.isnan(normed), 0, normed)
            color = LANG_COLORS.get(lang, "#333333")
            offset = (i - len(langs)/2 + 0.5) * width
            # Only plot bars where data exists
            alphas = [0.8 if not np.isnan(normed[j]) else 0.0 for j in range(len(normed))]
            for j in range(len(normed)):
                if not np.isnan(normed[j]):
                    ax.bar(x[j] + offset, normed[j], width, color=color, alpha=0.8,
                           edgecolor='white',
                           label=f"{LANG_NAMES.get(lang, lang)} (n={n})" if j == 0 else "_nolegend_")

        ax.axhline(y=1.0, color='lightgray', linestyle='--', linewidth=1, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(FEAT_LABELS, fontsize=10, fontweight='bold')
        ax.set_ylabel("Ratio to HC (1.0 = healthy)", fontsize=12)
        ax.set_title(f"{short} — HC-Normalized Profiles by Language",
                     fontsize=15, fontweight='bold', color=SMF_NAVY)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 1.3)

        plt.tight_layout()
        path = OUTDIR / f"cross_lingual_bars_{short}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")


def main():
    print("=" * 80)
    print("Cross-lingual aetiology figures")
    print("=" * 80)

    rows, by_aet_lang = load_data()
    print(f"Loaded {len(rows)} speakers\n")

    fig1_overlaid_radars(rows, by_aet_lang)
    fig2_cosine_heatmaps(rows, by_aet_lang)
    fig3_profile_bars(rows, by_aet_lang)

    print(f"\nAll figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
