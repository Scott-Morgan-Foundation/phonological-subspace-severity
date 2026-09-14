#!/usr/bin/env python3
"""
Cross-model comparison figures for Paper 2.

1. Heatmap: inter-model agreement (Spearman rho)
2. Grouped bar: severity means per model
3. Heatmap: aetiology profile cosine sim vs HuBERT-base
4. Grouped bar: severity rho per feature per model
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine

BASE = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
RESULTS_DIR = BASE / "results" / "track4"
MASTER = RESULTS_DIR / "track4_master.csv"
OUTDIR = RESULTS_DIR / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

SMF_NAVY = "#0B1F5C"
SMF_ORANGE = "#F04E23"

MODELS = {
    "HuBERT-base": MASTER,
    "WavLM": RESULTS_DIR / "track4_results_wavlm.csv",
    "wav2vec2": RESULTS_DIR / "track4_results_wav2vec2.csv",
    "XLS-R": RESULTS_DIR / "track4_results_xlsr.csv",
    "MMS": RESULTS_DIR / "track4_results_mms.csv",
    "HuBERT-large": RESULTS_DIR / "track4_results_hubert-large.csv",
}

MODEL_COLORS = {
    "HuBERT-base": "#377EB8",
    "WavLM": "#4DAF4A",
    "wav2vec2": "#E41A1C",
    "XLS-R": "#FF7F00",
    "MMS": "#984EA3",
    "HuBERT-large": "#0B1F5C",
}

CONS_FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
FEAT_LABELS = ["Nasality", "Voicing", "Sonorance", "Stridency", "Manner"]
SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
SEV_ORDER = ["control", "mild", "moderate", "severe"]
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}
AETS = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome"]


def safe_float(v):
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except:
        return None


def composite_score(row):
    vals = [safe_float(row.get(f)) for f in CONS_FEATS]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if len(vals) >= 3 else None


def load_all():
    model_data = {}
    for name, path in MODELS.items():
        if not path.exists():
            continue
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                data[(r["dataset"], r["speaker_id"])] = r
        model_data[name] = data
    return model_data


def fig1_inter_model_heatmap(model_data):
    """Heatmap of inter-model Spearman rho on composite d-prime."""
    names = list(model_data.keys())
    n = len(names)

    # Compute composite per speaker per model
    scores = {}
    for name in names:
        s = {}
        for key, row in model_data[name].items():
            cs = composite_score(row)
            if cs is not None:
                s[key] = cs
        scores[name] = s

    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                common = set(scores[names[i]].keys()) & set(scores[names[j]].keys())
                if len(common) >= 20:
                    s1 = [scores[names[i]][k] for k in common]
                    s2 = [scores[names[j]][k] for k in common]
                    matrix[i, j], _ = stats.spearmanr(s1, s2)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='YlGnBu', vmin=0.7, vmax=1.0, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=11, fontweight='bold', rotation=30, ha='right')
    ax.set_yticklabels(names, fontsize=11, fontweight='bold')

    for i in range(n):
        for j in range(n):
            color = 'white' if matrix[i, j] < 0.85 else 'black'
            if i == j:
                ax.text(j, i, "1.00", ha='center', va='center', fontsize=10, color='gray')
            else:
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("Inter-Model Speaker-Level Agreement\n(composite consonant d-prime)",
                 fontsize=14, fontweight='bold', color=SMF_NAVY)
    plt.tight_layout()
    path = OUTDIR / "cross_model_agreement_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig2_severity_bars(model_data, severity_labels):
    """Grouped bar: severity means per model."""
    names = list(model_data.keys())
    n_models = len(names)
    n_sev = len(SEV_ORDER)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_sev)
    width = 0.8 / n_models

    for i, name in enumerate(names):
        means = []
        for sev in SEV_ORDER:
            sev_num = SEV_MAP[sev]
            vals = []
            for key, row in model_data[name].items():
                if key in severity_labels and severity_labels[key] == sev_num:
                    cs = composite_score(row)
                    if cs is not None:
                        vals.append(cs)
            means.append(np.mean(vals) if vals else 0)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, color=MODEL_COLORS[name], alpha=0.85,
               label=name, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in SEV_ORDER], fontsize=12, fontweight='bold')
    ax.set_ylabel("Composite consonant d-prime", fontsize=12)
    ax.set_title("Severity Gradient Across 6 SSL Backbones\n(all models show monotonic decrease)",
                 fontsize=14, fontweight='bold', color=SMF_NAVY)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTDIR / "cross_model_severity_bars.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig3_aetiology_cosine_heatmap(model_data, aetiology_labels):
    """Heatmap: aetiology profile cosine sim vs HuBERT-base."""
    ref = model_data["HuBERT-base"]
    other_names = [n for n in model_data if n != "HuBERT-base"]

    def aet_profile(data, aet):
        rows = [data[k] for k in data if aetiology_labels.get(k) == aet]
        means = []
        for f in CONS_FEATS:
            vals = [safe_float(r.get(f)) for r in rows]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else np.nan)
        return np.array(means)

    ref_profiles = {aet: aet_profile(ref, aet) for aet in AETS}

    matrix = np.zeros((len(other_names), len(AETS)))
    for i, name in enumerate(other_names):
        for j, aet in enumerate(AETS):
            p = aet_profile(model_data[name], aet)
            mask = ~(np.isnan(ref_profiles[aet]) | np.isnan(p))
            if mask.sum() >= 3:
                matrix[i, j] = 1 - cosine(ref_profiles[aet][mask], p[mask])
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.95, vmax=1.0, aspect='auto')

    aet_labels = [AET_SHORT[a] for a in AETS]
    ax.set_xticks(range(len(AETS)))
    ax.set_yticks(range(len(other_names)))
    ax.set_xticklabels(aet_labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels(other_names, fontsize=11, fontweight='bold')

    for i in range(len(other_names)):
        for j in range(len(AETS)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.97 else 'black'
                ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity vs HuBERT-base")
    ax.set_title("Aetiology Profile Consistency Across SSL Backbones\n"
                 "(5 consonant d-primes, compared to HuBERT-base reference)",
                 fontsize=13, fontweight='bold', color=SMF_NAVY)
    plt.tight_layout()
    path = OUTDIR / "cross_model_aetiology_cosine.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def fig4_feature_rho_bars(model_data, severity_labels):
    """Grouped bar: absolute severity rho per feature per model."""
    names = list(model_data.keys())
    n_models = len(names)
    n_feats = len(CONS_FEATS)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_feats)
    width = 0.8 / n_models

    for i, name in enumerate(names):
        rhos = []
        for f in CONS_FEATS:
            pairs = [(severity_labels[k], safe_float(model_data[name][k].get(f)))
                     for k in model_data[name] if k in severity_labels]
            pairs = [(s, v) for s, v in pairs if v is not None]
            if len(pairs) >= 20:
                rho, _ = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
                rhos.append(abs(rho))
            else:
                rhos.append(0)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, rhos, width, color=MODEL_COLORS[name], alpha=0.85,
               label=name, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(FEAT_LABELS, fontsize=12, fontweight='bold')
    ax.set_ylabel("|Spearman rho| with severity", fontsize=12)
    ax.set_title("Feature-Level Severity Correlation Across SSL Backbones",
                 fontsize=14, fontweight='bold', color=SMF_NAVY)
    ax.legend(fontsize=9, framealpha=0.9, ncol=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 0.7)

    plt.tight_layout()
    path = OUTDIR / "cross_model_feature_rho.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 80)
    print("Cross-model comparison figures")
    print("=" * 80)

    model_data = load_all()
    print(f"Loaded {len(model_data)} models\n")

    # Get labels from master
    master = model_data["HuBERT-base"]
    severity_labels = {}
    aetiology_labels = {}
    for key, row in master.items():
        sev = row.get("severity_label", "unknown")
        if sev in SEV_MAP:
            severity_labels[key] = SEV_MAP[sev]
        aetiology_labels[key] = row.get("aetiology", "unknown")

    fig1_inter_model_heatmap(model_data)
    fig2_severity_bars(model_data, severity_labels)
    fig3_aetiology_cosine_heatmap(model_data, aetiology_labels)
    fig4_feature_rho_bars(model_data, severity_labels)

    print(f"\nAll figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
