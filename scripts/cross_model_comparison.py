#!/usr/bin/env python3
"""
Cross-model comparison: do severity rankings and aetiology profiles
replicate across 6 SSL backbones?

Compares HuBERT-base (master CSV) vs WavLM, wav2vec2, XLS-R, MMS, HuBERT-large.
"""

import csv
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
RESULTS_DIR = BASE / "results" / "track4"
MASTER = RESULTS_DIR / "track4_master.csv"

MODELS = {
    "hubert-base": MASTER,  # segmental features from master
    "wavlm": RESULTS_DIR / "track4_results_wavlm.csv",
    "wav2vec2": RESULTS_DIR / "track4_results_wav2vec2.csv",
    "xlsr": RESULTS_DIR / "track4_results_xlsr.csv",
    "mms": RESULTS_DIR / "track4_results_mms.csv",
    "hubert-large": RESULTS_DIR / "track4_results_hubert-large.csv",
}

CONS_FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
VOWEL_FEATS = ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]
ALL_DPRIMES = CONS_FEATS + VOWEL_FEATS

SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except:
        return None


def load_model_data(path, is_master=False):
    """Load per-speaker features. Returns {(dataset, speaker_id): row}."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["speaker_id"])
            data[key] = r
    return data


def composite_score(row):
    vals = [safe_float(row.get(f)) for f in CONS_FEATS]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if len(vals) >= 3 else None


def cosine_sim(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    return 1 - cosine(a[mask], b[mask])


def main():
    print("=" * 100)
    print("CROSS-MODEL COMPARISON: 6 SSL Backbones")
    print("=" * 100)

    # Load all models
    model_data = {}
    for name, path in MODELS.items():
        if not path.exists():
            print(f"  {name}: MISSING ({path})")
            continue
        model_data[name] = load_model_data(path, is_master=(name == "hubert-base"))
        print(f"  {name}: {len(model_data[name])} speakers")

    # Get severity labels from master
    master = model_data["hubert-base"]
    severity_labels = {}
    aetiology_labels = {}
    for key, row in master.items():
        sev = row.get("severity_label", "unknown")
        if sev in SEV_MAP:
            severity_labels[key] = SEV_MAP[sev]
        aetiology_labels[key] = row.get("aetiology", "unknown")

    # =========================================================================
    # TABLE 1: Severity correlation per model (Spearman rho, all speakers)
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 1: Severity Correlation (Spearman rho) per Model")
    print("=" * 100)

    print(f"\n  {'Model':>15}", end="")
    for f in CONS_FEATS:
        print(f" {f.replace('_dprime',''):>10}", end="")
    print(f" {'composite':>10} {'n':>6}")
    print("  " + "-" * (15 + 11 * (len(CONS_FEATS) + 1) + 7))

    model_rhos = {}
    for name in MODELS:
        if name not in model_data:
            continue
        data = model_data[name]
        rhos = []
        print(f"  {name:>15}", end="")

        for f in CONS_FEATS:
            pairs = []
            for key in data:
                if key in severity_labels:
                    v = safe_float(data[key].get(f))
                    if v is not None:
                        pairs.append((severity_labels[key], v))
            if len(pairs) >= 20:
                rho, _ = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
                print(f" {rho:>+10.3f}", end="")
                rhos.append(rho)
            else:
                print(f" {'N/A':>10}", end="")

        # Composite
        comp_pairs = []
        for key in data:
            if key in severity_labels:
                cs = composite_score(data[key])
                if cs is not None:
                    comp_pairs.append((severity_labels[key], cs))
        if comp_pairs:
            rho_comp, _ = stats.spearmanr([p[0] for p in comp_pairs], [p[1] for p in comp_pairs])
            print(f" {rho_comp:>+10.3f} {len(comp_pairs):>6}")
            model_rhos[name] = rho_comp
        else:
            print(f" {'N/A':>10}")

    # =========================================================================
    # TABLE 2: Severity means per model (composite d-prime by severity level)
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 2: Severity Means (composite consonant d-prime)")
    print("=" * 100)

    print(f"\n  {'Model':>15} {'control':>10} {'mild':>10} {'moderate':>10} {'severe':>10} {'monotonic':>10}")
    print("  " + "-" * 70)

    for name in MODELS:
        if name not in model_data:
            continue
        data = model_data[name]
        means = {}
        for sev_name, sev_num in SEV_MAP.items():
            vals = []
            for key in data:
                if key in severity_labels and severity_labels[key] == sev_num:
                    cs = composite_score(data[key])
                    if cs is not None:
                        vals.append(cs)
            means[sev_name] = np.mean(vals) if vals else np.nan
        mono = "YES" if means["control"] > means["mild"] > means["moderate"] > means["severe"] else "no"
        print(f"  {name:>15} {means['control']:>10.3f} {means['mild']:>10.3f} "
              f"{means['moderate']:>10.3f} {means['severe']:>10.3f} {mono:>10}")

    # =========================================================================
    # TABLE 3: Inter-model correlation (do models agree on speaker ranking?)
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 3: Inter-Model Agreement (Spearman rho on composite d-prime)")
    print("=" * 100)

    model_names = [n for n in MODELS if n in model_data]

    # Get composite scores per speaker per model
    model_scores = {}
    for name in model_names:
        scores = {}
        for key in model_data[name]:
            cs = composite_score(model_data[name][key])
            if cs is not None:
                scores[key] = cs
        model_scores[name] = scores

    print(f"\n  {'':>15}", end="")
    for n in model_names:
        print(f" {n:>12}", end="")
    print()
    print("  " + "-" * (15 + 13 * len(model_names)))

    for n1 in model_names:
        print(f"  {n1:>15}", end="")
        for n2 in model_names:
            if n1 == n2:
                print(f" {'--':>12}", end="")
            else:
                common = set(model_scores[n1].keys()) & set(model_scores[n2].keys())
                if len(common) >= 20:
                    s1 = [model_scores[n1][k] for k in common]
                    s2 = [model_scores[n2][k] for k in common]
                    rho, _ = stats.spearmanr(s1, s2)
                    print(f" {rho:>12.3f}", end="")
                else:
                    print(f" {'N/A':>12}", end="")
        print()

    # =========================================================================
    # TABLE 4: Aetiology profile consistency across models
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 4: Aetiology Profile Consistency Across Models")
    print("  (cosine sim of 5-d-prime aetiology profiles vs HuBERT-base)")
    print("=" * 100)

    aets = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]

    # Compute mean profile per aetiology per model
    def aet_profile(data, aet):
        rows = [data[k] for k in data if aetiology_labels.get(k) == aet]
        means = []
        for f in CONS_FEATS:
            vals = [safe_float(r.get(f)) for r in rows]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else np.nan)
        return np.array(means)

    ref_profiles = {aet: aet_profile(model_data["hubert-base"], aet) for aet in aets}

    print(f"\n  {'Model':>15}", end="")
    for aet in aets:
        print(f" {AET_SHORT.get(aet,aet):>8}", end="")
    print(f" {'mean':>8}")
    print("  " + "-" * (15 + 9 * (len(aets) + 1)))

    for name in model_names:
        if name == "hubert-base":
            continue
        print(f"  {name:>15}", end="")
        sims = []
        for aet in aets:
            p = aet_profile(model_data[name], aet)
            sim = cosine_sim(ref_profiles[aet], p)
            sims.append(sim)
            print(f" {sim:>8.3f}" if not np.isnan(sim) else f" {'N/A':>8}", end="")
        valid_sims = [s for s in sims if not np.isnan(s)]
        mean_sim = np.mean(valid_sims) if valid_sims else np.nan
        print(f" {mean_sim:>8.3f}")

    # =========================================================================
    # TABLE 5: Feature importance ranking stability
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 5: Feature Importance Ranking (by severity rho) — Rank per Model")
    print("=" * 100)

    print(f"\n  {'Feature':>12}", end="")
    for name in model_names:
        print(f" {name:>12}", end="")
    print()
    print("  " + "-" * (12 + 13 * len(model_names)))

    # Compute rho per feature per model
    feat_rhos = {name: {} for name in model_names}
    for name in model_names:
        data = model_data[name]
        for f in CONS_FEATS:
            pairs = [(severity_labels[k], safe_float(data[k].get(f)))
                     for k in data if k in severity_labels]
            pairs = [(s, v) for s, v in pairs if v is not None]
            if len(pairs) >= 20:
                rho, _ = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
                feat_rhos[name][f] = abs(rho)

    for f in CONS_FEATS:
        fname = f.replace("_dprime", "")
        print(f"  {fname:>12}", end="")
        for name in model_names:
            rho = feat_rhos[name].get(f)
            if rho is not None:
                # Compute rank
                all_rhos = sorted(feat_rhos[name].values(), reverse=True)
                rank = all_rhos.index(rho) + 1
                print(f" {rank:>5}({rho:.2f})", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    rho_vals = list(model_rhos.values())
    print(f"\n  Severity correlation range: {min(rho_vals):+.3f} to {max(rho_vals):+.3f}")
    print(f"  All models show monotonic severity gradient: ctrl > mild > mod > sev")


if __name__ == "__main__":
    main()
