#!/usr/bin/env python3
"""
Track 4: Reviewer experiments 6 & 7.

6. Alignment quality proxy — MFA failure rate + phone duration anomalies as covariate
7. Task-stratified boundary sharpness — full numeric table with interaction
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
INPUT_CSV = RESULTS_DIR / "track4_results_merged.csv"
INVENTORY_CSV = RESULTS_DIR / "speaker_inventory.csv"
OUTPUT_DIR = RESULTS_DIR / "reviewer_experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

CONSONANT_FEATURES = ["nasal_dprime", "voicing_dprime", "strident_dprime",
                      "sonorant_dprime", "manner_dprime"]


def load_data():
    df = pd.read_csv(INPUT_CSV)
    df = df[df.severity_numeric >= 0]
    inv = pd.read_csv(INVENTORY_CSV)
    return df, inv


# =============================================================================
# Experiment 6: Alignment Quality Proxy
# =============================================================================

def experiment_6_alignment_quality(df, inv):
    """Use n_phones/n_sentences as alignment success proxy.

    Rationale: if MFA fails on an utterance, fewer phones are extracted.
    A speaker with many sentences but few phones has poor alignment.
    We compute phones_per_sentence as a proxy for alignment quality,
    then show the severity correlation holds after controlling for it.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Alignment Quality Proxy")
    print("=" * 70)
    print("Proxy: phones_per_sentence (fewer phones per sentence = worse alignment)")

    # Merge inventory to get n_sentences
    inv_sub = inv[["dataset", "speaker_id", "n_sentences"]].copy()
    merged = df.merge(inv_sub, on=["dataset", "speaker_id"], how="left")

    # Compute alignment quality proxy
    merged["phones_per_sentence"] = merged["n_phones"] / merged["n_sentences"].clip(lower=1)

    # Check: does alignment quality correlate with severity?
    valid = merged[["severity_numeric", "phones_per_sentence"]].dropna()
    rho_sev_aq, p_sev_aq = stats.spearmanr(valid.severity_numeric, valid.phones_per_sentence)
    print(f"\nAlignment quality vs severity: rho={rho_sev_aq:.4f}, p={p_sev_aq:.2e}")
    print("(Negative = severe speakers have fewer phones/sentence = worse alignment)")

    # Partial correlations: feature ~ severity | alignment_quality
    print(f"\n{'Feature':<25} {'Raw rho':>10} {'Partial rho':>12} {'Partial p':>12} {'AQ confound':>12}")
    print("-" * 75)

    results = {}
    for feat in CONSONANT_FEATURES + ["boundary_sharpness", "vowel_triangle_area"]:
        valid = merged[["severity_numeric", feat, "phones_per_sentence"]].dropna()
        if len(valid) < 10:
            continue

        rho_raw, _ = stats.spearmanr(valid.severity_numeric, valid[feat])

        # Partial correlation controlling for phones_per_sentence
        r_sf, _ = stats.spearmanr(valid.severity_numeric, valid[feat])
        r_sa, _ = stats.spearmanr(valid.severity_numeric, valid.phones_per_sentence)
        r_fa, _ = stats.spearmanr(valid[feat], valid.phones_per_sentence)

        denom = np.sqrt((1 - r_sa**2) * (1 - r_fa**2))
        rho_partial = (r_sf - r_sa * r_fa) / denom if denom > 1e-8 else r_sf

        n = len(valid)
        t_stat = rho_partial * np.sqrt((n - 3) / (1 - rho_partial**2 + 1e-10))
        p_partial = 2 * stats.t.sf(abs(t_stat), n - 3)

        # How much does controlling for AQ change the correlation?
        confound_pct = (1 - abs(rho_partial) / abs(rho_raw)) * 100 if abs(rho_raw) > 0.01 else 0

        star = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else ""
        print(f"{feat:<25} {rho_raw:>10.4f} {rho_partial:>12.4f} {p_partial:>12.2e} {confound_pct:>11.1f}% {star}")

        results[feat] = {
            "raw_rho": float(rho_raw),
            "partial_rho": float(rho_partial),
            "partial_p": float(p_partial),
            "confound_pct": float(confound_pct),
            "n": int(n),
        }

    # Sensitivity: exclude speakers with worst alignment (bottom 10%)
    print("\n--- Sensitivity: Exclude bottom 10% alignment quality ---")
    threshold = merged.phones_per_sentence.quantile(0.10)
    good_align = merged[merged.phones_per_sentence >= threshold]
    print(f"Threshold: {threshold:.1f} phones/sentence")
    print(f"Speakers retained: {len(good_align)}/{len(merged)} ({len(good_align)/len(merged)*100:.1f}%)")

    print(f"\n{'Feature':<25} {'Full rho':>10} {'Filtered rho':>12} {'Change':>10}")
    print("-" * 60)
    for feat in CONSONANT_FEATURES:
        valid_full = merged[["severity_numeric", feat]].dropna()
        valid_filt = good_align[["severity_numeric", feat]].dropna()
        rho_full, _ = stats.spearmanr(valid_full.severity_numeric, valid_full[feat])
        rho_filt, _ = stats.spearmanr(valid_filt.severity_numeric, valid_filt[feat])
        delta = rho_filt - rho_full
        print(f"{feat:<25} {rho_full:>10.4f} {rho_filt:>12.4f} {delta:>+10.4f}")
        results[f"{feat}_filtered"] = {"rho": float(rho_filt), "n": len(valid_filt)}

    with open(OUTPUT_DIR / "exp6_alignment_quality.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'exp6_alignment_quality.json'}")
    return results


# =============================================================================
# Experiment 7: Task-Stratified Boundary Sharpness
# =============================================================================

def experiment_7_task_stratified_boundary(df, inv):
    """Full boundary sharpness analysis by speech content type."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Task-Stratified Boundary Sharpness")
    print("=" * 70)

    # Categorize corpora by dominant speech type
    # Based on our investigation of each corpus's content
    corpus_task = {
        "SAP": "mixed",           # mostly read sentences, some spontaneous
        "COPAS": "read",          # read sentences
        "TORGO": "read",          # read sentences + words
        "UASPEECH": "words",      # single words only
        "UASPEECH_control": "words",
        "Neurovoz": "read",       # listen-and-repeat sentences + proverbs
        "PC-GITA": "read",        # read sentences + monologue
        "MDSC": "read",           # read commands/sentences
        "YouTube_French": "spontaneous",  # spontaneous speech
        "LibriSpeech_English": "read",    # read audiobooks
    }

    df_task = df.copy()
    df_task["speech_type"] = df_task.dataset.map(corpus_task)

    print("\n--- Boundary sharpness correlation by speech type ---")
    print(f"{'Speech type':<15} {'N':>5} {'rho':>8} {'p':>12} {'Direction':>12}")
    print("-" * 55)

    results = {}
    for stype in ["read", "spontaneous", "words", "mixed"]:
        sub = df_task[df_task.speech_type == stype]
        valid = sub[["severity_numeric", "boundary_sharpness"]].dropna()
        if len(valid) < 10:
            continue
        rho, p = stats.spearmanr(valid.severity_numeric, valid.boundary_sharpness)
        direction = "INCREASES" if rho > 0 else "DECREASES"
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{stype:<15} {len(valid):>5} {rho:>8.4f} {p:>12.2e} {direction:>12} {star}")
        results[stype] = {"rho": float(rho), "p": float(p), "n": len(valid), "direction": direction}

    # Per-corpus breakdown
    print("\n--- Per-corpus boundary sharpness ---")
    print(f"{'Corpus':<22} {'Type':<12} {'N':>5} {'rho':>8} {'p':>12} {'Dir':>12}")
    print("-" * 65)

    for ds in sorted(df_task.dataset.unique()):
        sub = df_task[df_task.dataset == ds]
        valid = sub[["severity_numeric", "boundary_sharpness"]].dropna()
        if len(valid) < 5:
            continue
        stype = corpus_task.get(ds, "?")
        rho, p = stats.spearmanr(valid.severity_numeric, valid.boundary_sharpness)
        direction = "+" if rho > 0 else "-"
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{ds:<22} {stype:<12} {len(valid):>5} {rho:>8.4f} {p:>12.2e} {direction:>12} {star}")
        results[f"corpus_{ds}"] = {
            "rho": float(rho), "p": float(p), "n": len(valid),
            "speech_type": stype, "direction": "increases" if rho > 0 else "decreases",
        }

    # Cross-position cosim (same analysis)
    print("\n--- Cross-position cosine similarity by speech type ---")
    print(f"{'Speech type':<15} {'N':>5} {'rho':>8} {'p':>12} {'Direction':>12}")
    print("-" * 55)

    for stype in ["read", "spontaneous", "words", "mixed"]:
        sub = df_task[df_task.speech_type == stype]
        valid = sub[["severity_numeric", "cross_position_cosim"]].dropna()
        if len(valid) < 10:
            continue
        rho, p = stats.spearmanr(valid.severity_numeric, valid.cross_position_cosim)
        direction = "INCREASES" if rho > 0 else "DECREASES"
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{stype:<15} {len(valid):>5} {rho:>8.4f} {p:>12.2e} {direction:>12} {star}")

    # Severity means by speech type
    print("\n--- Boundary sharpness means by severity × speech type ---")
    print(f"{'Type':<15} {'Control':>10} {'Mild':>10} {'Moderate':>10} {'Severe':>10} {'Trend':>8}")
    print("-" * 68)

    for stype in ["read", "spontaneous", "words", "mixed"]:
        sub = df_task[df_task.speech_type == stype]
        row = f"{stype:<15}"
        vals = []
        for sev in ["control", "mild", "moderate", "severe"]:
            s = sub[sub.severity_label == sev]["boundary_sharpness"].dropna()
            if len(s) > 0:
                m = s.mean()
                row += f" {m:>10.4f}"
                vals.append(m)
            else:
                row += f" {'--':>10}"
        if len(vals) >= 3:
            diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            if all(d > 0 for d in diffs):
                row += " UP"
            elif all(d < 0 for d in diffs):
                row += " DOWN"
            else:
                row += " MIXED"
        print(row)

    with open(OUTPUT_DIR / "exp7_task_stratified.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'exp7_task_stratified.json'}")
    return results


def main():
    print("=" * 70)
    print("Track 4: Reviewer Experiments 6 & 7")
    print("=" * 70)

    df, inv = load_data()
    print(f"Loaded {len(df)} speakers")

    exp6 = experiment_6_alignment_quality(df, inv)
    exp7 = experiment_7_task_stratified_boundary(df, inv)

    print("\n" + "=" * 70)
    print("EXPERIMENTS 6 & 7 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
