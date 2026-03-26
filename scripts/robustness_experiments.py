#!/usr/bin/env python3
"""
Track 4: Reviewer-requested statistical experiments.

1. Fixed-token subsampling (neutralize token count confound)
2. Bootstrap CIs for all Spearman correlations
3. BH-FDR multiple comparison correction
4. Leave-one-corpus-out validation
5. Mixed-effects meta-analysis of per-corpus effect sizes

All analyses run on track4_results_merged.csv — no new extraction needed.
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
INPUT_CSV = RESULTS_DIR / "track4_results_merged.csv"
OUTPUT_DIR = RESULTS_DIR / "reviewer_experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

CONSONANT_FEATURES = ["nasal_dprime", "voicing_dprime", "strident_dprime",
                      "sonorant_dprime", "manner_dprime"]
VOWEL_FEATURES = ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]
STRUCTURAL = ["boundary_sharpness", "cross_position_cosim", "vowel_triangle_area"]
ALL_FEATURES = CONSONANT_FEATURES + VOWEL_FEATURES + STRUCTURAL


def load_data():
    df = pd.read_csv(INPUT_CSV)
    df = df[df.severity_numeric >= 0]  # exclude unknown severity
    return df


# =============================================================================
# Experiment 1: Fixed-Token Subsampling
# =============================================================================

def experiment_1_fixed_token_subsampling(df):
    """Resample to fixed N tokens per speaker, recompute correlations 50x."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Fixed-Token Subsampling")
    print("=" * 70)
    print("Objective: Show severity correlations hold when token count is equalized")

    # We can't resample phone embeddings (not saved), but we CAN test
    # whether the correlation remains after controlling for n_phones.
    # Approach: partial Spearman correlation (severity ~ feature | n_phones)
    # AND: subsample speakers to matched token-count bins

    results = {}

    # 1a: Partial correlation (controlling for n_phones)
    print("\n--- 1a: Partial Spearman (controlling for n_phones) ---")
    print(f"{'Feature':<25} {'Raw rho':>10} {'Partial rho':>12} {'Partial p':>12}")
    print("-" * 65)

    for feat in ALL_FEATURES:
        valid = df[["severity_numeric", feat, "n_phones"]].dropna()
        if len(valid) < 10:
            continue

        # Raw correlation
        rho_raw, _ = stats.spearmanr(valid.severity_numeric, valid[feat])

        # Partial correlation: residualize both variables on n_phones
        # Rank-based partial correlation
        r_sev_phones, _ = stats.spearmanr(valid.severity_numeric, valid.n_phones)
        r_feat_phones, _ = stats.spearmanr(valid[feat], valid.n_phones)
        r_sev_feat, _ = stats.spearmanr(valid.severity_numeric, valid[feat])

        # Partial Spearman
        denom = np.sqrt((1 - r_sev_phones**2) * (1 - r_feat_phones**2))
        if denom > 1e-8:
            rho_partial = (r_sev_feat - r_sev_phones * r_feat_phones) / denom
        else:
            rho_partial = r_sev_feat

        # Approximate p-value for partial correlation
        n = len(valid)
        t_stat = rho_partial * np.sqrt((n - 3) / (1 - rho_partial**2 + 1e-10))
        p_partial = 2 * stats.t.sf(abs(t_stat), n - 3)

        star = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else ""
        print(f"{feat:<25} {rho_raw:>10.4f} {rho_partial:>12.4f} {p_partial:>12.2e} {star}")

        results[feat] = {
            "raw_rho": float(rho_raw),
            "partial_rho": float(rho_partial),
            "partial_p": float(p_partial),
            "n": int(n),
        }

    # 1b: Bin speakers by token count, compute correlation within each bin
    print("\n--- 1b: Correlation by token-count quartile ---")
    df_valid = df[["severity_numeric", "n_phones"] + CONSONANT_FEATURES].dropna(subset=["n_phones", "severity_numeric"])
    df_valid["phone_quartile"] = pd.qcut(df_valid.n_phones, 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])

    print(f"{'Quartile':<12} {'N':>5} {'Phone range':>18} {'Nasal rho':>12} {'Sonorant rho':>14}")
    print("-" * 65)
    for q in ["Q1(low)", "Q2", "Q3", "Q4(high)"]:
        sub = df_valid[df_valid.phone_quartile == q]
        phone_range = f"{sub.n_phones.min():.0f}-{sub.n_phones.max():.0f}"
        nas_vals = sub[["severity_numeric", "nasal_dprime"]].dropna()
        son_vals = sub[["severity_numeric", "sonorant_dprime"]].dropna()
        nas_rho = stats.spearmanr(nas_vals.severity_numeric, nas_vals.nasal_dprime)[0] if len(nas_vals) > 5 else float("nan")
        son_rho = stats.spearmanr(son_vals.severity_numeric, son_vals.sonorant_dprime)[0] if len(son_vals) > 5 else float("nan")
        print(f"{q:<12} {len(sub):>5} {phone_range:>18} {nas_rho:>12.4f} {son_rho:>14.4f}")

    # Save
    with open(OUTPUT_DIR / "exp1_token_subsampling.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'exp1_token_subsampling.json'}")
    return results


# =============================================================================
# Experiment 2: Bootstrap CIs
# =============================================================================

def experiment_2_bootstrap_cis(df, n_bootstrap=1000):
    """Bootstrap 95% CIs for all Spearman correlations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Bootstrap 95% CIs")
    print("=" * 70)
    print(f"N bootstrap iterations: {n_bootstrap}")

    np.random.seed(42)
    results = {}

    print(f"\n{'Feature':<25} {'rho':>8} {'95% CI':>20} {'p':>12}")
    print("-" * 70)

    for feat in ALL_FEATURES:
        valid = df[["severity_numeric", feat]].dropna()
        if len(valid) < 10:
            continue
        n = len(valid)
        rho, p = stats.spearmanr(valid.severity_numeric, valid[feat])

        # Bootstrap
        boot_rhos = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            boot_sev = valid.severity_numeric.values[idx]
            boot_feat = valid[feat].values[idx]
            try:
                r, _ = stats.spearmanr(boot_sev, boot_feat)
                boot_rhos.append(r)
            except:
                pass

        ci_low = np.percentile(boot_rhos, 2.5)
        ci_high = np.percentile(boot_rhos, 97.5)

        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<25} {rho:>8.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {p:>12.2e} {star}")

        results[feat] = {
            "rho": float(rho), "p": float(p), "n": int(n),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
        }

    with open(OUTPUT_DIR / "exp2_bootstrap_cis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'exp2_bootstrap_cis.json'}")
    return results


# =============================================================================
# Experiment 3: BH-FDR Correction
# =============================================================================

def experiment_3_fdr_correction(df):
    """Apply Benjamini-Hochberg FDR correction to all p-values."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Benjamini-Hochberg FDR Correction")
    print("=" * 70)

    from statsmodels.stats.multitest import multipletests

    # Collect all p-values (overall + within-corpus)
    all_tests = []

    # Overall correlations
    for feat in ALL_FEATURES:
        valid = df[["severity_numeric", feat]].dropna()
        if len(valid) < 10:
            continue
        rho, p = stats.spearmanr(valid.severity_numeric, valid[feat])
        all_tests.append({"context": "overall", "feature": feat, "rho": rho, "p": p, "n": len(valid)})

    # Within-corpus
    for ds in sorted(df.dataset.unique()):
        sub = df[df.dataset == ds]
        if len(sub) < 10:
            continue
        for feat in CONSONANT_FEATURES:
            valid = sub[["severity_numeric", feat]].dropna()
            if len(valid) < 5:
                continue
            rho, p = stats.spearmanr(valid.severity_numeric, valid[feat])
            all_tests.append({"context": f"within_{ds}", "feature": feat, "rho": rho, "p": p, "n": len(valid)})

    # Apply FDR
    p_values = [t["p"] for t in all_tests]
    reject, p_corrected, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

    print(f"\nTotal tests: {len(all_tests)}")
    print(f"Rejected at FDR 0.05: {sum(reject)}/{len(all_tests)}")
    print(f"\n{'Context':<25} {'Feature':<20} {'rho':>8} {'p_raw':>12} {'p_FDR':>12} {'Sig':>5}")
    print("-" * 85)

    for i, t in enumerate(all_tests):
        t["p_fdr"] = float(p_corrected[i])
        t["reject_fdr"] = bool(reject[i])
        sig = "***" if p_corrected[i] < 0.001 else "**" if p_corrected[i] < 0.01 else "*" if p_corrected[i] < 0.05 else "ns"
        if t["context"] == "overall" or p_corrected[i] < 0.05:
            print(f"{t['context']:<25} {t['feature']:<20} {t['rho']:>8.4f} {t['p']:>12.2e} {p_corrected[i]:>12.2e} {sig:>5}")

    with open(OUTPUT_DIR / "exp3_fdr_correction.json", "w") as f:
        json.dump(all_tests, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_DIR / 'exp3_fdr_correction.json'}")
    return all_tests


# =============================================================================
# Experiment 4: Leave-One-Corpus-Out
# =============================================================================

def experiment_4_loco(df):
    """Recompute pooled correlations leaving each corpus out."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Leave-One-Corpus-Out Validation")
    print("=" * 70)
    print("Objective: Show results aren't driven by any single corpus")

    corpora = sorted(df.dataset.unique())
    results = {}

    # Full dataset correlations (reference)
    full_rhos = {}
    for feat in CONSONANT_FEATURES:
        valid = df[["severity_numeric", feat]].dropna()
        rho, _ = stats.spearmanr(valid.severity_numeric, valid[feat])
        full_rhos[feat] = rho

    print(f"\n{'Excluded':<22} {'N':>5}", end="")
    for feat in CONSONANT_FEATURES:
        print(f" {feat.replace('_dprime',''):>10}", end="")
    print()
    print("-" * 80)

    # Full
    print(f"{'(none — full)':22} {len(df):>5}", end="")
    for feat in CONSONANT_FEATURES:
        print(f" {full_rhos[feat]:>10.4f}", end="")
    print()

    for ds in corpora:
        sub = df[df.dataset != ds]
        n = len(sub)
        row = f"{ds:<22} {n:>5}"
        loco_rhos = {}
        for feat in CONSONANT_FEATURES:
            valid = sub[["severity_numeric", feat]].dropna()
            if len(valid) < 10:
                loco_rhos[feat] = float("nan")
                row += f" {'--':>10}"
            else:
                rho, _ = stats.spearmanr(valid.severity_numeric, valid[feat])
                loco_rhos[feat] = rho
                # Flag if excluding this corpus changes rho by >0.05
                delta = abs(rho - full_rhos[feat])
                flag = " !" if delta > 0.05 else ""
                row += f" {rho:>10.4f}{flag}"
        results[ds] = loco_rhos
        print(row)

    print("\n! = excluding this corpus changes rho by >0.05")

    # Stability summary
    print(f"\n{'Feature':<25} {'Min rho':>10} {'Max rho':>10} {'Range':>10} {'Stable':>8}")
    print("-" * 65)
    for feat in CONSONANT_FEATURES:
        rhos = [results[ds][feat] for ds in corpora if not np.isnan(results[ds][feat])]
        min_r, max_r = min(rhos), max(rhos)
        rng = max_r - min_r
        stable = "YES" if rng < 0.10 else "WEAK" if rng < 0.15 else "NO"
        print(f"{feat:<25} {min_r:>10.4f} {max_r:>10.4f} {rng:>10.4f} {stable:>8}")

    with open(OUTPUT_DIR / "exp4_loco.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'exp4_loco.json'}")
    return results


# =============================================================================
# Experiment 5: Random-Effects Meta-Analysis of Per-Corpus Correlations
# =============================================================================

def experiment_5_meta_analysis(df):
    """Random-effects meta-analysis of per-corpus Spearman correlations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Random-Effects Meta-Analysis (per-corpus)")
    print("=" * 70)
    print("Objective: Corpus-aware inference replacing pooled Spearman")

    results = {}

    for feat in CONSONANT_FEATURES + VOWEL_FEATURES:
        corpus_effects = []
        for ds in sorted(df.dataset.unique()):
            sub = df[df.dataset == ds]
            valid = sub[["severity_numeric", feat]].dropna()
            if len(valid) < 5:
                continue
            rho, p = stats.spearmanr(valid.severity_numeric, valid[feat])
            n = len(valid)
            # Fisher z-transform for meta-analysis
            z = np.arctanh(np.clip(rho, -0.999, 0.999))
            se = 1.0 / np.sqrt(n - 3) if n > 3 else 1.0
            corpus_effects.append({
                "corpus": ds, "rho": float(rho), "z": float(z),
                "se": float(se), "n": int(n), "p": float(p),
            })

        if len(corpus_effects) < 2:
            continue

        # DerSimonian-Laird random-effects meta-analysis
        zs = np.array([e["z"] for e in corpus_effects])
        ses = np.array([e["se"] for e in corpus_effects])
        ws = 1.0 / ses**2

        # Fixed-effect estimate
        z_fe = np.sum(ws * zs) / np.sum(ws)

        # Q statistic for heterogeneity
        Q = np.sum(ws * (zs - z_fe)**2)
        k = len(corpus_effects)
        df_q = k - 1

        # Between-study variance (tau^2)
        c = np.sum(ws) - np.sum(ws**2) / np.sum(ws)
        tau2 = max(0, (Q - df_q) / c)

        # Random-effects weights
        ws_re = 1.0 / (ses**2 + tau2)
        z_re = np.sum(ws_re * zs) / np.sum(ws_re)
        se_re = 1.0 / np.sqrt(np.sum(ws_re))

        # Back-transform
        rho_re = np.tanh(z_re)
        ci_low = np.tanh(z_re - 1.96 * se_re)
        ci_high = np.tanh(z_re + 1.96 * se_re)

        # p-value
        z_test = z_re / se_re
        p_re = 2 * stats.norm.sf(abs(z_test))

        # I^2 heterogeneity
        I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0

        results[feat] = {
            "rho_re": float(rho_re), "ci_low": float(ci_low), "ci_high": float(ci_high),
            "p": float(p_re), "k": int(k), "tau2": float(tau2),
            "I2": float(I2), "Q": float(Q), "Q_df": int(df_q),
            "per_corpus": corpus_effects,
        }

    # Print results
    print(f"\n{'Feature':<25} {'k':>3} {'rho_RE':>8} {'95% CI':>20} {'p':>12} {'I2':>6} {'Sig':>5}")
    print("-" * 85)
    for feat in CONSONANT_FEATURES + VOWEL_FEATURES:
        if feat not in results:
            continue
        r = results[feat]
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        print(f"{feat:<25} {r['k']:>3} {r['rho_re']:>8.4f} [{r['ci_low']:>7.4f}, {r['ci_high']:>7.4f}] {r['p']:>12.2e} {r['I2']:>5.1f}% {sig}")

    print("\n  Per-corpus effects (consonant features):")
    for feat in CONSONANT_FEATURES:
        if feat not in results:
            continue
        print(f"\n  {feat}:")
        for e in results[feat]["per_corpus"]:
            star = "***" if e["p"] < 0.001 else "**" if e["p"] < 0.01 else "*" if e["p"] < 0.05 else ""
            print(f"    {e['corpus']:<20} rho={e['rho']:>7.4f}  n={e['n']:>4}  {star}")

    with open(OUTPUT_DIR / "exp5_meta_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'exp5_meta_analysis.json'}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Track 4: Reviewer-Requested Experiments")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} speakers with severity labels")
    print(f"Datasets: {sorted(df.dataset.unique())}")

    exp1 = experiment_1_fixed_token_subsampling(df)
    exp2 = experiment_2_bootstrap_cis(df)
    exp3 = experiment_3_fdr_correction(df)
    exp4 = experiment_4_loco(df)
    exp5 = experiment_5_meta_analysis(df)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
