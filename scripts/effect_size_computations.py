#!/usr/bin/env python3
"""
Review 2: Additional computations needed.
1. Cliff's delta for all Mann-Whitney comparisons
2. Bootstrap CIs with 5000 iterations
3. HKSJ meta-analysis sensitivity
4. Within-corpus CIs for top features
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")

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
    df = df[df.severity_numeric >= 0]
    return df


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    count = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                count += 1
            elif xi < yi:
                count -= 1
    return count / (nx * ny)


def compute_cliffs_delta(df):
    """Cliff's delta for control vs severe, all features."""
    print("=" * 70)
    print("CLIFF'S DELTA (Control vs Severe)")
    print("=" * 70)

    ctrl = df[df.severity_label == "control"]
    sev = df[df.severity_label == "severe"]

    print(f"\n{'Feature':<25} {'Cliff d':>10} {'Magnitude':>12} {'U':>10} {'p':>12} {'n_ctrl':>7} {'n_sev':>7}")
    print("-" * 85)

    results = {}
    for feat in ALL_FEATURES:
        c = ctrl[feat].dropna().values
        s = sev[feat].dropna().values
        if len(c) < 3 or len(s) < 3:
            continue
        delta = cliffs_delta(c, s)
        U, p = stats.mannwhitneyu(c, s, alternative="two-sided")

        # Magnitude interpretation
        ad = abs(delta)
        mag = "large" if ad >= 0.474 else "medium" if ad >= 0.33 else "small" if ad >= 0.147 else "negligible"

        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<25} {delta:>10.4f} {mag:>12} {U:>10.0f} {p:>12.2e} {len(c):>7} {len(s):>7} {star}")

        results[feat] = {"cliffs_delta": float(delta), "magnitude": mag,
                         "U": float(U), "p": float(p), "n_ctrl": len(c), "n_sev": len(s)}

    with open(OUTPUT_DIR / "review2_cliffs_delta.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def bootstrap_5000(df):
    """Bootstrap CIs with 5000 iterations + fixed seed."""
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CIs (5000 iterations, seed=42)")
    print("=" * 70)

    np.random.seed(42)
    N_BOOT = 5000
    results = {}

    print(f"\n{'Feature':<25} {'rho':>8} {'95% CI':>22} {'p':>12} {'n':>5}")
    print("-" * 75)

    for feat in ALL_FEATURES:
        valid = df[["severity_numeric", feat]].dropna()
        if len(valid) < 10:
            continue
        n = len(valid)
        rho, p = stats.spearmanr(valid.severity_numeric, valid[feat])

        boot_rhos = []
        for _ in range(N_BOOT):
            idx = np.random.choice(n, n, replace=True)
            try:
                r, _ = stats.spearmanr(valid.severity_numeric.values[idx], valid[feat].values[idx])
                boot_rhos.append(r)
            except:
                pass

        ci_low = np.percentile(boot_rhos, 2.5)
        ci_high = np.percentile(boot_rhos, 97.5)

        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<25} {rho:>8.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {p:>12.2e} {n:>5} {star}")

        results[feat] = {"rho": float(rho), "p": float(p), "n": int(n),
                         "ci_low": float(ci_low), "ci_high": float(ci_high),
                         "n_bootstrap": N_BOOT, "seed": 42}

    with open(OUTPUT_DIR / "review2_bootstrap_5000.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def hksj_meta_analysis(df):
    """HKSJ (Hartung-Knapp-Sidik-Jonkman) meta-analysis sensitivity."""
    print("\n" + "=" * 70)
    print("META-ANALYSIS: DerSimonian-Laird vs HKSJ Comparison")
    print("=" * 70)
    print("Fisher z-transform: z = atanh(rho), variance = 1/(n-3)")
    print("Back-transform: rho = tanh(z)")

    results = {}

    for feat in CONSONANT_FEATURES:
        corpus_data = []
        for ds in sorted(df.dataset.unique()):
            sub = df[df.dataset == ds]
            valid = sub[["severity_numeric", feat]].dropna()
            if valid.severity_numeric.nunique() < 2 or len(valid) < 5:
                continue
            rho, p = stats.spearmanr(valid.severity_numeric, valid[feat])
            if np.isnan(rho):
                continue
            n = len(valid)
            z = np.arctanh(np.clip(rho, -0.999, 0.999))
            se = 1.0 / np.sqrt(n - 3)
            corpus_data.append({"corpus": ds, "rho": rho, "z": z, "se": se, "n": n})

        if len(corpus_data) < 2:
            continue

        k = len(corpus_data)
        zs = np.array([d["z"] for d in corpus_data])
        ses = np.array([d["se"] for d in corpus_data])
        ws = 1.0 / ses**2

        # --- DerSimonian-Laird ---
        z_fe = np.sum(ws * zs) / np.sum(ws)
        Q = np.sum(ws * (zs - z_fe)**2)
        c = np.sum(ws) - np.sum(ws**2) / np.sum(ws)
        tau2_dl = max(0, (Q - (k - 1)) / c)
        ws_dl = 1.0 / (ses**2 + tau2_dl)
        z_dl = np.sum(ws_dl * zs) / np.sum(ws_dl)
        se_dl = 1.0 / np.sqrt(np.sum(ws_dl))
        p_dl = 2 * stats.norm.sf(abs(z_dl / se_dl))
        rho_dl = np.tanh(z_dl)
        ci_dl = (np.tanh(z_dl - 1.96 * se_dl), np.tanh(z_dl + 1.96 * se_dl))

        # --- HKSJ adjustment ---
        # HKSJ uses t-distribution instead of normal, with adjusted SE
        z_hksj = z_dl  # same point estimate
        # HKSJ variance: q_hksj = (1/k(k-1)) * sum(w_i * (z_i - z_hat)^2)
        q_hksj = np.sum(ws_dl * (zs - z_dl)**2) / (k - 1)
        # Adjust: multiply DL variance by q_hksj
        se_hksj = se_dl * np.sqrt(max(1.0, q_hksj))  # max(1, q) ensures no narrower than DL
        # Use t-distribution with k-1 df
        t_crit = stats.t.ppf(0.975, k - 1)
        p_hksj = 2 * stats.t.sf(abs(z_hksj / se_hksj), k - 1)
        rho_hksj = np.tanh(z_hksj)
        ci_hksj = (np.tanh(z_hksj - t_crit * se_hksj), np.tanh(z_hksj + t_crit * se_hksj))

        # I^2
        I2 = max(0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0

        # Prediction interval (DL)
        pi_se = np.sqrt(se_dl**2 + tau2_dl)
        pi = (np.tanh(z_dl - 1.96 * pi_se), np.tanh(z_dl + 1.96 * pi_se))

        results[feat] = {
            "k": k, "I2": float(I2), "tau2": float(tau2_dl), "Q": float(Q),
            "DL": {"rho": float(rho_dl), "ci": [float(ci_dl[0]), float(ci_dl[1])], "p": float(p_dl)},
            "HKSJ": {"rho": float(rho_hksj), "ci": [float(ci_hksj[0]), float(ci_hksj[1])], "p": float(p_hksj)},
            "prediction_interval": [float(pi[0]), float(pi[1])],
        }

    print(f"\n{'Feature':<20} {'k':>3} {'I2':>6} | {'DL rho':>8} {'DL CI':>22} {'DL p':>10} | {'HKSJ rho':>9} {'HKSJ CI':>22} {'HKSJ p':>10}")
    print("-" * 120)
    for feat in CONSONANT_FEATURES:
        if feat not in results:
            continue
        r = results[feat]
        dl = r["DL"]
        hk = r["HKSJ"]
        dl_sig = "***" if dl["p"] < 0.001 else "**" if dl["p"] < 0.01 else "*" if dl["p"] < 0.05 else ""
        hk_sig = "***" if hk["p"] < 0.001 else "**" if hk["p"] < 0.01 else "*" if hk["p"] < 0.05 else ""
        print(f"{feat:<20} {r['k']:>3} {r['I2']:>5.1f}% | {dl['rho']:>8.4f} [{dl['ci'][0]:>7.4f},{dl['ci'][1]:>7.4f}] {dl['p']:>10.2e}{dl_sig} | {hk['rho']:>9.4f} [{hk['ci'][0]:>7.4f},{hk['ci'][1]:>7.4f}] {hk['p']:>10.2e}{hk_sig}")

    print("\n  Prediction intervals (where true effect of a NEW corpus would lie):")
    for feat in CONSONANT_FEATURES:
        if feat not in results:
            continue
        pi = results[feat]["prediction_interval"]
        print(f"    {feat:<20} [{pi[0]:>7.4f}, {pi[1]:>7.4f}]")

    with open(OUTPUT_DIR / "review2_hksj_meta.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def within_corpus_cis(df):
    """Bootstrap CIs for within-corpus correlations (top 5 features)."""
    print("\n" + "=" * 70)
    print("WITHIN-CORPUS BOOTSTRAP CIs (1000 iters per corpus)")
    print("=" * 70)

    np.random.seed(42)
    results = {}

    print(f"\n{'Corpus':<22} {'Feature':<20} {'rho':>8} {'95% CI':>22} {'n':>5}")
    print("-" * 80)

    for ds in sorted(df.dataset.unique()):
        sub = df[df.dataset == ds]
        if len(sub) < 10:
            continue
        for feat in CONSONANT_FEATURES[:3]:  # top 3 for space
            valid = sub[["severity_numeric", feat]].dropna()
            if valid.severity_numeric.nunique() < 2 or len(valid) < 5:
                continue
            n = len(valid)
            rho, p = stats.spearmanr(valid.severity_numeric, valid[feat])
            if np.isnan(rho):
                continue

            boot_rhos = []
            for _ in range(1000):
                idx = np.random.choice(n, n, replace=True)
                try:
                    r, _ = stats.spearmanr(valid.severity_numeric.values[idx], valid[feat].values[idx])
                    boot_rhos.append(r)
                except:
                    pass
            if boot_rhos:
                ci_low = np.percentile(boot_rhos, 2.5)
                ci_high = np.percentile(boot_rhos, 97.5)
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"{ds:<22} {feat:<20} {rho:>8.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {n:>5} {star}")
                results[f"{ds}_{feat}"] = {"rho": float(rho), "ci_low": float(ci_low),
                                           "ci_high": float(ci_high), "n": n, "p": float(p)}

    with open(OUTPUT_DIR / "review2_within_corpus_cis.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def dataset_accounting(df):
    """Precise dataset accounting: direction estimation vs evaluation."""
    print("\n" + "=" * 70)
    print("DATASET ACCOUNTING")
    print("=" * 70)

    inv = pd.read_csv(RESULTS_DIR / "speaker_inventory.csv")

    print("\nDirection estimation (HC speakers per language):")
    hc = df[df.severity_label == "control"]
    for lang in sorted(hc.language.unique()):
        lang_hc = hc[hc.language == lang]
        ds_counts = lang_hc.groupby("dataset").size()
        total = len(lang_hc)
        print(f"  {lang}: {total} HC speakers — {dict(ds_counts)}")

    print("\nSeverity evaluation (speakers with severity labels):")
    eval_df = df[df.severity_numeric >= 0]
    for ds in sorted(eval_df.dataset.unique()):
        sub = eval_df[eval_df.dataset == ds]
        sevs = sub.severity_label.value_counts().to_dict()
        print(f"  {ds}: {len(sub)} speakers — {sevs}")

    print(f"\nTotal direction estimation: {len(hc)} HC speakers")
    print(f"Total severity evaluation: {len(eval_df)} speakers")
    print(f"  Control: {len(eval_df[eval_df.severity_label=='control'])}")
    print(f"  Mild: {len(eval_df[eval_df.severity_label=='mild'])}")
    print(f"  Moderate: {len(eval_df[eval_df.severity_label=='moderate'])}")
    print(f"  Severe: {len(eval_df[eval_df.severity_label=='severe'])}")


def main():
    print("=" * 70)
    print("Track 4: Review 2 Computations")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} speakers")

    compute_cliffs_delta(df)
    bootstrap_5000(df)
    hksj_meta_analysis(df)
    within_corpus_cis(df)
    dataset_accounting(df)

    print("\n" + "=" * 70)
    print("ALL REVIEW 2 COMPUTATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
