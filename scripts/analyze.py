#!/usr/bin/env python3
"""
Track 4: Full analysis and visualization of phonological subspace results.

Reads track4_results.csv and speaker_inventory.csv, generates:
  1. Summary statistics dashboard
  2. Heatmap matrix (speakers x features)
  3. Faceted box plots (12 features x severity x aetiology)
  4. Radar charts per aetiology (phonological fingerprints)
  5. Longitudinal plots (where available)
  6. Statistical tests (Spearman correlation, Mann-Whitney U)
  7. Aetiology discrimination analysis (kNN/logistic on 12-metric profile)
  8. Cross-lingual validation (train directions in one language, test in another)

Usage:
    python scripts/track4_analyze.py
    python scripts/track4_analyze.py --results-csv results/track4/track4_results.csv

Runs on any machine with matplotlib, seaborn, scipy, sklearn.
"""

import argparse
import csv
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
RESULTS_DIR = BASE_DIR / "results" / "track4"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "track4_results.csv"
INVENTORY_CSV = RESULTS_DIR / "speaker_inventory.csv"

# Feature columns in track4_results.csv
CONSONANT_FEATURES = ["nasal_dprime", "voicing_dprime", "strident_dprime",
                      "sonorant_dprime", "manner_dprime"]
VOWEL_FEATURES = ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]
STRUCTURAL_FEATURES = ["boundary_sharpness", "cross_position_cosim", "vowel_triangle_area"]
ALL_FEATURES = CONSONANT_FEATURES + VOWEL_FEATURES + STRUCTURAL_FEATURES

SEVERITY_ORDER = ["control", "mild", "moderate", "severe"]
SEVERITY_COLORS = {"control": "#2ecc71", "mild": "#f1c40f",
                   "moderate": "#e67e22", "severe": "#e74c3c"}
AETIOLOGY_ORDER = ["PD", "ALS", "CP", "DS", "Stroke", "Mixed", "control"]

# SMF brand colors for main palette
SMF_ORANGE = "#F04E23"
SMF_NAVY = "#0B1F5C"
SMF_BLUE = "#4DB8C8"


def load_data(results_csv: Path, inventory_csv: Path):
    """Load and merge results with inventory."""
    df_results = pd.read_csv(results_csv)
    df_inv = pd.read_csv(inventory_csv)

    # Merge on dataset + speaker_id (results already have most fields)
    # Add any missing inventory columns
    inv_cols = set(df_inv.columns) - set(df_results.columns) - {"dataset", "speaker_id"}
    if inv_cols:
        df = df_results.merge(
            df_inv[["dataset", "speaker_id"] + list(inv_cols)],
            on=["dataset", "speaker_id"], how="left"
        )
    else:
        df = df_results

    # Ensure severity ordering
    df["severity_label"] = pd.Categorical(
        df["severity_label"], categories=SEVERITY_ORDER, ordered=True)

    # Clean aetiology
    df["aetiology"] = df["aetiology"].fillna("unknown").str.upper()
    df.loc[df["is_control"].astype(str).str.lower() == "true", "aetiology"] = "control"

    print(f"Loaded {len(df)} speakers with {len(ALL_FEATURES)} features")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Languages: {sorted(df['language'].unique())}")
    print(f"  Aetiologies: {sorted(df['aetiology'].unique())}")
    print(f"  Severity: {df['severity_label'].value_counts().to_dict()}")

    return df


# =============================================================================
# 1. Summary Statistics Dashboard
# =============================================================================

def plot_summary_dashboard(df: pd.DataFrame):
    """Overview of the dataset: speakers by dataset, aetiology, severity, language."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Track 4: Phonological Subspace Analysis — Dataset Overview",
                 fontsize=16, fontweight="bold", color=SMF_NAVY)

    # 1a: Speakers by dataset
    ax = axes[0, 0]
    ds_counts = df.groupby("dataset").size().sort_values(ascending=True)
    ds_counts.plot(kind="barh", ax=ax, color=SMF_BLUE)
    ax.set_title("Speakers by Dataset")
    ax.set_xlabel("Number of Speakers")

    # 1b: Speakers by aetiology
    ax = axes[0, 1]
    aet_counts = df["aetiology"].value_counts()
    colors = [SMF_ORANGE if a != "control" else "#2ecc71" for a in aet_counts.index]
    aet_counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Speakers by Aetiology")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    # 1c: Speakers by severity
    ax = axes[1, 0]
    sev_counts = df["severity_label"].value_counts().reindex(SEVERITY_ORDER)
    sev_colors = [SEVERITY_COLORS.get(s, "#95a5a6") for s in sev_counts.index]
    sev_counts.plot(kind="bar", ax=ax, color=sev_colors)
    ax.set_title("Speakers by Severity")
    ax.set_ylabel("Count")

    # 1d: Speakers by language
    ax = axes[1, 1]
    lang_counts = df["language"].value_counts().sort_values(ascending=True)
    lang_counts.plot(kind="barh", ax=ax, color=SMF_NAVY)
    ax.set_title("Speakers by Language")
    ax.set_xlabel("Count")

    plt.tight_layout()
    path = FIGURES_DIR / "summary_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 2. Heatmap Matrix
# =============================================================================

def plot_heatmap(df: pd.DataFrame):
    """Heatmap: rows=speakers (grouped by aetiology+severity), columns=12 features."""
    # Sort by aetiology then severity
    df_sorted = df.copy()
    df_sorted["aet_order"] = df_sorted["aetiology"].map(
        {a: i for i, a in enumerate(AETIOLOGY_ORDER)}).fillna(99)
    df_sorted = df_sorted.sort_values(["aet_order", "severity_label"])

    # Get feature matrix
    feature_cols = [c for c in ALL_FEATURES if c in df_sorted.columns]
    if not feature_cols:
        print("  No features to plot heatmap")
        return

    data = df_sorted[feature_cols].values.astype(float)

    # Z-score normalize each column for visualization
    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(data, axis=0)
        col_stds = np.nanstd(data, axis=0)
        col_stds[col_stds < 1e-8] = 1.0
        data_z = (data - col_means) / col_stds

    # Row labels
    row_labels = [f"{r['aetiology'][:3]}-{r['severity_label'][:3]}-{r['speaker_id'][:10]}"
                  for _, r in df_sorted.iterrows()]

    fig_height = max(8, len(row_labels) * 0.15)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    im = ax.imshow(data_z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   interpolation="nearest")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([c.replace("_dprime", "").replace("_", " ") for c in feature_cols],
                       rotation=45, ha="right")

    # Only show every Nth label if too many
    if len(row_labels) > 50:
        step = max(1, len(row_labels) // 50)
        ax.set_yticks(range(0, len(row_labels), step))
        ax.set_yticklabels([row_labels[i] for i in range(0, len(row_labels), step)],
                           fontsize=6)
    else:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)

    ax.set_title("Phonological Feature Heatmap (Z-scored)", fontsize=14,
                 fontweight="bold", color=SMF_NAVY)
    plt.colorbar(im, ax=ax, label="Z-score (blue=healthy, red=degraded)")

    plt.tight_layout()
    path = FIGURES_DIR / "heatmap_features.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 3. Faceted Box Plots
# =============================================================================

def plot_boxplots(df: pd.DataFrame):
    """One figure per feature: severity x aetiology box plots."""
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    for feat in feature_cols:
        vals = df[feat].dropna()
        if len(vals) < 5:
            continue

        # Get unique aetiologies with enough data
        aets = [a for a in df["aetiology"].unique()
                if df[df["aetiology"] == a][feat].dropna().shape[0] >= 3]
        n_aets = len(aets)
        if n_aets == 0:
            continue

        ncols = min(3, n_aets)
        nrows = (n_aets + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                                 squeeze=False)

        fig.suptitle(f"{feat.replace('_', ' ').title()} by Severity & Aetiology",
                     fontsize=14, fontweight="bold", color=SMF_NAVY, y=1.02)

        for idx, aet in enumerate(sorted(aets)):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            subset = df[df["aetiology"] == aet]

            # Filter to severities with data
            sevs = [s for s in SEVERITY_ORDER if s in subset["severity_label"].values]

            if not sevs:
                ax.set_visible(False)
                continue

            box_data = [subset[subset["severity_label"] == s][feat].dropna().values
                        for s in sevs]
            box_data = [d for d in box_data if len(d) > 0]
            sevs_used = [s for s, d in zip(sevs, [subset[subset["severity_label"] == s][feat].dropna()
                                                   for s in sevs]) if len(d) > 0]

            if box_data:
                bp = ax.boxplot(box_data, labels=sevs_used, patch_artist=True)
                for patch, sev in zip(bp["boxes"], sevs_used):
                    patch.set_facecolor(SEVERITY_COLORS.get(sev, "#95a5a6"))
                    patch.set_alpha(0.7)

            ax.set_title(f"{aet} (n={len(subset)})", fontsize=11)
            ax.set_ylabel(feat.replace("_", " "))

        # Hide empty subplots
        for idx in range(len(aets), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        plt.tight_layout()
        path = FIGURES_DIR / f"boxplot_{feat}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {len(feature_cols)} box plot figures")


# =============================================================================
# 4. Radar Charts
# =============================================================================

def plot_radar_charts(df: pd.DataFrame):
    """Radar chart per aetiology showing phonological fingerprint."""
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    if len(feature_cols) < 3:
        print("  Not enough features for radar charts")
        return

    # Compute mean d-prime per severity per aetiology
    aets = sorted([a for a in df["aetiology"].unique() if a != "control"])

    for aet in aets + ["ALL"]:
        if aet == "ALL":
            subset = df[df["aetiology"] != "control"]
            title = "All Aetiologies Combined"
        else:
            subset = df[df["aetiology"] == aet]
            title = f"Aetiology: {aet}"

        if len(subset) < 3:
            continue

        # Also include controls for reference
        ctrl = df[df["aetiology"] == "control"]

        # Compute means per severity
        angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
        angles.append(angles[0])  # close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for sev in SEVERITY_ORDER:
            if sev == "control":
                sev_data = ctrl
            else:
                sev_data = subset[subset["severity_label"] == sev]

            if len(sev_data) < 2:
                continue

            means = []
            for feat in feature_cols:
                vals = sev_data[feat].dropna()
                means.append(vals.mean() if len(vals) > 0 else 0)
            means.append(means[0])  # close

            color = SEVERITY_COLORS.get(sev, "#95a5a6")
            ax.plot(angles, means, "o-", linewidth=2, label=f"{sev} (n={len(sev_data)})",
                    color=color)
            ax.fill(angles, means, alpha=0.1, color=color)

        # Labels
        labels = [c.replace("_dprime", "").replace("_", "\n") for c in feature_cols]
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
        ax.set_title(title, fontsize=13, fontweight="bold", color=SMF_NAVY, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        safe_name = aet.replace(" ", "_").replace("/", "_")
        path = FIGURES_DIR / f"radar_{safe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {len(aets) + 1} radar chart figures")


# =============================================================================
# 5. Longitudinal Plots
# =============================================================================

def plot_longitudinal(df: pd.DataFrame):
    """Plot feature trajectories for speakers with multiple timepoints."""
    # Find speakers that appear multiple times with different severities
    longitudinal_speakers = []

    for _, group in df.groupby(["dataset", "language"]):
        pass

    # Known longitudinal cases (anonymized)
    known_longitudinal = [
        {"dataset": "YouTube_French", "speakers": ["f1_SPEAKER_06"],
         "name": "Speaker A (ALS)", "severities": ["moderate", "severe"]},
        {"dataset": "YouTube_French", "speakers": ["f3_SPEAKER_02", "f4_SPEAKER_01"],
         "name": "Speaker B (ALS)", "severities": ["mild", "moderate"]},
    ]

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    plotted = 0

    for case in known_longitudinal:
        ds = case["dataset"]
        name = case["name"]
        spk_ids = case["speakers"]
        sevs = case["severities"]

        # Find matching rows
        rows = df[(df["dataset"] == ds) & (df["speaker_id"].isin(spk_ids))]
        if len(rows) < 2:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(feature_cols))
        width = 0.35

        for i, (_, row) in enumerate(rows.iterrows()):
            sev = row["severity_label"]
            vals = [row.get(f, float("nan")) for f in feature_cols]
            color = SEVERITY_COLORS.get(sev, "#95a5a6")
            ax.bar(x + i * width, vals, width, label=f"{sev}", color=color, alpha=0.8)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([c.replace("_dprime", "").replace("_", " ") for c in feature_cols],
                           rotation=45, ha="right")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"Longitudinal: {name}", fontsize=13, fontweight="bold", color=SMF_NAVY)
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = FIGURES_DIR / f"longitudinal_{safe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plotted += 1

    print(f"  Saved: {plotted} longitudinal figures")


# =============================================================================
# 6. Statistical Tests
# =============================================================================

def run_statistical_tests(df: pd.DataFrame):
    """Spearman correlation and Mann-Whitney U tests."""
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    results = {"spearman": [], "mann_whitney": []}

    print("\n  SPEARMAN CORRELATION (feature vs severity_numeric):")
    print(f"  {'Feature':<25} {'rho':>8} {'p-value':>10} {'n':>5}")
    print("  " + "-" * 55)

    for feat in feature_cols:
        valid = df[[feat, "severity_numeric"]].dropna()
        valid = valid[valid["severity_numeric"] >= 0]  # exclude unknown
        if len(valid) < 5:
            continue

        rho, p = stats.spearmanr(valid["severity_numeric"], valid[feat])
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat:<25} {rho:>8.4f} {p:>10.4e} {len(valid):>5} {star}")
        results["spearman"].append({
            "feature": feat, "rho": rho, "p_value": p, "n": len(valid)
        })

    # Per-aetiology Spearman
    print("\n  PER-AETIOLOGY SPEARMAN:")
    for aet in sorted(df["aetiology"].unique()):
        if aet == "control":
            continue
        subset = df[df["aetiology"] == aet]
        if len(subset) < 5:
            continue
        print(f"\n  {aet}:")
        for feat in feature_cols:
            valid = subset[[feat, "severity_numeric"]].dropna()
            valid = valid[valid["severity_numeric"] >= 0]
            if len(valid) < 5:
                continue
            rho, p = stats.spearmanr(valid["severity_numeric"], valid[feat])
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if abs(rho) > 0.3 or p < 0.05:
                print(f"    {feat:<25} rho={rho:>7.3f}  p={p:.4e} {star}")

    # Mann-Whitney U: control vs severe
    print("\n  MANN-WHITNEY U (control vs severe):")
    print(f"  {'Feature':<25} {'U':>10} {'p-value':>10} {'ctrl_n':>7} {'sev_n':>7}")
    print("  " + "-" * 65)

    ctrl = df[df["severity_label"] == "control"]
    sev = df[df["severity_label"] == "severe"]
    for feat in feature_cols:
        ctrl_vals = ctrl[feat].dropna()
        sev_vals = sev[feat].dropna()
        if len(ctrl_vals) < 3 or len(sev_vals) < 3:
            continue
        U, p = stats.mannwhitneyu(ctrl_vals, sev_vals, alternative="two-sided")
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat:<25} {U:>10.0f} {p:>10.4e} {len(ctrl_vals):>7} {len(sev_vals):>7} {star}")
        results["mann_whitney"].append({
            "feature": feat, "U": U, "p_value": p,
            "ctrl_n": len(ctrl_vals), "sev_n": len(sev_vals)
        })

    # Save statistical results
    stats_path = RESULTS_DIR / "statistical_tests.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Statistical results saved to {stats_path}")

    return results


# =============================================================================
# 7. Aetiology Discrimination
# =============================================================================

def run_aetiology_discrimination(df: pd.DataFrame):
    """Test if 12-metric profile can distinguish aetiologies without training."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.metrics import classification_report, accuracy_score
    except ImportError:
        print("  sklearn not available, skipping discrimination analysis")
        return

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    # Only dysarthric speakers with known aetiology
    dys = df[(df["aetiology"] != "control") & (df["aetiology"] != "UNKNOWN")].copy()
    dys = dys.dropna(subset=feature_cols)

    if len(dys) < 10:
        print("  Not enough dysarthric speakers for discrimination analysis")
        return

    X = dys[feature_cols].values.astype(float)
    y = dys["aetiology"].values
    groups = dys["dataset"].values  # LOSO = leave-one-dataset-out

    # Handle NaN: impute with column mean
    col_means = np.nanmean(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_means[i]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # kNN (k=5)
    print(f"\n  AETIOLOGY DISCRIMINATION (n={len(dys)}, {len(feature_cols)} features)")
    print(f"  Aetiologies: {sorted(set(y))}")

    # Simple cross-validation (leave-one-dataset-out if enough datasets)
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        logo = LeaveOneGroupOut()
        splits = list(logo.split(X_scaled, y, groups))
    else:
        # Fallback to 5-fold
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=min(5, len(dys)), shuffle=True, random_state=42)
        splits = list(skf.split(X_scaled, y))

    for name, clf_cls in [("kNN (k=5)", lambda: KNeighborsClassifier(n_neighbors=5)),
                           ("Logistic", lambda: LogisticRegression(max_iter=1000, multi_class="multinomial"))]:
        preds = np.empty_like(y)
        for train_idx, test_idx in splits:
            clf = clf_cls()
            clf.fit(X_scaled[train_idx], y[train_idx])
            preds[test_idx] = clf.predict(X_scaled[test_idx])

        acc = accuracy_score(y, preds)
        print(f"\n  {name}: accuracy = {acc:.1%}")
        report = classification_report(y, preds, zero_division=0)
        print(f"  {report}")

    # Feature importance (logistic regression coefficients)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_scaled, y)
    print("\n  FEATURE IMPORTANCE (logistic regression |coef|):")
    for i, feat in enumerate(feature_cols):
        importance = np.mean(np.abs(clf.coef_[:, i]))
        print(f"    {feat:<25} {importance:.4f}")


# =============================================================================
# 8. Cross-Lingual Validation
# =============================================================================

def run_cross_lingual_validation(df: pd.DataFrame):
    """Test generalization: compute d-prime using directions from one language,
    applied to speakers of another language."""
    # This requires the feature directions, which we don't have in the CSV.
    # Instead, we compare d-prime distributions across languages.
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    print("\n  CROSS-LINGUAL COMPARISON (mean d-prime by language x severity):")

    for feat in feature_cols:
        vals = df[[feat, "language", "severity_label"]].dropna()
        if len(vals) < 10:
            continue

        print(f"\n  {feat}:")
        print(f"  {'Language':<8} {'Control':>10} {'Mild':>10} {'Moderate':>10} {'Severe':>10}")
        print("  " + "-" * 50)

        for lang in sorted(vals["language"].unique()):
            lang_data = vals[vals["language"] == lang]
            row = f"  {lang:<8}"
            for sev in SEVERITY_ORDER:
                sev_data = lang_data[lang_data["severity_label"] == sev][feat]
                if len(sev_data) > 0:
                    row += f" {sev_data.mean():>10.3f}"
                else:
                    row += f" {'—':>10}"
            print(row)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Track 4: Analysis & Visualization")
    parser.add_argument("--results-csv", type=str, default=str(RESULTS_CSV))
    parser.add_argument("--inventory-csv", type=str, default=str(INVENTORY_CSV))
    parser.add_argument("--skip-plots", action="store_true", help="Only run statistics")
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    inventory_csv = Path(args.inventory_csv)

    if not results_csv.exists():
        print(f"ERROR: Results CSV not found: {results_csv}")
        print("Run track4_extract_features.py first")
        sys.exit(1)

    print("=" * 80)
    print("Track 4: Phonological Subspace Analysis — Full Analysis")
    print("=" * 80)

    df = load_data(results_csv, inventory_csv)

    if not args.skip_plots:
        print("\n--- Generating visualizations ---")
        print("\n1. Summary Dashboard")
        plot_summary_dashboard(df)

        print("\n2. Feature Heatmap")
        plot_heatmap(df)

        print("\n3. Box Plots")
        plot_boxplots(df)

        print("\n4. Radar Charts")
        plot_radar_charts(df)

        print("\n5. Longitudinal Plots")
        plot_longitudinal(df)

    print("\n--- Statistical Analysis ---")

    print("\n6. Statistical Tests")
    run_statistical_tests(df)

    print("\n7. Aetiology Discrimination")
    run_aetiology_discrimination(df)

    print("\n8. Cross-Lingual Validation")
    run_cross_lingual_validation(df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
