#!/usr/bin/env python3
"""
Task #1: Aetiology Discrimination Analysis

For each of the 13 active features:
  - Kruskal-Wallis H across aetiologies (omnibus test)
  - Eta-squared (eta2) effect size
  - Pairwise Mann-Whitney U with BH-FDR correction
  - Cliff's delta effect sizes for all pairs
  - Cohen's d for all pairs

Aetiologies included: PD, CP, ALS, DS, Stroke (n≥50 each, except Stroke n=99).
HC reference included for contrast.

Output: tables + summary for Paper 2.
"""

import csv
import os
import numpy as np
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from scipy import stats

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
MASTER = BASE / "results" / "track4" / "track4_master.csv"

FEATS_13 = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime",
    "high_dprime", "low_dprime", "back_dprime", "round_dprime",
    "vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv",
]

FEAT_SHORT = {
    "nasal_dprime": "nasal", "voicing_dprime": "voicing", "sonorant_dprime": "sonorant",
    "strident_dprime": "strident", "manner_dprime": "manner",
    "high_dprime": "high", "low_dprime": "low", "back_dprime": "back", "round_dprime": "round",
    "vowel_triangle_area": "VTA", "speech_rate": "spkrate", "pause_rate": "pause", "vowel_duration_cv": "vdurCV",
}

AETIOLOGIES = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
             "down_syndrome": "DS", "stroke": "Stroke"}


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except (ValueError, TypeError):
        return None


def cliffs_delta(x, y):
    """Cliff's delta effect size."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (nx * ny)


def cohens_d(x, y):
    """Cohen's d (pooled SD)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return (mx - my) / sp if sp > 0 else 0.0


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj = min(p * n / rank, prev)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev = adj
    return adjusted


def main():
    with open(MASTER, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by aetiology
    by_aet = defaultdict(list)
    for r in rows:
        aet = r.get("aetiology", "unknown")
        if aet in AETIOLOGIES:
            by_aet[aet].append(r)

    print("=" * 110)
    print("AETIOLOGY DISCRIMINATION ANALYSIS")
    print("=" * 110)
    print(f"\nAetiologies: {', '.join(AET_SHORT[a] + '=' + str(len(by_aet[a])) for a in AETIOLOGIES)}")

    # =========================================================================
    # TABLE 1: Kruskal-Wallis + eta2 for each feature
    # =========================================================================
    print("\n" + "=" * 110)
    print("TABLE 1: Kruskal-Wallis H-test across 6 groups (HC, PD, CP, ALS, DS, Stroke)")
    print("=" * 110)

    print(f"\n  {'Feature':>12} {'H':>10} {'p':>12} {'eta2':>8} {'Effect':>10} {'HC':>7} {'PD':>7} {'CP':>7} {'ALS':>7} {'DS':>7} {'Str':>7}")
    print("  " + "-" * 100)

    feat_data = {}  # feat -> {aet: [values]}
    kw_results = []

    for feat in FEATS_13:
        fname = FEAT_SHORT[feat]
        groups = {}
        for aet in AETIOLOGIES:
            vals = [safe_float(r.get(feat)) for r in by_aet[aet]]
            vals = [v for v in vals if v is not None]
            groups[aet] = vals
        feat_data[feat] = groups

        # Kruskal-Wallis
        group_lists = [groups[a] for a in AETIOLOGIES if len(groups[a]) >= 5]
        if len(group_lists) < 2:
            continue

        H, p = stats.kruskal(*group_lists)

        # eta2 = (H - k + 1) / (N - k)
        N = sum(len(g) for g in group_lists)
        k = len(group_lists)
        eta_sq = (H - k + 1) / (N - k) if N > k else 0
        eta_sq = max(eta_sq, 0)

        effect = "large" if eta_sq >= 0.14 else "medium" if eta_sq >= 0.06 else "small"

        means = [np.mean(groups[a]) if groups[a] else float('nan') for a in AETIOLOGIES]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        print(f"  {fname:>12} {H:>10.1f} {p:>12.2e}{sig:>3} {eta_sq:>8.3f} {effect:>10}", end="")
        for m in means:
            print(f" {m:>7.2f}" if not np.isnan(m) else f" {'N/A':>7}", end="")
        print()

        kw_results.append((feat, fname, H, p, eta_sq, effect))

    # Sort by eta2
    print("\n  Features ranked by eta2 (effect size):")
    for feat, fname, H, p, eta_sq, effect in sorted(kw_results, key=lambda x: -x[4]):
        bar = "#" * int(eta_sq * 100)
        print(f"    {fname:>12}: eta2={eta_sq:.3f} {bar} ({effect})")

    # =========================================================================
    # TABLE 2: Pairwise comparisons (dysarthric aetiologies only, excluding HC)
    # =========================================================================
    print("\n" + "=" * 110)
    print("TABLE 2: Pairwise Mann-Whitney U (dysarthric aetiologies, top 5 consonant d-primes)")
    print("=" * 110)

    dys_aets = ["parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
    pairs = list(combinations(dys_aets, 2))
    consonant_feats = FEATS_13[:5]

    print(f"\n  {'Pair':>16}", end="")
    for feat in consonant_feats:
        print(f" {FEAT_SHORT[feat]:>10}", end="")
    print(f" {'mean |d|':>10}")
    print("  " + "-" * (16 + 11 * (len(consonant_feats) + 1)))

    all_pvals = []
    pair_results = []

    for a1, a2 in pairs:
        pair_name = f"{AET_SHORT[a1]}-{AET_SHORT[a2]}"
        ds = []
        for feat in consonant_feats:
            x = feat_data[feat].get(a1, [])
            y = feat_data[feat].get(a2, [])
            if len(x) >= 5 and len(y) >= 5:
                _, p = stats.mannwhitneyu(x, y, alternative='two-sided')
                d = cohens_d(x, y)
                all_pvals.append(p)
                ds.append((d, p))
            else:
                ds.append((0, 1))
        pair_results.append((pair_name, ds))

    # BH-FDR correction on all pairwise p-values
    adj_pvals = bh_fdr(all_pvals)
    pidx = 0

    for pair_name, ds_list in pair_results:
        print(f"  {pair_name:>16}", end="")
        abs_ds = []
        for d, p in ds_list:
            adj_p = adj_pvals[pidx]
            pidx += 1
            sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else ""
            print(f" {d:>+7.2f}{sig:>3}", end="")
            abs_ds.append(abs(d))
        mean_d = np.mean(abs_ds)
        print(f" {mean_d:>10.3f}")

    # =========================================================================
    # TABLE 3: HC vs each aetiology (all 13 features)
    # =========================================================================
    print("\n" + "=" * 110)
    print("TABLE 3: HC vs Each Aetiology — Cohen's d (all 13 features)")
    print("=" * 110)

    print(f"\n  {'Feature':>12}", end="")
    for aet in dys_aets:
        print(f" {AET_SHORT[aet]:>9}", end="")
    print()
    print("  " + "-" * (12 + 10 * len(dys_aets)))

    for feat in FEATS_13:
        fname = FEAT_SHORT[feat]
        hc_vals = feat_data[feat].get("healthy", [])
        if len(hc_vals) < 5:
            continue
        print(f"  {fname:>12}", end="")
        for aet in dys_aets:
            vals = feat_data[feat].get(aet, [])
            if len(vals) >= 5:
                d = cohens_d(hc_vals, vals)
                print(f" {d:>+9.2f}", end="")
            else:
                print(f" {'N/A':>9}", end="")
        print()

    # =========================================================================
    # TABLE 4: Cliff's delta for hardest pairs
    # =========================================================================
    print("\n" + "=" * 110)
    print("TABLE 4: Cliff's Delta — Hardest Aetiology Pairs (composite consonant d-prime)")
    print("=" * 110)

    # Composite score per speaker
    comp_by_aet = {}
    for aet in AETIOLOGIES:
        scores = []
        for r in by_aet[aet]:
            vals = [safe_float(r.get(f)) for f in FEATS_13[:5]]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 3:
                scores.append(np.mean(vals))
        comp_by_aet[aet] = scores

    print(f"\n  {'Pair':>16} {'n1':>5} {'n2':>5} {'d_cliff':>8} {'|d|':>6} {'Interpretation':>15} {'U':>12} {'p_adj':>10}")
    print("  " + "-" * 80)

    pair_cliff = []
    cliff_pvals = []
    for a1, a2 in combinations(AETIOLOGIES, 2):
        x, y = comp_by_aet.get(a1, []), comp_by_aet.get(a2, [])
        if len(x) >= 5 and len(y) >= 5:
            cd = cliffs_delta(x, y)
            U, p = stats.mannwhitneyu(x, y, alternative='two-sided')
            cliff_pvals.append(p)
            pair_cliff.append((a1, a2, len(x), len(y), cd, U, p))

    adj_cliff = bh_fdr(cliff_pvals)

    for i, (a1, a2, n1, n2, cd, U, p) in enumerate(sorted(pair_cliff, key=lambda x: abs(x[4]))):
        interp = "negligible" if abs(cd) < 0.147 else "small" if abs(cd) < 0.33 else "medium" if abs(cd) < 0.474 else "large"
        sig = "***" if adj_cliff[i] < 0.001 else "**" if adj_cliff[i] < 0.01 else "*" if adj_cliff[i] < 0.05 else ""
        pair_name = f"{AET_SHORT[a1]}-{AET_SHORT[a2]}"
        print(f"  {pair_name:>16} {n1:>5} {n2:>5} {cd:>+8.3f} {abs(cd):>6.3f} {interp:>15} {U:>12.0f} {adj_cliff[i]:>9.2e}{sig}")

    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 110)
    print("KEY FINDINGS")
    print("=" * 110)

    # Most discriminative features
    top3 = sorted(kw_results, key=lambda x: -x[4])[:3]
    print(f"\n  Most discriminative features (by eta2):")
    for feat, fname, H, p, eta_sq, effect in top3:
        print(f"    {fname}: eta2={eta_sq:.3f} ({effect})")

    # Hardest pair
    hardest = min(pair_cliff, key=lambda x: abs(x[4]))
    print(f"\n  Hardest aetiology pair: {AET_SHORT[hardest[0]]}-{AET_SHORT[hardest[1]]} "
          f"(Cliff's d={hardest[4]:+.3f}, n={hardest[2]}+{hardest[3]})")

    # Easiest pair
    easiest = max(pair_cliff, key=lambda x: abs(x[4]))
    print(f"  Easiest aetiology pair: {AET_SHORT[easiest[0]]}-{AET_SHORT[easiest[1]]} "
          f"(Cliff's d={easiest[4]:+.3f}, n={easiest[2]}+{easiest[3]})")


if __name__ == "__main__":
    main()
