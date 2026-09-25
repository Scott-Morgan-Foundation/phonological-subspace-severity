#!/usr/bin/env python3
"""
SAP-excluded sensitivity analysis for Paper 2.
LaVonne's concern: SAP contributes 1,233/3,374 speakers (36.5%). DS and Stroke
are exclusively from SAP. Rerun aetiology discrimination without SAP to show
which aetiology claims are robust vs SAP-dependent.
"""

import csv
import sys
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
MASTER = BASE / "results" / "track4" / "track4_master.csv"

SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
MAIN_AETS = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}

CONS_FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
ALL_13 = CONS_FEATS + ["high_dprime", "low_dprime", "back_dprime", "round_dprime",
                        "vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv"]


def safe_float(v):
    if v in ("", "nan", None): return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except: return None


def composite(row):
    vals = [safe_float(row.get(f)) for f in CONS_FEATS]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if len(vals) >= 3 else None


def run_analysis(rows, label):
    print(f"\n{'='*80}")
    print(f"{label} (n={len(rows)})")
    print('='*80)

    # Aetiology composition
    by_aet = defaultdict(int)
    for r in rows:
        if r.get("aetiology") in MAIN_AETS:
            by_aet[r["aetiology"]] += 1
    print(f"\nAetiology composition:")
    for a in MAIN_AETS:
        print(f"  {AET_SHORT[a]:>6}: {by_aet[a]}")

    # Per-feature Kruskal-Wallis
    print(f"\n=== Aetiology Kruskal-Wallis (5 main aetiologies, excluding HC) ===")
    print(f"{'Feature':>22} {'H':>8} {'p':>12} {'eps2':>8} {'effect':>8}")
    print("-" * 60)
    results = []
    for feat in ALL_13:
        groups = defaultdict(list)
        for r in rows:
            aet = r.get("aetiology", "")
            if aet not in ["parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]:
                continue
            v = safe_float(r.get(feat))
            if v is not None:
                groups[aet].append(v)
        valid = {a: g for a, g in groups.items() if len(g) >= 5}
        if len(valid) < 2: continue
        H, p = stats.kruskal(*valid.values())
        N = sum(len(g) for g in valid.values())
        k = len(valid)
        eps2 = max((H - k + 1) / (N - k), 0)
        effect = "Large" if eps2 > 0.14 else "Medium" if eps2 > 0.06 else "Small"
        results.append((feat, H, p, eps2, effect, k))
        print(f"{feat:>22} {H:>8.1f} {p:>12.2e} {eps2:>8.3f} {effect:>8} (k={k})")

    # Composite consonant d-prime across all 6 groups (including HC)
    print(f"\n=== Composite consonant d-prime — 6-group discrimination ===")
    groups = defaultdict(list)
    for r in rows:
        aet = r.get("aetiology", "")
        if aet not in MAIN_AETS: continue
        c = composite(r)
        if c is not None:
            groups[aet].append(c)
    valid = {a: g for a, g in groups.items() if len(g) >= 5}
    if len(valid) >= 2:
        H, p = stats.kruskal(*valid.values())
        N = sum(len(g) for g in valid.values())
        k = len(valid)
        eps2 = max((H - k + 1) / (N - k), 0)
        print(f"H={H:.1f}, p={p:.2e}, eps2={eps2:.3f}, k={k}, N={N}")
        for aet in sorted(valid.keys()):
            print(f"  {AET_SHORT[aet]:>6}: mean={np.mean(valid[aet]):+.3f}, n={len(valid[aet])}")

    # Severity correlation on composite
    print(f"\n=== Severity correlation (composite, labelled speakers only) ===")
    pairs = []
    for r in rows:
        sev = r.get("severity_label", "")
        if sev not in SEV_MAP: continue
        aet = r.get("aetiology", "")
        if aet not in MAIN_AETS: continue
        c = composite(r)
        if c is not None:
            pairs.append((SEV_MAP[sev], c))
    if pairs:
        rho, p = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"n={len(pairs)}, rho={rho:+.4f}, p={p:.2e}")

    return results


def main():
    with open(MASTER, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Total speakers loaded: {len(rows)}")

    full = run_analysis(rows, "FULL DATASET (all 25 datasets)")
    no_sap = [r for r in rows if r["dataset"] != "SAP"]
    excl = run_analysis(no_sap, "SAP EXCLUDED")

    # Comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON: Full vs SAP-excluded")
    print('='*80)
    print(f"{'Feature':>22} {'eps2(full)':>12} {'eps2(no-SAP)':>14} {'delta':>8}")
    print("-" * 60)
    full_dict = {f[0]: f[3] for f in full}
    excl_dict = {f[0]: f[3] for f in excl}
    for feat in ALL_13:
        fv = full_dict.get(feat, float("nan"))
        ev = excl_dict.get(feat, float("nan"))
        delta = ev - fv if not (np.isnan(fv) or np.isnan(ev)) else float("nan")
        print(f"{feat:>22} {fv:>12.3f} {ev:>14.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
