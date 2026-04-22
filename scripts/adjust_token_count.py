#!/usr/bin/env python3
"""
Regress out n_phones from d-prime features to remove token count confound.

For each d-prime feature:
  1. Fit linear regression: d_prime ~ n_phones (on HC speakers only, where the
     confound is cleanest — dysarthric speakers have genuine d-prime reduction)
  2. Compute residuals for ALL speakers: adjusted = observed - beta * n_phones
  3. Add back the grand mean so values stay interpretable

Outputs: track4_master_adjusted.csv with adjusted d-prime columns.
Also prints before/after correlation with n_phones to verify confound removal.
"""

import csv
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
MASTER = BASE / "results" / "track4" / "track4_master.csv"
OUTPUT = BASE / "results" / "track4" / "track4_master_adjusted.csv"

DPRIME_FEATS = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime",
    "high_dprime", "low_dprime", "back_dprime", "round_dprime",
]

# Also adjust vowel_triangle_area (also phone-count dependent)
ADJUST_FEATS = DPRIME_FEATS + ["vowel_triangle_area"]


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except (ValueError, TypeError):
        return None


def main():
    with open(MASTER, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print("=" * 80)
    print("Token count adjustment (regress out n_phones from d-prime features)")
    print("=" * 80)
    print(f"Loaded {len(rows)} speakers")

    # Step 1: Fit regression on ALL speakers (captures overall confound)
    print(f"Fitting regression on all speakers")

    betas = {}
    intercepts = {}

    print(f"\n{'Feature':>20} {'beta':>8} {'r(raw)':>8} {'p':>10} {'Direction':>12}")
    print("-" * 65)

    for feat in ADJUST_FEATS:
        pairs = []
        for r in rows:
            n = safe_float(r.get("n_phones"))
            v = safe_float(r.get(feat))
            if n is not None and v is not None and n > 0:
                pairs.append((n, v))

        if len(pairs) < 20:
            print(f"  {feat}: too few data points ({len(pairs)}), skipping")
            betas[feat] = 0
            intercepts[feat] = 0
            continue

        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])

        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        betas[feat] = slope
        intercepts[feat] = intercept

        direction = "+" if slope > 0 else "-"
        fname = feat.replace("_dprime", "").replace("_", " ")
        print(f"  {fname:>18} {slope:>8.5f} {r_val:>8.3f} {p_val:>10.2e} {direction:>12}")

    # Step 2: Adjust ALL speakers
    print(f"\nAdjusting all {len(rows)} speakers...")

    # Compute grand mean per feature (to add back after residualization)
    grand_means = {}
    for feat in ADJUST_FEATS:
        vals = [safe_float(r.get(feat)) for r in rows]
        vals = [v for v in vals if v is not None]
        grand_means[feat] = np.mean(vals) if vals else 0

    n_adjusted = 0
    for row in rows:
        n_phones = safe_float(row.get("n_phones"))
        if n_phones is None:
            continue

        for feat in ADJUST_FEATS:
            v = safe_float(row.get(feat))
            if v is None:
                continue

            beta = betas[feat]
            if beta == 0:
                continue

            # Residual + grand mean
            adjusted = v - beta * n_phones + grand_means[feat]
            row[feat] = f"{adjusted:.6f}"

        n_adjusted += 1

    print(f"Adjusted {n_adjusted} speakers")

    # Step 3: Verify — correlation with n_phones should be ~0 now
    print(f"\n{'Feature':>20} {'r(before)':>10} {'r(after)':>10} {'reduction':>10}")
    print("-" * 55)

    # Reload original for comparison
    with open(MASTER, "r", encoding="utf-8") as f:
        orig_rows = list(csv.DictReader(f))

    for feat in ADJUST_FEATS:
        # Before
        pairs_before = [(safe_float(r.get("n_phones")), safe_float(r.get(feat)))
                        for r in orig_rows]
        pairs_before = [(n, v) for n, v in pairs_before if n is not None and v is not None]
        if len(pairs_before) < 10:
            continue
        r_before = stats.pearsonr([p[0] for p in pairs_before], [p[1] for p in pairs_before])[0]

        # After
        pairs_after = [(safe_float(r.get("n_phones")), safe_float(r.get(feat)))
                       for r in rows]
        pairs_after = [(n, v) for n, v in pairs_after if n is not None and v is not None]
        r_after = stats.pearsonr([p[0] for p in pairs_after], [p[1] for p in pairs_after])[0]

        reduction = (1 - abs(r_after) / abs(r_before)) * 100 if abs(r_before) > 0.001 else 0
        fname = feat.replace("_dprime", "").replace("_", " ")
        print(f"  {fname:>18} {r_before:>10.3f} {r_after:>10.3f} {reduction:>9.0f}%")

    # Step 4: Write adjusted CSV
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUTPUT}")

    # Step 5: Quick sanity — severity means before and after
    print(f"\nSeverity means (composite consonant d-prime):")
    print(f"  {'Severity':>10} {'before':>8} {'after':>8}")
    for sev in ["control", "mild", "moderate", "severe"]:
        # Before
        bvals = []
        for r in orig_rows:
            if r.get("severity_label") != sev:
                continue
            vs = [safe_float(r.get(f)) for f in DPRIME_FEATS[:5]]
            vs = [v for v in vs if v is not None]
            if vs:
                bvals.append(np.mean(vs))

        # After
        avals = []
        for r in rows:
            if r.get("severity_label") != sev:
                continue
            vs = [safe_float(r.get(f)) for f in DPRIME_FEATS[:5]]
            vs = [v for v in vs if v is not None]
            if vs:
                avals.append(np.mean(vs))

        bm = np.mean(bvals) if bvals else 0
        am = np.mean(avals) if avals else 0
        print(f"  {sev:>10} {bm:>8.3f} {am:>8.3f}")


if __name__ == "__main__":
    main()
