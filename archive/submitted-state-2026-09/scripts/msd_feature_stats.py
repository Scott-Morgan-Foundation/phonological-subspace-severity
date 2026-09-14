#!/usr/bin/env python3
"""Per-dataset feature significance for the MSD bundle.

Inputs:
  - track4/results/track4_master_msd.csv  (415 speakers, 3 OneVoice-MSD corpora)

For each (dataset, feature) we compute:
  - Spearman rho, p          (severity_numeric, excluding "unknown")
  - Mann-Whitney U, p        (HC vs all-PD, regardless of severity)
  - Cliff's delta            (HC vs all-PD effect size)
  - BH-FDR adjusted p        (rho_p, per-dataset, across the feature set)
  - n_with_severity / n_total

Features tested:
  Segmental (12):  back_dprime, boundary_sharpness, cross_position_cosim,
                   high_dprime, low_dprime, manner_dprime, nasal_dprime,
                   round_dprime, sonorant_dprime, strident_dprime,
                   voicing_dprime, vowel_triangle_area
  Prosodic (5):    speech_rate, pause_rate, vowel_duration_cv,
                   f0_std_semitones, f0_range_semitones
  Voice qual (5):  jitter_local, shimmer_local, hnr_mean, f0_cv, tremor_ratio

Counts (n_phones, n_utterances, total_phones, f0_mean_hz) are reported but
not significance-tested (token-count proxies / sex-confounded).

Output: track4/results/msd_feature_stats.csv  +  console summary table.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "results" / "track4_master_msd.csv"
OUT_CSV = ROOT / "results" / "msd_feature_stats.csv"

FEATURES = [
    # segmental (sign expectation: degradation, so rho with severity should be NEGATIVE
    # for d-prime features and VTA; boundary_sharpness can flip sign with task type)
    "back_dprime", "boundary_sharpness", "cross_position_cosim",
    "high_dprime", "low_dprime", "manner_dprime", "nasal_dprime",
    "round_dprime", "sonorant_dprime", "strident_dprime",
    "voicing_dprime", "vowel_triangle_area",
    # prosodic
    "speech_rate", "pause_rate", "vowel_duration_cv",
    "f0_std_semitones", "f0_range_semitones",
    # voice quality
    "jitter_local", "shimmer_local", "hnr_mean", "f0_cv", "tremor_ratio",
]

SEV_NUM = {"control": 0, "mild": 1, "moderate": 2, "severe": 3, "unknown": -1}


def parse_float(x):
    if x in ("", None, "nan", "NaN", "None"):
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_master():
    rows = []
    with open(MASTER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def cliffs_delta(x: List[float], y: List[float]) -> float:
    """Cliff's delta: P(x>y) - P(x<y). Equivalent to (2*U - n1*n2)/(n1*n2) but
    computed directly here (small samples). Range [-1, +1]."""
    if not x or not y:
        return float("nan")
    gt = lt = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                gt += 1
            elif xi < yi:
                lt += 1
    n = len(x) * len(y)
    return (gt - lt) / n


def bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR adjustment. Keeps NaN p-values as NaN."""
    arr = np.array(pvals, dtype=float)
    valid = ~np.isnan(arr)
    out = np.full_like(arr, np.nan, dtype=float)
    if not valid.any():
        return out.tolist()
    p = arr[valid]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the right
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out_valid = np.empty_like(adj)
    out_valid[order] = adj
    out[valid] = out_valid
    return out.tolist()


def per_dataset_stats(rows: List[dict], dataset: str) -> List[Dict]:
    """Compute feature stats for one dataset. Returns list of result dicts."""
    ds_rows = [r for r in rows if r["dataset"] == dataset]
    n_total = len(ds_rows)

    # Severity-numeric values (skip 'unknown' / -1)
    sev = []
    for r in ds_rows:
        try:
            v = int(r.get("severity_numeric", "-1"))
        except (TypeError, ValueError):
            v = -1
        sev.append(v)

    # Aetiology-binary: HC vs PD
    is_pd = [r["aetiology"] == "parkinsons" for r in ds_rows]
    is_hc = [r["aetiology"] == "healthy" for r in ds_rows]
    n_pd = sum(is_pd)
    n_hc = sum(is_hc)

    results = []
    for feat in FEATURES:
        vals = [parse_float(r.get(feat, "")) for r in ds_rows]

        # Severity correlation: drop unknown (sev=-1) and NaN feature values
        s_pairs = [(v, s) for v, s in zip(vals, sev)
                   if s >= 0 and not np.isnan(v)]
        n_sev = len(s_pairs)
        if n_sev >= 5:
            xs = [p[0] for p in s_pairs]
            ys = [p[1] for p in s_pairs]
            rho, p_rho = spearmanr(xs, ys)
        else:
            rho, p_rho = float("nan"), float("nan")

        # HC vs PD Mann-Whitney
        hc_vals = [v for v, h in zip(vals, is_hc) if h and not np.isnan(v)]
        pd_vals = [v for v, p in zip(vals, is_pd) if p and not np.isnan(v)]
        if len(hc_vals) >= 3 and len(pd_vals) >= 3:
            u, p_u = mannwhitneyu(hc_vals, pd_vals, alternative="two-sided")
            cd = cliffs_delta(hc_vals, pd_vals)
        else:
            u, p_u, cd = float("nan"), float("nan"), float("nan")

        results.append({
            "dataset": dataset,
            "feature": feat,
            "n_total": n_total,
            "n_pd": n_pd,
            "n_hc": n_hc,
            "n_sev_correlation": n_sev,
            "spearman_rho": rho,
            "spearman_p": p_rho,
            "spearman_p_fdr": float("nan"),  # filled below
            "mwu_p": p_u,
            "cliffs_delta_hc_vs_pd": cd,
        })

    # BH-FDR within this dataset
    p_rhos = [r["spearman_p"] for r in results]
    fdr = bh_fdr(p_rhos)
    for r, q in zip(results, fdr):
        r["spearman_p_fdr"] = q

    return results


def fmt(x, prec=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "    -- "
    return f"{x:>{prec+4}.{prec}f}"


def print_table(rows: List[dict]):
    print(f"\n{'feature':<22} {'n_sev':>6} {'rho':>8} {'p':>10} {'q(FDR)':>10} "
          f"{'mwu_p':>10} {'CD':>8} {'sig':>5}")
    print("-" * 90)
    for r in rows:
        sig_marks = []
        q = r["spearman_p_fdr"]
        if not (isinstance(q, float) and np.isnan(q)):
            if q < 0.001:
                sig_marks.append("***")
            elif q < 0.01:
                sig_marks.append("**")
            elif q < 0.05:
                sig_marks.append("*")
        sig = "".join(sig_marks) if sig_marks else "ns"
        print(f"{r['feature']:<22} {r['n_sev_correlation']:>6} "
              f"{fmt(r['spearman_rho'])} {fmt(r['spearman_p'])} {fmt(r['spearman_p_fdr'])} "
              f"{fmt(r['mwu_p'])} {fmt(r['cliffs_delta_hc_vs_pd'])} {sig:>5}")


def main():
    if not MASTER.exists():
        print(f"ERROR: master not found at {MASTER}")
        sys.exit(1)

    rows = load_master()
    datasets = sorted({r["dataset"] for r in rows})

    all_results = []
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        n_pd = sum(1 for r in ds_rows if r["aetiology"] == "parkinsons")
        n_hc = sum(1 for r in ds_rows if r["aetiology"] == "healthy")
        n_unk_sev = sum(1 for r in ds_rows
                        if r.get("severity_label") in ("unknown", "", None))
        print(f"\n{'='*90}\n  {ds}  (n={len(ds_rows)}, PD={n_pd}, HC={n_hc}, "
              f"unknown_sev={n_unk_sev})\n{'='*90}")
        ds_results = per_dataset_stats(rows, ds)
        print_table(ds_results)
        all_results.extend(ds_results)

    # Write combined CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        for r in all_results:
            row = dict(r)
            for k, v in row.items():
                if isinstance(v, float) and np.isnan(v):
                    row[k] = ""
            w.writerow(row)
    print(f"\nWrote {len(all_results)} rows -> {OUT_CSV}")

    # Cross-dataset summary: which features survive FDR<0.05 in how many datasets?
    print(f"\n{'='*90}\n  Cross-dataset summary: features with q < 0.05 (BH-FDR)\n{'='*90}")
    by_feat = defaultdict(list)
    for r in all_results:
        by_feat[r["feature"]].append((r["dataset"], r["spearman_p_fdr"], r["spearman_rho"]))
    print(f"\n{'feature':<22} {'datasets sig (rho)':<60}")
    print("-" * 80)
    for feat in FEATURES:
        sigs = []
        for ds, q, rho in by_feat[feat]:
            if not (isinstance(q, float) and np.isnan(q)) and q < 0.05:
                short_ds = ds.replace("_OneVoice-MSD26", "").replace("PC-GITA", "PCG")
                sigs.append(f"{short_ds}({rho:+.2f})")
        sigs_str = ", ".join(sigs) if sigs else "—"
        print(f"{feat:<22} {sigs_str:<60}")


if __name__ == "__main__":
    main()
