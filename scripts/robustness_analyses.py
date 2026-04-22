#!/usr/bin/env python3
"""
Paper 2 robustness analyses (reviewer response):
  1. Fixed-token bootstrap d-prime estimator
  2. Bootstrap CIs for cross-lingual cosine similarities
  3. Leave-one-dataset-out stability

Run on DGX:
    cd ~/dysarthria && source venv/bin/activate
    python -u track4/scripts/robustness_analyses.py > logs/robustness_analyses.log 2>&1
"""

import csv
import os
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
MASTER = BASE / "results" / "track4" / "track4_master.csv"
OUTPUT_DIR = BASE / "results" / "track4" / "robustness"

CONS_FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
ALL_FEATS = CONS_FEATS + ["high_dprime", "low_dprime", "back_dprime", "round_dprime",
                           "vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv"]

SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
MAIN_AETS = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}

N_BOOTSTRAP = 1000
FIXED_TOKEN_N = 50  # tokens per class
RNG = np.random.RandomState(42)


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except:
        return None


def load_data():
    with open(MASTER, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cosine_sim(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    return 1 - cosine(a[mask], b[mask])


def mean_profile(subset, feats=CONS_FEATS):
    means = []
    for f in feats:
        vals = [safe_float(r.get(f)) for r in subset]
        vals = [v for v in vals if v is not None]
        means.append(np.mean(vals) if vals else np.nan)
    return np.array(means)


# =========================================================================
# ANALYSIS 1: Fixed-token bootstrap d-prime
# =========================================================================
def analysis1_fixed_token_dprime(rows):
    """For each speaker, resample FIXED_TOKEN_N tokens per class and compute d-prime.

    Since we don't have raw per-token embeddings here, we simulate by:
    - Using the existing d-prime as point estimate
    - Computing bootstrap CI over speakers (resampling speakers, not tokens)
    - Checking if severity correlation holds with subsampled speaker sets

    NOTE: True fixed-token d-prime requires raw embeddings. This analysis
    resamples speakers stratified by n_phones quartile to approximate
    controlling for token count.
    """
    print("=" * 80, flush=True)
    print("ANALYSIS 1: Token-count-stratified bootstrap severity correlation", flush=True)
    print("=" * 80, flush=True)

    labelled = [r for r in rows if r.get("severity_label") in SEV_MAP
                and r.get("aetiology") in MAIN_AETS]

    # Get composite d-prime and n_phones
    speakers = []
    for r in labelled:
        vals = [safe_float(r.get(f)) for f in CONS_FEATS]
        vals_clean = [v for v in vals if v is not None]
        n_phones = safe_float(r.get("n_phones"))
        if len(vals_clean) >= 3 and n_phones is not None and n_phones > 0:
            speakers.append({
                "composite": np.mean(vals_clean),
                "severity": SEV_MAP[r["severity_label"]],
                "n_phones": n_phones,
                "dataset": r["dataset"],
            })

    print(f"  Speakers with labels + features: {len(speakers)}", flush=True)

    # Stratify by n_phones quartile
    n_phones_vals = [s["n_phones"] for s in speakers]
    quartiles = np.percentile(n_phones_vals, [25, 50, 75])

    def get_quartile(n):
        if n <= quartiles[0]: return 0
        elif n <= quartiles[1]: return 1
        elif n <= quartiles[2]: return 2
        else: return 3

    for s in speakers:
        s["quartile"] = get_quartile(s["n_phones"])

    # Bootstrap: resample speakers within each quartile, compute rho
    bootstrap_rhos = []
    for b in range(N_BOOTSTRAP):
        # Stratified resample
        resampled = []
        for q in range(4):
            q_speakers = [s for s in speakers if s["quartile"] == q]
            if q_speakers:
                idx = RNG.choice(len(q_speakers), size=len(q_speakers), replace=True)
                resampled.extend([q_speakers[i] for i in idx])

        sevs = [s["severity"] for s in resampled]
        comps = [s["composite"] for s in resampled]
        rho, _ = stats.spearmanr(sevs, comps)
        bootstrap_rhos.append(rho)

    rhos = np.array(bootstrap_rhos)
    ci_lo, ci_hi = np.percentile(rhos, [2.5, 97.5])
    mean_rho = np.mean(rhos)

    print(f"\n  Stratified bootstrap severity correlation (n={N_BOOTSTRAP}):", flush=True)
    print(f"    Mean rho: {mean_rho:.4f}", flush=True)
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    # Also compute per-quartile rho
    print(f"\n  Per-quartile severity correlation:", flush=True)
    for q in range(4):
        q_spk = [s for s in speakers if s["quartile"] == q]
        if len(q_spk) >= 20:
            rho, p = stats.spearmanr([s["severity"] for s in q_spk],
                                      [s["composite"] for s in q_spk])
            n_range = (min(s["n_phones"] for s in q_spk), max(s["n_phones"] for s in q_spk))
            print(f"    Q{q+1} (n_phones {n_range[0]:.0f}-{n_range[1]:.0f}, "
                  f"n={len(q_spk)}): rho = {rho:+.3f}, p = {p:.2e}", flush=True)

    # Per-feature bootstrap
    print(f"\n  Per-feature bootstrap CIs:", flush=True)
    for feat in CONS_FEATS:
        feat_rhos = []
        feat_speakers = [(SEV_MAP[r["severity_label"]], safe_float(r.get(feat)))
                         for r in labelled if safe_float(r.get(feat)) is not None]
        for b in range(N_BOOTSTRAP):
            idx = RNG.choice(len(feat_speakers), size=len(feat_speakers), replace=True)
            sample = [feat_speakers[i] for i in idx]
            rho, _ = stats.spearmanr([s[0] for s in sample], [s[1] for s in sample])
            feat_rhos.append(rho)
        lo, hi = np.percentile(feat_rhos, [2.5, 97.5])
        fname = feat.replace("_dprime", "")
        print(f"    {fname:>12}: rho = {np.mean(feat_rhos):+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)

    return {"mean_rho": mean_rho, "ci_lo": ci_lo, "ci_hi": ci_hi}


# =========================================================================
# ANALYSIS 2: Bootstrap CIs for cross-lingual cosine similarity
# =========================================================================
def analysis2_bootstrap_cosine(rows):
    print("\n" + "=" * 80, flush=True)
    print("ANALYSIS 2: Bootstrap CIs for cross-lingual cosine similarity", flush=True)
    print("=" * 80, flush=True)

    by_aet_lang = defaultdict(lambda: defaultdict(list))
    for r in rows:
        aet = r.get("aetiology", "unknown")
        lang = r.get("language", "?")
        by_aet_lang[aet][lang].append(r)

    for aet in ["parkinsons", "cerebral_palsy", "als", "healthy"]:
        langs = sorted([l for l, rs in by_aet_lang[aet].items() if len(rs) >= 3])
        if len(langs) < 2:
            continue

        short = AET_SHORT.get(aet, aet)
        print(f"\n  {short} ({len(langs)} languages):", flush=True)

        # Compute observed cosine similarities (all pairs)
        from itertools import combinations
        pair_results = {}
        for l1, l2 in combinations(langs, 2):
            p1 = mean_profile(by_aet_lang[aet][l1])
            p2 = mean_profile(by_aet_lang[aet][l2])
            observed = cosine_sim(p1, p2)

            # Bootstrap: resample speakers within each language
            boot_sims = []
            for b in range(N_BOOTSTRAP):
                rs1 = by_aet_lang[aet][l1]
                rs2 = by_aet_lang[aet][l2]
                idx1 = RNG.choice(len(rs1), size=len(rs1), replace=True)
                idx2 = RNG.choice(len(rs2), size=len(rs2), replace=True)
                bp1 = mean_profile([rs1[i] for i in idx1])
                bp2 = mean_profile([rs2[i] for i in idx2])
                boot_sims.append(cosine_sim(bp1, bp2))

            boot_sims = [s for s in boot_sims if not np.isnan(s)]
            if boot_sims:
                lo, hi = np.percentile(boot_sims, [2.5, 97.5])
                pair_results[(l1, l2)] = {"obs": observed, "lo": lo, "hi": hi}

        # Summary
        if pair_results:
            all_obs = [v["obs"] for v in pair_results.values()]
            all_lo = [v["lo"] for v in pair_results.values()]
            print(f"    Mean cosine: {np.mean(all_obs):.3f} "
                  f"[{np.mean(all_lo):.3f}, {np.mean([v['hi'] for v in pair_results.values()]):.3f}]",
                  flush=True)
            print(f"    Min pair CI lower bound: {min(all_lo):.3f}", flush=True)
            # Show worst pair
            worst = min(pair_results.items(), key=lambda x: x[1]["lo"])
            print(f"    Tightest pair: {worst[0][0]}-{worst[0][1]}: "
                  f"{worst[1]['obs']:.3f} [{worst[1]['lo']:.3f}, {worst[1]['hi']:.3f}]", flush=True)


# =========================================================================
# ANALYSIS 3: Leave-one-dataset-out stability
# =========================================================================
def analysis3_lodo(rows):
    print("\n" + "=" * 80, flush=True)
    print("ANALYSIS 3: Leave-one-dataset-out stability", flush=True)
    print("=" * 80, flush=True)

    datasets = sorted(set(r["dataset"] for r in rows))
    labelled = [r for r in rows if r.get("severity_label") in SEV_MAP
                and r.get("aetiology") in MAIN_AETS]

    # Full-data severity correlation (composite)
    full_pairs = []
    for r in labelled:
        vals = [safe_float(r.get(f)) for f in CONS_FEATS]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 3:
            full_pairs.append((SEV_MAP[r["severity_label"]], np.mean(vals)))
    full_rho, _ = stats.spearmanr([p[0] for p in full_pairs], [p[1] for p in full_pairs])
    print(f"  Full data: rho = {full_rho:+.3f} (n={len(full_pairs)})", flush=True)

    # Full-data aetiology eta-squared (composite)
    def compute_eta2(subset):
        groups = defaultdict(list)
        for r in subset:
            aet = r.get("aetiology")
            if aet not in MAIN_AETS:
                continue
            vals = [safe_float(r.get(f)) for f in CONS_FEATS]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 3:
                groups[aet].append(np.mean(vals))
        valid = {a: g for a, g in groups.items() if len(g) >= 5}
        if len(valid) < 2:
            return np.nan
        H, p = stats.kruskal(*valid.values())
        N = sum(len(g) for g in valid.values())
        k = len(valid)
        return max((H - k + 1) / (N - k), 0)

    full_eta2 = compute_eta2(labelled)
    print(f"  Full data: eta2 = {full_eta2:.3f}", flush=True)

    # LODO
    print(f"\n  {'Dataset dropped':>25} {'n_drop':>7} {'rho':>8} {'eta2':>8} {'stable?':>8}", flush=True)
    print("  " + "-" * 65, flush=True)

    lodo_rhos = []
    lodo_etas = []
    for ds in datasets:
        subset = [r for r in labelled if r["dataset"] != ds]
        n_drop = sum(1 for r in labelled if r["dataset"] == ds)
        if n_drop == 0:
            continue

        # Severity correlation
        pairs = []
        for r in subset:
            vals = [safe_float(r.get(f)) for f in CONS_FEATS]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 3:
                pairs.append((SEV_MAP[r["severity_label"]], np.mean(vals)))
        if len(pairs) >= 20:
            rho, _ = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        else:
            rho = np.nan

        # Eta-squared
        eta2 = compute_eta2(subset)

        stable = "YES" if (not np.isnan(rho) and rho < -0.3 and
                          not np.isnan(eta2) and eta2 > 0.1) else "CHECK"

        print(f"  {ds:>25} {n_drop:>7} {rho:>+8.3f} {eta2:>8.3f} {stable:>8}", flush=True)
        if not np.isnan(rho):
            lodo_rhos.append(rho)
        if not np.isnan(eta2):
            lodo_etas.append(eta2)

    print(f"\n  LODO severity rho: range [{min(lodo_rhos):+.3f}, {max(lodo_rhos):+.3f}], "
          f"mean {np.mean(lodo_rhos):+.3f}", flush=True)
    print(f"  LODO eta2: range [{min(lodo_etas):.3f}, {max(lodo_etas):.3f}], "
          f"mean {np.mean(lodo_etas):.3f}", flush=True)
    print(f"  All folds stable: {all(r < -0.3 for r in lodo_rhos)}", flush=True)


def main():
    print("=" * 80, flush=True)
    print("PAPER 2 ROBUSTNESS ANALYSES", flush=True)
    print("=" * 80, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} speakers\n", flush=True)

    t0 = time.time()

    analysis1_fixed_token_dprime(rows)
    analysis2_bootstrap_cosine(rows)
    analysis3_lodo(rows)

    elapsed = time.time() - t0
    print(f"\n\nAll analyses completed in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
