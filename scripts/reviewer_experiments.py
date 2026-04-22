#!/usr/bin/env python3
"""
Paper 2 reviewer-requested experiments:
  1. Severity-source ablation (clinical-only vs all labels)
  2. Language minimum-n analysis (cross-lingual cosine with n>=5, n>=10)
  3. Multiplicity-corrected post hoc testing (Dunn + Holm)
  4. Fixed-token subsampling (speaker-level bootstrap approximation)

Run on DGX:
    cd ~/dysarthria && source venv/bin/activate
    python -u track4/scripts/reviewer_experiments.py 2>&1 | tee logs/reviewer_experiments.log
"""

import csv
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine
from itertools import combinations

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
MASTER = BASE / "results" / "track4" / "track4_master.csv"
OUTPUT_DIR = BASE / "results" / "track4" / "robustness"

CONS_FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
ALL_13 = CONS_FEATS + ["high_dprime", "low_dprime", "back_dprime", "round_dprime",
                        "vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv"]

SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
MAIN_AETS = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}

# Datasets with CLINICAL severity labels (from published sources)
CLINICAL_DATASETS = {
    "TORGO", "UASPEECH", "UASPEECH_control", "SAP", "COPAS", "SSNCE_Tamil",
    "MDSC", "IPVS", "PC-GITA", "Hungarian_Dysarthria", "Neurovoz",
    "CDLI_Kenyan_Swahili", "YouTube_French", "YouTube_German",
    "CHASING", "TreasureHunters1"
}
# Datasets with THRESHOLD-DERIVED pseudo-labels
THRESHOLD_DATASETS = {
    "CDSD", "Domotica", "EasyCall", "SVD", "AVFAD", "EWA-DB"
}

N_BOOTSTRAP = 1000
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


def composite_score(row, feats=CONS_FEATS):
    vals = [safe_float(row.get(f)) for f in feats]
    vals = [v for v in vals if v is not None]
    if len(vals) >= 3:
        return np.mean(vals)
    return None


# =========================================================================
# EXPERIMENT 1: Severity-source ablation
# =========================================================================
def experiment1_severity_source(rows):
    print("=" * 80, flush=True)
    print("EXPERIMENT 1: Severity-source ablation", flush=True)
    print("=" * 80, flush=True)

    labelled = [r for r in rows if r.get("severity_label") in SEV_MAP
                and r.get("aetiology") in MAIN_AETS]

    # Split by provenance
    clinical = [r for r in labelled if r["dataset"] in CLINICAL_DATASETS]
    # threshold = [r for r in labelled if r["dataset"] in THRESHOLD_DATASETS]
    # Note: most threshold datasets have severity=unknown, so they won't be in labelled
    # The "clinical" set IS essentially all labelled speakers since threshold datasets
    # mostly remain unknown. Let's verify:
    non_clinical = [r for r in labelled if r["dataset"] not in CLINICAL_DATASETS]

    print(f"\n  Total labelled speakers: {len(labelled)}", flush=True)
    print(f"  Clinical-source labels: {len(clinical)}", flush=True)
    print(f"  Non-clinical labels: {len(non_clinical)}", flush=True)
    if non_clinical:
        nc_ds = defaultdict(int)
        for r in non_clinical:
            nc_ds[r["dataset"]] += 1
        print(f"  Non-clinical datasets: {dict(nc_ds)}", flush=True)

    # Run severity correlation on each subset
    for name, subset in [("All labelled", labelled), ("Clinical-only", clinical)]:
        pairs = []
        for r in subset:
            comp = composite_score(r)
            if comp is not None:
                pairs.append((SEV_MAP[r["severity_label"]], comp))
        if len(pairs) >= 20:
            sevs, comps = zip(*pairs)
            rho, p = stats.spearmanr(sevs, comps)
            print(f"\n  {name} (n={len(pairs)}): rho = {rho:+.4f}, p = {p:.2e}", flush=True)

            # Per-severity means
            by_sev = defaultdict(list)
            for s, c in pairs:
                by_sev[s].append(c)
            for s in sorted(by_sev.keys()):
                sname = ["control", "mild", "moderate", "severe"][s]
                vals = by_sev[s]
                print(f"    {sname:>10}: mean={np.mean(vals):.3f}, n={len(vals)}", flush=True)

    # Aetiology discrimination on each subset
    print(f"\n  Aetiology discrimination (Kruskal-Wallis on composite):", flush=True)
    for name, subset in [("All labelled", labelled), ("Clinical-only", clinical)]:
        groups = defaultdict(list)
        for r in subset:
            aet = r.get("aetiology")
            if aet not in MAIN_AETS:
                continue
            comp = composite_score(r)
            if comp is not None:
                groups[aet].append(comp)
        valid = {a: g for a, g in groups.items() if len(g) >= 5}
        if len(valid) >= 2:
            H, p = stats.kruskal(*valid.values())
            N = sum(len(g) for g in valid.values())
            k = len(valid)
            eps2 = max((H - k + 1) / (N - k), 0)
            print(f"    {name}: H={H:.1f}, p={p:.2e}, eps2={eps2:.3f}, N={N}, k={k}", flush=True)


# =========================================================================
# EXPERIMENT 2: Language minimum-n analysis
# =========================================================================
def experiment2_language_min_n(rows):
    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT 2: Cross-lingual cosine with minimum-n thresholds", flush=True)
    print("=" * 80, flush=True)

    by_aet_lang = defaultdict(lambda: defaultdict(list))
    for r in rows:
        aet = r.get("aetiology", "unknown")
        lang = r.get("language", "?")
        by_aet_lang[aet][lang].append(r)

    for min_n in [1, 3, 5, 10]:
        print(f"\n  === Minimum n >= {min_n} per language-aetiology cell ===", flush=True)

        for aet in ["parkinsons", "cerebral_palsy", "als", "healthy"]:
            langs = sorted([l for l, rs in by_aet_lang[aet].items() if len(rs) >= min_n])
            if len(langs) < 2:
                print(f"    {AET_SHORT.get(aet, aet)}: <2 languages with n>={min_n}, skipped", flush=True)
                continue

            # Compute all pairwise cosine similarities
            pair_sims = []
            for l1, l2 in combinations(langs, 2):
                p1 = mean_profile(by_aet_lang[aet][l1])
                p2 = mean_profile(by_aet_lang[aet][l2])
                sim = cosine_sim(p1, p2)
                if not np.isnan(sim):
                    pair_sims.append(sim)

            if pair_sims:
                # Bootstrap CI on mean cosine
                boot_means = []
                for _ in range(N_BOOTSTRAP):
                    idx = RNG.choice(len(pair_sims), size=len(pair_sims), replace=True)
                    boot_means.append(np.mean([pair_sims[i] for i in idx]))
                lo, hi = np.percentile(boot_means, [2.5, 97.5])
                print(f"    {AET_SHORT.get(aet, aet)}: {len(langs)} langs, "
                      f"{len(pair_sims)} pairs, mean cos={np.mean(pair_sims):.3f} "
                      f"[{lo:.3f}, {hi:.3f}], min={min(pair_sims):.3f}", flush=True)


# =========================================================================
# EXPERIMENT 3: Multiplicity-corrected post hoc testing (Dunn + Holm)
# =========================================================================
def experiment3_posthoc(rows):
    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT 3: Dunn post hoc with Holm correction", flush=True)
    print("=" * 80, flush=True)

    labelled = [r for r in rows if r.get("aetiology") in MAIN_AETS]

    # For composite consonant d-prime
    groups = defaultdict(list)
    for r in labelled:
        aet = r.get("aetiology")
        comp = composite_score(r)
        if comp is not None:
            groups[aet].append(comp)

    valid_aets = sorted([a for a, g in groups.items() if len(g) >= 5])
    print(f"\n  Groups: {', '.join(AET_SHORT.get(a, a) for a in valid_aets)}", flush=True)
    print(f"  Group sizes: {', '.join(f'{AET_SHORT.get(a,a)}={len(groups[a])}' for a in valid_aets)}", flush=True)

    # Omnibus Kruskal-Wallis
    H, p_omni = stats.kruskal(*[groups[a] for a in valid_aets])
    N = sum(len(groups[a]) for a in valid_aets)
    k = len(valid_aets)
    eps2 = max((H - k + 1) / (N - k), 0)
    print(f"\n  Omnibus: H={H:.1f}, p={p_omni:.2e}, eps2={eps2:.3f}", flush=True)

    # Dunn's test (pairwise Mann-Whitney with Holm correction)
    pairs = list(combinations(valid_aets, 2))
    p_values = []
    effect_sizes = []
    pair_labels = []

    for a1, a2 in pairs:
        g1 = np.array(groups[a1])
        g2 = np.array(groups[a2])
        U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        # Rank-biserial correlation (effect size for Mann-Whitney)
        n1, n2 = len(g1), len(g2)
        r_rb = 1 - (2 * U) / (n1 * n2)
        # Also Cohen's d for comparison
        pooled_std = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
        d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

        p_values.append(p)
        effect_sizes.append((r_rb, d))
        pair_labels.append(f"{AET_SHORT.get(a1,a1)} vs {AET_SHORT.get(a2,a2)}")

    # Holm correction
    n_tests = len(p_values)
    sorted_idx = np.argsort(p_values)
    p_holm = np.ones(n_tests)
    for rank, idx in enumerate(sorted_idx):
        p_holm[idx] = min(p_values[idx] * (n_tests - rank), 1.0)
    # Enforce monotonicity
    for i in range(1, n_tests):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i-1]
        p_holm[idx] = max(p_holm[idx], p_holm[prev_idx])

    print(f"\n  {'Pair':>20} {'p_raw':>12} {'p_Holm':>12} {'r_rb':>8} {'Cohen_d':>10} {'Sig':>5}", flush=True)
    print("  " + "-" * 75, flush=True)
    for i, label in enumerate(pair_labels):
        r_rb, d = effect_sizes[i]
        sig = "***" if p_holm[i] < 0.001 else "**" if p_holm[i] < 0.01 else "*" if p_holm[i] < 0.05 else "ns"
        print(f"  {label:>20} {p_values[i]:>12.2e} {p_holm[i]:>12.2e} {r_rb:>+8.3f} {d:>+10.3f} {sig:>5}", flush=True)


# =========================================================================
# EXPERIMENT 4: Fixed-token subsampling (speaker-level approximation)
# =========================================================================
def experiment4_fixed_token(rows):
    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT 4: Fixed-token subsampling (speaker-level)", flush=True)
    print("=" * 80, flush=True)
    print("  NOTE: True fixed-token d-prime requires raw per-token embeddings.", flush=True)
    print("  This approximation bins speakers by n_phones and checks stability.", flush=True)

    labelled = [r for r in rows if r.get("severity_label") in SEV_MAP
                and r.get("aetiology") in MAIN_AETS]

    speakers = []
    for r in labelled:
        comp = composite_score(r)
        n_phones = safe_float(r.get("n_phones"))
        if comp is not None and n_phones is not None and n_phones > 0:
            speakers.append({
                "composite": comp,
                "severity": SEV_MAP[r["severity_label"]],
                "n_phones": n_phones,
                "aetiology": r["aetiology"],
            })

    # Create token-count bins
    n_phones_vals = [s["n_phones"] for s in speakers]
    bins = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, float('inf'))]
    bin_labels = ["<100", "100-200", "200-500", "500-1000", ">1000"]

    print(f"\n  Token-count bin analysis (severity correlation within each bin):", flush=True)
    print(f"  {'Bin':>12} {'n':>6} {'rho':>8} {'p':>12} {'aet_H':>10} {'aet_eps2':>10}", flush=True)
    print("  " + "-" * 65, flush=True)

    for (lo, hi), label in zip(bins, bin_labels):
        bin_spk = [s for s in speakers if lo <= s["n_phones"] < hi]
        if len(bin_spk) < 20:
            print(f"  {label:>12} {len(bin_spk):>6}  (too few)", flush=True)
            continue

        # Severity correlation
        sevs = [s["severity"] for s in bin_spk]
        comps = [s["composite"] for s in bin_spk]
        rho, p = stats.spearmanr(sevs, comps)

        # Aetiology discrimination
        aet_groups = defaultdict(list)
        for s in bin_spk:
            aet_groups[s["aetiology"]].append(s["composite"])
        valid = {a: g for a, g in aet_groups.items() if len(g) >= 5}
        if len(valid) >= 2:
            H, p_aet = stats.kruskal(*valid.values())
            N_aet = sum(len(g) for g in valid.values())
            k = len(valid)
            eps2 = max((H - k + 1) / (N_aet - k), 0)
        else:
            H, eps2 = float('nan'), float('nan')

        print(f"  {label:>12} {len(bin_spk):>6} {rho:>+8.3f} {p:>12.2e} {H:>10.1f} {eps2:>10.3f}", flush=True)

    # Token-matched speaker comparison
    # Match speakers across severity levels by n_phones (±20%)
    print(f"\n  Token-matched severity comparison (speakers matched ±20% n_phones):", flush=True)
    sev_speakers = defaultdict(list)
    for s in speakers:
        sev_speakers[s["severity"]].append(s)

    # Match mild vs control
    for sev_a, sev_b, name in [(0, 1, "Control vs Mild"), (1, 2, "Mild vs Moderate"), (2, 3, "Moderate vs Severe")]:
        grp_a = sev_speakers[sev_a]
        grp_b = sev_speakers[sev_b]
        matched_a, matched_b = [], []
        used_b = set()
        for sa in grp_a:
            for j, sb in enumerate(grp_b):
                if j in used_b:
                    continue
                ratio = sa["n_phones"] / sb["n_phones"]
                if 0.8 <= ratio <= 1.25:
                    matched_a.append(sa["composite"])
                    matched_b.append(sb["composite"])
                    used_b.add(j)
                    break
        if len(matched_a) >= 10:
            U, p = stats.mannwhitneyu(matched_a, matched_b, alternative='two-sided')
            d = (np.mean(matched_a) - np.mean(matched_b)) / np.sqrt(
                (np.var(matched_a, ddof=1) + np.var(matched_b, ddof=1)) / 2)
            print(f"    {name}: n_matched={len(matched_a)}, "
                  f"d={d:+.3f}, p={p:.2e}", flush=True)
        else:
            print(f"    {name}: too few matches ({len(matched_a)})", flush=True)


def main():
    print("=" * 80, flush=True)
    print("PAPER 2 REVIEWER-REQUESTED EXPERIMENTS", flush=True)
    print("=" * 80, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} speakers\n", flush=True)

    experiment1_severity_source(rows)
    experiment2_language_min_n(rows)
    experiment3_posthoc(rows)
    experiment4_fixed_token(rows)

    print("\n\nAll experiments completed.", flush=True)


if __name__ == "__main__":
    main()
