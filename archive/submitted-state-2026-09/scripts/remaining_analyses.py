#!/usr/bin/env python3
"""
Tasks #3-#6: CP across languages, severity correlation, feature importance, HC universals.
"""

import csv
import os
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from scipy import stats

BASE = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
MASTER = BASE / "results" / "track4" / "track4_master.csv"

FEATS_9 = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime",
    "high_dprime", "low_dprime", "back_dprime", "round_dprime",
]
FEATS_13 = FEATS_9 + ["vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv"]

FEAT_SHORT = {f: f.replace("_dprime", "").replace("vowel_triangle_area", "VTA")
              .replace("speech_rate", "spkrate").replace("pause_rate", "pause")
              .replace("vowel_duration_cv", "vdurCV") for f in FEATS_13}

SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except:
        return None


def load():
    with open(MASTER, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =========================================================================
# TASK 3: CP across languages
# =========================================================================
def task3_cp_across_languages(rows):
    print("\n" + "=" * 100)
    print("TASK 3: CP ACROSS LANGUAGES")
    print("=" * 100)

    cp = [r for r in rows if r["aetiology"] == "cerebral_palsy"]
    by_lang = defaultdict(list)
    for r in cp:
        by_lang[r["language"]].append(r)

    print(f"\nCP speakers: {len(cp)}")
    for lang in sorted(by_lang, key=lambda l: -len(by_lang[l])):
        sev = Counter(r["severity_label"] for r in by_lang[lang])
        print(f"  {lang}: n={len(by_lang[lang])} — {dict(sev)}")

    # Per-severity comparison across languages
    print(f"\n--- Severity-matched CP profiles (HC-normalized) ---")
    for sev in ["mild", "moderate", "severe"]:
        print(f"\n  {sev.upper()}:")
        print(f"  {'lang':>5} {'n':>4}", end="")
        for f in FEATS_9:
            print(f" {FEAT_SHORT[f]:>8}", end="")
        print()

        for lang in sorted(by_lang, key=lambda l: -len(by_lang[l])):
            subset = [r for r in by_lang[lang] if r["severity_label"] == sev]
            if len(subset) < 2:
                continue
            # HC for this language
            hc = [r for r in rows if r["language"] == lang and r["aetiology"] == "healthy"]
            if len(hc) < 1:
                continue

            print(f"  {lang:>5} {len(subset):>4}", end="")
            for f in FEATS_9:
                dys_vals = [safe_float(r.get(f)) for r in subset]
                dys_vals = [v for v in dys_vals if v is not None]
                hc_vals = [safe_float(r.get(f)) for r in hc]
                hc_vals = [v for v in hc_vals if v is not None]
                if dys_vals and hc_vals:
                    ratio = np.mean(dys_vals) / np.mean(hc_vals) if np.mean(hc_vals) > 0 else 0
                    print(f" {ratio:>8.3f}", end="")
                else:
                    print(f" {'N/A':>8}", end="")
            print()

    # Which features collapse most in CP across all languages?
    print(f"\n--- CP feature collapse ranking (mean ratio to HC, all languages pooled) ---")
    ratios = {}
    for f in FEATS_9:
        all_ratios = []
        for lang in by_lang:
            hc = [r for r in rows if r["language"] == lang and r["aetiology"] == "healthy"]
            if len(hc) < 1:
                continue
            dys_vals = [safe_float(r.get(f)) for r in by_lang[lang]]
            dys_vals = [v for v in dys_vals if v is not None]
            hc_vals = [safe_float(r.get(f)) for r in hc]
            hc_vals = [v for v in hc_vals if v is not None]
            if dys_vals and hc_vals and np.mean(hc_vals) > 0:
                all_ratios.append(np.mean(dys_vals) / np.mean(hc_vals))
        if all_ratios:
            ratios[f] = np.mean(all_ratios)

    for f, ratio in sorted(ratios.items(), key=lambda x: x[1]):
        fname = FEAT_SHORT[f]
        bar = "#" * int((1 - ratio) * 50)
        print(f"  {fname:>12}: {ratio:.3f} (collapse={1-ratio:.1%}) {bar}")


# =========================================================================
# TASK 4: Severity correlation per dataset
# =========================================================================
def task4_severity_correlation(rows):
    print("\n" + "=" * 100)
    print("TASK 4: SEVERITY CORRELATION PER DATASET")
    print("=" * 100)

    # Only speakers with known severity
    labelled = [r for r in rows if r["severity_label"] in SEV_MAP]

    by_ds = defaultdict(list)
    for r in labelled:
        by_ds[r["dataset"]].append(r)

    print(f"\n  {'Dataset':>25} {'n':>5}", end="")
    for f in FEATS_13:
        print(f" {FEAT_SHORT[f]:>8}", end="")
    print()
    print("  " + "-" * (30 + 9 * len(FEATS_13)))

    # Collect all correlations for summary
    all_rhos = defaultdict(list)

    for ds in sorted(by_ds, key=lambda d: -len(by_ds[d])):
        ds_rows = by_ds[ds]
        if len(ds_rows) < 10:
            continue
        # Need at least 2 severity levels
        sevs = set(r["severity_label"] for r in ds_rows)
        if len(sevs) < 2:
            continue

        print(f"  {ds:>25} {len(ds_rows):>5}", end="")
        for f in FEATS_13:
            pairs = [(SEV_MAP[r["severity_label"]], safe_float(r.get(f)))
                     for r in ds_rows]
            pairs = [(s, v) for s, v in pairs if v is not None]
            if len(pairs) >= 10 and len(set(p[0] for p in pairs)) >= 2:
                rho, p = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
                sig = "*" if p < 0.05 else " "
                print(f" {rho:>+7.2f}{sig}", end="")
                all_rhos[f].append(rho)
            else:
                print(f" {'N/A':>8}", end="")
        print()

    # Summary: mean rho per feature across datasets
    print(f"\n  Mean Spearman rho across datasets (negative = decreases with severity):")
    for f in sorted(all_rhos, key=lambda x: np.mean(all_rhos[x])):
        rhos = all_rhos[f]
        fname = FEAT_SHORT[f]
        m = np.mean(rhos)
        n_sig = sum(1 for r in rhos if abs(r) > 0.2)
        print(f"    {fname:>12}: mean rho={m:>+.3f} ({len(rhos)} datasets, {n_sig} with |rho|>0.2)")


# =========================================================================
# TASK 5: Feature importance ranking
# =========================================================================
def task5_feature_importance(rows):
    print("\n" + "=" * 100)
    print("TASK 5: FEATURE IMPORTANCE RANKING")
    print("=" * 100)

    labelled = [r for r in rows if r["severity_label"] in SEV_MAP]
    print(f"\nLabelled speakers: {len(labelled)}")

    # Method 1: Kruskal-Wallis H across 4 severity levels
    print(f"\n--- Method 1: Kruskal-Wallis H (4 severity classes) ---")
    kw_results = []
    for f in FEATS_13:
        groups = defaultdict(list)
        for r in labelled:
            v = safe_float(r.get(f))
            if v is not None:
                groups[r["severity_label"]].append(v)
        valid = {s: g for s, g in groups.items() if len(g) >= 5}
        if len(valid) >= 2:
            H, p = stats.kruskal(*valid.values())
            N = sum(len(g) for g in valid.values())
            k = len(valid)
            eta2 = max((H - k + 1) / (N - k), 0)
            kw_results.append((f, H, p, eta2))

    kw_results.sort(key=lambda x: -x[3])
    print(f"  {'Rank':>4} {'Feature':>12} {'H':>10} {'eta2':>8} {'Effect':>10}")
    print("  " + "-" * 50)
    for rank, (f, H, p, eta2) in enumerate(kw_results, 1):
        effect = "large" if eta2 >= 0.14 else "medium" if eta2 >= 0.06 else "small"
        print(f"  {rank:>4} {FEAT_SHORT[f]:>12} {H:>10.1f} {eta2:>8.3f} {effect:>10}")

    # Method 2: Mean absolute Spearman rho across all datasets
    print(f"\n--- Method 2: Mean |Spearman rho| across datasets ---")
    by_ds = defaultdict(list)
    for r in labelled:
        by_ds[r["dataset"]].append(r)

    feat_rhos = defaultdict(list)
    for ds, ds_rows in by_ds.items():
        if len(ds_rows) < 10:
            continue
        sevs = set(r["severity_label"] for r in ds_rows)
        if len(sevs) < 2:
            continue
        for f in FEATS_13:
            pairs = [(SEV_MAP[r["severity_label"]], safe_float(r.get(f))) for r in ds_rows]
            pairs = [(s, v) for s, v in pairs if v is not None]
            if len(pairs) >= 10:
                rho, _ = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
                feat_rhos[f].append(abs(rho))

    ranked = sorted(feat_rhos.items(), key=lambda x: -np.mean(x[1]))
    print(f"  {'Rank':>4} {'Feature':>12} {'mean|rho|':>10} {'n_datasets':>10} {'min':>6} {'max':>6}")
    print("  " + "-" * 55)
    for rank, (f, rhos) in enumerate(ranked, 1):
        print(f"  {rank:>4} {FEAT_SHORT[f]:>12} {np.mean(rhos):>10.3f} {len(rhos):>10} {min(rhos):>6.3f} {max(rhos):>6.3f}")

    # Minimal predictive set
    print(f"\n--- Minimal set (top 5 by both methods) ---")
    top_kw = set(f for f, _, _, _ in kw_results[:5])
    top_rho = set(f for f, _ in ranked[:5])
    both = top_kw & top_rho
    print(f"  Top 5 by KW eta2: {[FEAT_SHORT[f] for f,_,_,_ in kw_results[:5]]}")
    print(f"  Top 5 by |rho|:   {[FEAT_SHORT[f] for f,_ in ranked[:5]]}")
    print(f"  In both:          {[FEAT_SHORT[f] for f in both]}")


# =========================================================================
# TASK 6: HC distribution across languages
# =========================================================================
def task6_hc_universals(rows):
    print("\n" + "=" * 100)
    print("TASK 6: HC DISTRIBUTION ACROSS LANGUAGES")
    print("=" * 100)

    hc = [r for r in rows if r["aetiology"] == "healthy"]
    by_lang = defaultdict(list)
    for r in hc:
        by_lang[r["language"]].append(r)

    print(f"\nHC speakers: {len(hc)} across {len(by_lang)} languages")

    # Mean profile per language
    print(f"\n--- HC mean profiles per language ---")
    print(f"  {'lang':>5} {'n':>5}", end="")
    for f in FEATS_9:
        print(f" {FEAT_SHORT[f]:>8}", end="")
    print()
    print("  " + "-" * (10 + 9 * len(FEATS_9)))

    for lang in sorted(by_lang, key=lambda l: -len(by_lang[l])):
        n = len(by_lang[lang])
        if n < 3:
            continue
        print(f"  {lang:>5} {n:>5}", end="")
        for f in FEATS_9:
            vals = [safe_float(r.get(f)) for r in by_lang[lang]]
            vals = [v for v in vals if v is not None]
            print(f" {np.mean(vals):>8.2f}" if vals else f" {'N/A':>8}", end="")
        print()

    # Coefficient of variation across languages (lower = more universal)
    print(f"\n--- Feature universality (CV across language means, lower = more universal) ---")
    cvs = {}
    for f in FEATS_9:
        lang_means = []
        for lang, lr in by_lang.items():
            if len(lr) < 3:
                continue
            vals = [safe_float(r.get(f)) for r in lr]
            vals = [v for v in vals if v is not None]
            if vals:
                lang_means.append(np.mean(vals))
        if len(lang_means) >= 3:
            cv = np.std(lang_means) / np.mean(lang_means) if np.mean(lang_means) > 0 else 0
            cvs[f] = cv

    for f, cv in sorted(cvs.items(), key=lambda x: x[1]):
        fname = FEAT_SHORT[f]
        label = "UNIVERSAL" if cv < 0.25 else "variable" if cv < 0.40 else "LANGUAGE-DEPENDENT"
        print(f"  {fname:>12}: CV={cv:.3f} ({label})")

    # Kruskal-Wallis: does language significantly affect HC d-primes?
    print(f"\n--- Kruskal-Wallis: language effect on HC features ---")
    for f in FEATS_9:
        groups = []
        lang_labels = []
        for lang in sorted(by_lang):
            vals = [safe_float(r.get(f)) for r in by_lang[lang]]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 5:
                groups.append(vals)
                lang_labels.append(lang)
        if len(groups) >= 2:
            H, p = stats.kruskal(*groups)
            N = sum(len(g) for g in groups)
            k = len(groups)
            eta2 = max((H - k + 1) / (N - k), 0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            fname = FEAT_SHORT[f]
            print(f"  {fname:>12}: H={H:>8.1f}, eta2={eta2:.3f}, p={p:.2e} {sig}")


def main():
    rows = load()
    print(f"Loaded {len(rows)} speakers")

    task3_cp_across_languages(rows)
    task4_severity_correlation(rows)
    task5_feature_importance(rows)
    task6_hc_universals(rows)


if __name__ == "__main__":
    main()
