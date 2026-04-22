#!/usr/bin/env python3
"""
Task #2: Cross-lingual aetiology consistency.

For each aetiology present in 2+ languages:
  - Compare 13-feature profiles across languages
  - Cosine similarity between language-specific profiles
  - Kruskal-Wallis across languages within same aetiology
  - Are the signatures language-invariant?
"""

import csv
import os
import numpy as np
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
MASTER = BASE / "results" / "track4" / "track4_master.csv"

FEATS_9 = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime",
    "high_dprime", "low_dprime", "back_dprime", "round_dprime",
]

FEATS_13 = FEATS_9 + ["vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv"]

FEAT_SHORT = {f: f.replace("_dprime", "").replace("vowel_triangle_area", "VTA")
              .replace("speech_rate", "spkrate").replace("pause_rate", "pause")
              .replace("vowel_duration_cv", "vdurCV") for f in FEATS_13}

AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke",
             "mixed": "Mixed", "cleft_palate": "Cleft", "laryngectomy": "Laryn",
             "voice_disorder": "VoiceDis", "unknown": "Unk"}


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except (ValueError, TypeError):
        return None


def get_mean_profile(rows_subset, feats):
    """Per-feature mean from available data."""
    means = []
    ns = []
    for f in feats:
        vals = [safe_float(r.get(f)) for r in rows_subset]
        vals = [v for v in vals if v is not None]
        means.append(np.mean(vals) if vals else np.nan)
        ns.append(len(vals))
    return np.array(means), ns


def cosine_sim(a, b):
    """Cosine similarity, handling NaN."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    return 1 - cosine(a[mask], b[mask])


def main():
    with open(MASTER, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=" * 100)
    print("CROSS-LINGUAL AETIOLOGY CONSISTENCY")
    print("=" * 100)

    # Group by aetiology + language
    by_aet_lang = defaultdict(lambda: defaultdict(list))
    for r in rows:
        aet = r.get("aetiology", "unknown")
        lang = r.get("language", "?")
        by_aet_lang[aet][lang].append(r)

    # =========================================================================
    # TABLE 1: Which aetiologies span multiple languages?
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 1: Aetiology x Language Coverage (n speakers)")
    print("=" * 100)

    all_langs = sorted(set(r.get("language", "?") for r in rows))
    aets_multi = []

    print(f"\n  {'Aetiology':>18}", end="")
    for lang in all_langs:
        print(f" {lang:>5}", end="")
    print(f" {'#langs':>6}")
    print("  " + "-" * (18 + 6 * len(all_langs) + 7))

    for aet in sorted(by_aet_lang.keys()):
        langs_with_data = {l for l, rs in by_aet_lang[aet].items() if len(rs) >= 3}
        n_langs = len(langs_with_data)
        short = AET_SHORT.get(aet, aet[:6])
        print(f"  {short:>18}", end="")
        for lang in all_langs:
            n = len(by_aet_lang[aet].get(lang, []))
            print(f" {n:>5}" if n > 0 else f" {'':>5}", end="")
        print(f" {n_langs:>6}")
        if n_langs >= 2:
            aets_multi.append(aet)

    print(f"\n  Aetiologies in 2+ languages (n>=3): {', '.join(AET_SHORT.get(a,a) for a in aets_multi)}")

    # =========================================================================
    # TABLE 2: Profile similarity across languages (cosine sim on 9 d-primes)
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 2: Cross-lingual Profile Similarity (cosine sim, 9 d-prime features)")
    print("=" * 100)

    for aet in aets_multi:
        langs = sorted([l for l, rs in by_aet_lang[aet].items() if len(rs) >= 3])
        if len(langs) < 2:
            continue

        short = AET_SHORT.get(aet, aet)
        profiles = {}
        for lang in langs:
            m, ns = get_mean_profile(by_aet_lang[aet][lang], FEATS_9)
            profiles[lang] = m

        print(f"\n  {short} ({len(langs)} languages):")

        # Pairwise cosine similarity
        print(f"    {'':>5}", end="")
        for l2 in langs:
            n = len(by_aet_lang[aet][l2])
            print(f" {l2+'('+str(n)+')':>10}", end="")
        print()

        for l1 in langs:
            n1 = len(by_aet_lang[aet][l1])
            print(f"    {l1+'('+str(n1)+')':>10}", end="")
            for l2 in langs:
                if l1 == l2:
                    print(f" {'--':>10}", end="")
                else:
                    sim = cosine_sim(profiles[l1], profiles[l2])
                    print(f" {sim:>10.3f}" if not np.isnan(sim) else f" {'N/A':>10}", end="")
            print()

    # =========================================================================
    # TABLE 3: Per-feature consistency — Kruskal-Wallis across languages
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 3: Per-feature Language Effect within Aetiology (Kruskal-Wallis p-value)")
    print("  (p > 0.05 = feature is language-invariant for this aetiology)")
    print("=" * 100)

    for aet in aets_multi:
        langs = sorted([l for l, rs in by_aet_lang[aet].items() if len(rs) >= 5])
        if len(langs) < 2:
            continue

        short = AET_SHORT.get(aet, aet)
        print(f"\n  {short}:")
        print(f"    {'Feature':>12}", end="")
        for lang in langs:
            print(f" {lang:>7}", end="")
        print(f" {'H':>8} {'p':>10} {'Invariant':>10}")
        print("    " + "-" * (12 + 8 * len(langs) + 30))

        n_invariant = 0
        for feat in FEATS_9:
            fname = FEAT_SHORT[feat]
            groups = []
            print(f"    {fname:>12}", end="")
            for lang in langs:
                vals = [safe_float(r.get(feat)) for r in by_aet_lang[aet][lang]]
                vals = [v for v in vals if v is not None]
                groups.append(vals)
                m = np.mean(vals) if vals else float('nan')
                print(f" {m:>7.2f}" if not np.isnan(m) else f" {'N/A':>7}", end="")

            valid_groups = [g for g in groups if len(g) >= 3]
            if len(valid_groups) >= 2:
                H, p = stats.kruskal(*valid_groups)
                inv = "YES" if p > 0.05 else "no"
                if p > 0.05:
                    n_invariant += 1
                print(f" {H:>8.1f} {p:>10.3f} {inv:>10}")
            else:
                print(f" {'N/A':>8} {'N/A':>10} {'N/A':>10}")

        print(f"    Language-invariant features: {n_invariant}/9")

    # =========================================================================
    # TABLE 4: HC-normalized profiles — do shapes match across languages?
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 4: HC-Normalized Aetiology Profiles by Language (ratio to language-specific HC)")
    print("=" * 100)

    for aet in ["parkinsons", "cerebral_palsy", "als"]:
        if aet not in aets_multi:
            continue
        langs = sorted([l for l, rs in by_aet_lang[aet].items() if len(rs) >= 5])
        if len(langs) < 2:
            continue

        short = AET_SHORT.get(aet, aet)
        print(f"\n  {short} (HC-normalized per language):")
        print(f"    {'Lang':>5} {'n_dys':>5} {'n_hc':>5}", end="")
        for feat in FEATS_9:
            print(f" {FEAT_SHORT[feat]:>8}", end="")
        print()
        print("    " + "-" * (15 + 9 * len(FEATS_9)))

        for lang in langs:
            dys_rows = by_aet_lang[aet][lang]
            hc_rows = by_aet_lang["healthy"].get(lang, [])
            if len(hc_rows) < 3:
                continue

            dys_m, _ = get_mean_profile(dys_rows, FEATS_9)
            hc_m, _ = get_mean_profile(hc_rows, FEATS_9)
            hc_safe = hc_m.copy()
            hc_safe[hc_safe == 0] = 1
            ratio = dys_m / hc_safe

            print(f"    {lang:>5} {len(dys_rows):>5} {len(hc_rows):>5}", end="")
            for v in ratio:
                print(f" {v:>8.3f}" if not np.isnan(v) else f" {'N/A':>8}", end="")
            print()

    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    # Compute mean cosine similarity per aetiology
    for aet in aets_multi:
        langs = sorted([l for l, rs in by_aet_lang[aet].items() if len(rs) >= 3])
        if len(langs) < 2:
            continue
        profiles = {}
        for lang in langs:
            m, _ = get_mean_profile(by_aet_lang[aet][lang], FEATS_9)
            profiles[lang] = m
        sims = []
        for l1, l2 in combinations(langs, 2):
            s = cosine_sim(profiles[l1], profiles[l2])
            if not np.isnan(s):
                sims.append(s)
        if sims:
            short = AET_SHORT.get(aet, aet)
            print(f"  {short}: mean cosine sim = {np.mean(sims):.3f} (range {min(sims):.3f}-{max(sims):.3f}, {len(langs)} languages)")


if __name__ == "__main__":
    main()
