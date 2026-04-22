#!/usr/bin/env python3
"""
True fixed-token d-prime estimator + permutation baseline for cross-lingual cosine.

For each speaker and each phonological contrast:
  1. Cap tokens at N per class (subsample if more, skip if fewer)
  2. Compute d-prime from the subsampled tokens
  3. Repeat K times with different random draws
  4. Report mean d-prime and bootstrap CI

Also: permutation test for cross-lingual cosine similarity.

Run on DGX:
    cd ~/dysarthria && source venv/bin/activate
    python -u track4/scripts/fixed_token_dprime.py 2>&1 | tee logs/fixed_token_dprime.log
"""

import csv
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine
from itertools import combinations

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
MASTER = BASE / "results" / "track4" / "track4_master.csv"
EMB_DIR = BASE / "results" / "track4" / "embeddings"
CONFIG_DIR = BASE / "config"
OUTPUT_DIR = BASE / "results" / "track4" / "robustness"

SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
MAIN_AETS = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}

CONS_CONTRASTS = ["nasal", "voicing", "sonorant", "strident", "manner"]
CONS_FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]

TOKEN_BUDGETS = [20, 50, 100, 200]
N_REPEATS = 50  # repeats per token budget
N_PERM = 1000   # permutations for null distribution
RNG = np.random.RandomState(42)


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except:
        return None


def load_master():
    with open(MASTER, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_phone_features(language):
    """Load phone-to-feature mapping for a language."""
    config_path = CONFIG_DIR / f"phone_features_{language}.json"
    if not config_path.exists():
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def compute_dprime_from_embeddings(emb_pos, emb_neg):
    """Compute d-prime from two sets of embeddings along their discriminant axis."""
    if len(emb_pos) < 2 or len(emb_neg) < 2:
        return np.nan
    mean_pos = np.mean(emb_pos, axis=0)
    mean_neg = np.mean(emb_neg, axis=0)
    direction = mean_pos - mean_neg
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return 0.0
    direction = direction / norm
    proj_pos = emb_pos @ direction
    proj_neg = emb_neg @ direction
    mu_pos, mu_neg = np.mean(proj_pos), np.mean(proj_neg)
    var_pos, var_neg = np.var(proj_pos, ddof=1), np.var(proj_neg, ddof=1)
    pooled_std = np.sqrt((var_pos + var_neg) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return (mu_pos - mu_neg) / pooled_std


LANG_MAP = {
    "en": "en", "nl": "nl", "es": "es", "fr": "fr", "zh": "zh",
    "it": "it", "de": "de", "pt": "pt", "hu": "hu", "ta": "ta",
    "sk": "de",  # Slovak uses Czech/German phone features as fallback
    "sw": "en",  # Swahili uses English phone features as fallback
}


def get_phone_classes(phone_features, contrast):
    """Get positive and negative phone sets for a contrast."""
    cons_feats = phone_features.get("consonant_features", {})
    mapping = cons_feats.get(contrast, {})
    pos = set(mapping.get("positive", []))
    neg = set(mapping.get("negative", []))
    return pos, neg


# =========================================================================
# EXPERIMENT A: True fixed-token d-prime
# =========================================================================
def experiment_fixed_token(master_rows):
    print("=" * 80, flush=True)
    print("EXPERIMENT A: True fixed-token d-prime estimator", flush=True)
    print("=" * 80, flush=True)

    # Build speaker metadata
    speaker_meta = {}
    for r in master_rows:
        key = f"{r['dataset']}__{r['speaker_id']}"
        speaker_meta[key] = r

    # Load phone features per language
    phone_feats = {}
    for lang in LANG_MAP.values():
        if lang not in phone_feats:
            pf = load_phone_features(lang)
            if pf:
                phone_feats[lang] = pf

    # For each token budget, compute d-prime for all speakers
    results_by_budget = {}

    for budget in TOKEN_BUDGETS:
        print(f"\n  --- Token budget: {budget} per class ---", flush=True)
        all_speaker_results = []

        npz_files = sorted(EMB_DIR.glob("*.npz"))
        n_processed = 0
        n_skipped = 0

        for npz_path in npz_files:
            key = npz_path.stem
            meta = speaker_meta.get(key)
            if meta is None:
                continue

            lang = meta.get("language", "en")
            lang_key = LANG_MAP.get(lang, "en")
            pf = phone_feats.get(lang_key)
            if pf is None:
                n_skipped += 1
                continue

            sev = meta.get("severity_label", "unknown")
            aet = meta.get("aetiology", "unknown")
            if sev not in SEV_MAP or aet not in MAIN_AETS:
                continue

            # Load embeddings
            data = np.load(npz_path, allow_pickle=True)
            phones = data["phones"]
            embeddings = data["embeddings"]

            # For each consonant contrast, collect pos/neg embeddings
            contrast_dprimes = []
            for contrast in CONS_CONTRASTS:
                pos_phones, neg_phones = get_phone_classes(pf, contrast)
                if not pos_phones or not neg_phones:
                    contrast_dprimes.append(np.nan)
                    continue

                pos_idx = [i for i, p in enumerate(phones) if p in pos_phones]
                neg_idx = [i for i, p in enumerate(phones) if p in neg_phones]

                if len(pos_idx) < budget or len(neg_idx) < budget:
                    contrast_dprimes.append(np.nan)
                    continue

                # Repeated subsampling
                repeat_dprimes = []
                for _ in range(N_REPEATS):
                    sub_pos = RNG.choice(pos_idx, size=budget, replace=False)
                    sub_neg = RNG.choice(neg_idx, size=budget, replace=False)
                    dp = compute_dprime_from_embeddings(
                        embeddings[sub_pos], embeddings[sub_neg]
                    )
                    repeat_dprimes.append(dp)

                contrast_dprimes.append(np.mean(repeat_dprimes))

            valid = [d for d in contrast_dprimes if not np.isnan(d)]
            if len(valid) >= 3:
                composite = np.mean(valid)
                all_speaker_results.append({
                    "speaker": key,
                    "severity": SEV_MAP[sev],
                    "aetiology": aet,
                    "composite": composite,
                    "n_contrasts": len(valid),
                })
                n_processed += 1

        print(f"  Processed: {n_processed}, Skipped: {n_skipped}", flush=True)

        if len(all_speaker_results) < 20:
            print(f"  Too few speakers for budget {budget}", flush=True)
            continue

        # Severity correlation
        sevs = [s["severity"] for s in all_speaker_results]
        comps = [s["composite"] for s in all_speaker_results]
        rho, p = stats.spearmanr(sevs, comps)
        print(f"  Severity rho: {rho:+.4f} (p={p:.2e}, n={len(all_speaker_results)})", flush=True)

        # Per-severity means
        by_sev = defaultdict(list)
        for s in all_speaker_results:
            by_sev[s["severity"]].append(s["composite"])
        for sev_num in sorted(by_sev.keys()):
            sname = ["control", "mild", "moderate", "severe"][sev_num]
            vals = by_sev[sev_num]
            print(f"    {sname:>10}: mean={np.mean(vals):.3f}, n={len(vals)}", flush=True)

        # Aetiology discrimination
        aet_groups = defaultdict(list)
        for s in all_speaker_results:
            aet_groups[s["aetiology"]].append(s["composite"])
        valid_aets = {a: g for a, g in aet_groups.items() if len(g) >= 5}
        if len(valid_aets) >= 2:
            H, p_aet = stats.kruskal(*valid_aets.values())
            N_aet = sum(len(g) for g in valid_aets.values())
            k = len(valid_aets)
            eps2 = max((H - k + 1) / (N_aet - k), 0)
            print(f"  Aetiology: H={H:.1f}, p={p_aet:.2e}, eps2={eps2:.3f}", flush=True)

        results_by_budget[budget] = {
            "n": len(all_speaker_results),
            "rho": rho,
            "p": p,
        }

    # Summary table
    print(f"\n  Summary across token budgets:", flush=True)
    print(f"  {'Budget':>8} {'n':>6} {'rho':>8} {'p':>12}", flush=True)
    print("  " + "-" * 40, flush=True)
    for budget in TOKEN_BUDGETS:
        if budget in results_by_budget:
            r = results_by_budget[budget]
            print(f"  {budget:>8} {r['n']:>6} {r['rho']:>+8.4f} {r['p']:>12.2e}", flush=True)

    return results_by_budget


# =========================================================================
# EXPERIMENT B: Permutation baseline for cross-lingual cosine
# =========================================================================
def experiment_permutation_cosine(master_rows):
    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT B: Permutation baseline for cross-lingual cosine", flush=True)
    print("=" * 80, flush=True)

    by_aet_lang = defaultdict(lambda: defaultdict(list))
    for r in master_rows:
        aet = r.get("aetiology", "unknown")
        lang = r.get("language", "?")
        by_aet_lang[aet][lang].append(r)

    def mean_profile(subset):
        means = []
        for f in CONS_FEATS:
            vals = [safe_float(r.get(f)) for r in subset]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else np.nan)
        return np.array(means)

    def cosine_sim(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 3:
            return np.nan
        return 1 - cosine(a[mask], b[mask])

    for aet in ["parkinsons", "cerebral_palsy", "als", "healthy"]:
        langs = sorted([l for l, rs in by_aet_lang[aet].items() if len(rs) >= 3])
        if len(langs) < 2:
            continue

        short = AET_SHORT.get(aet, aet)

        # Observed mean pairwise cosine
        pair_sims = []
        for l1, l2 in combinations(langs, 2):
            p1 = mean_profile(by_aet_lang[aet][l1])
            p2 = mean_profile(by_aet_lang[aet][l2])
            sim = cosine_sim(p1, p2)
            if not np.isnan(sim):
                pair_sims.append(sim)

        if not pair_sims:
            continue

        observed_mean = np.mean(pair_sims)

        # Permutation null: shuffle language labels within this aetiology
        all_speakers = []
        for lang in langs:
            for r in by_aet_lang[aet][lang]:
                all_speakers.append(r)

        perm_means = []
        for _ in range(N_PERM):
            # Shuffle language assignments
            shuffled_langs = [s.get("language") for s in all_speakers]
            RNG.shuffle(shuffled_langs)

            # Regroup by shuffled language
            perm_groups = defaultdict(list)
            for s, sl in zip(all_speakers, shuffled_langs):
                perm_groups[sl].append(s)

            # Compute pairwise cosine on permuted groups
            perm_sims = []
            for l1, l2 in combinations(langs, 2):
                if len(perm_groups[l1]) >= 3 and len(perm_groups[l2]) >= 3:
                    p1 = mean_profile(perm_groups[l1])
                    p2 = mean_profile(perm_groups[l2])
                    sim = cosine_sim(p1, p2)
                    if not np.isnan(sim):
                        perm_sims.append(sim)
            if perm_sims:
                perm_means.append(np.mean(perm_sims))

        perm_means = np.array(perm_means)
        p_perm = np.mean(perm_means >= observed_mean)
        null_mean = np.mean(perm_means)
        null_lo, null_hi = np.percentile(perm_means, [2.5, 97.5])

        print(f"\n  {short} ({len(langs)} languages, {len(pair_sims)} pairs):", flush=True)
        print(f"    Observed mean cosine: {observed_mean:.4f}", flush=True)
        print(f"    Null distribution: mean={null_mean:.4f} [{null_lo:.4f}, {null_hi:.4f}]", flush=True)
        print(f"    p(perm >= observed): {p_perm:.4f}", flush=True)
        print(f"    Observed exceeds {(1-p_perm)*100:.1f}% of null", flush=True)


def main():
    print("=" * 80, flush=True)
    print("FIXED-TOKEN D-PRIME + PERMUTATION BASELINE", flush=True)
    print("=" * 80, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_master()
    print(f"Loaded {len(rows)} speakers", flush=True)
    print(f"Embeddings dir: {EMB_DIR}", flush=True)
    print(f"NPZ files: {len(list(EMB_DIR.glob('*.npz')))}", flush=True)

    t0 = time.time()

    results_a = experiment_fixed_token(rows)
    experiment_permutation_cosine(rows)

    elapsed = time.time() - t0
    print(f"\n\nAll experiments completed in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
