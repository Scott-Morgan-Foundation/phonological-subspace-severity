#!/usr/bin/env python3
"""
Track 4 Phase 2A.1: Fixed-Token Subsampling Analysis.

Loads saved per-speaker phone embeddings (.npz), subsamples exactly N tokens
per phonological class per speaker, recomputes d', repeats 100x, and reports
mean d' ± 95% CI. Demonstrates that severity correlations hold when token
count is equalized across speakers.

Usage:
    python track4/scripts/fixed_token_subsampling.py
    python track4/scripts/fixed_token_subsampling.py --n-tokens 50 --n-repeats 200
"""

import argparse
import csv
import json
import os
import sys
import warnings
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EMB_DIR = RESULTS_DIR / "embeddings"
INVENTORY_CSV = RESULTS_DIR / "speaker_inventory.csv"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
OUTPUT_DIR = RESULTS_DIR / "reviewer_experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

CONSONANT_FEATURES = ["nasal", "voicing", "strident", "sonorant", "manner"]
MIN_TOKENS_DEFAULT = 30
N_REPEATS_DEFAULT = 100


def load_phone_config(lang):
    config_path = CONFIG_DIR / f"phone_features_{lang}.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_phone(phone, config):
    cats = {}
    for feat_name, feat_def in config.get("consonant_features", {}).items():
        if phone in set(feat_def["positive"]):
            cats[feat_name] = "positive"
        elif phone in set(feat_def["negative"]):
            cats[feat_name] = "negative"
    return cats


def compute_dprime(embs_pos, embs_neg, direction):
    proj_pos = embs_pos @ direction
    proj_neg = embs_neg @ direction
    gap = proj_pos.mean() - proj_neg.mean()
    pooled_std = np.sqrt((proj_pos.var() + proj_neg.var()) / 2)
    return float(abs(gap / pooled_std)) if pooled_std > 1e-8 else 0.0


def compute_direction(embs_pos, embs_neg):
    d = embs_pos.mean(0) - embs_neg.mean(0)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-8 else None


def load_inventory():
    records = []
    with open(INVENTORY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def load_speaker_embeddings(dataset, speaker_id):
    """Load saved .npz embeddings for a speaker."""
    safe_spk = speaker_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    path = EMB_DIR / f"{dataset}__{safe_spk}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "phones": list(data["phones"]),
        "embeddings": data["embeddings"],
        "durations": data["durations"],
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed-token subsampling analysis")
    parser.add_argument("--n-tokens", type=int, default=MIN_TOKENS_DEFAULT,
                        help=f"Tokens per class per speaker (default {MIN_TOKENS_DEFAULT})")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS_DEFAULT,
                        help=f"Subsampling repetitions (default {N_REPEATS_DEFAULT})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N = args.n_tokens
    R = args.n_repeats

    print("=" * 70)
    print(f"FIXED-TOKEN SUBSAMPLING ANALYSIS")
    print(f"N = {N} tokens per class, R = {R} repeats, seed = {args.seed}")
    print("=" * 70)

    np.random.seed(args.seed)

    if not EMB_DIR.exists():
        print(f"ERROR: Embeddings dir not found: {EMB_DIR}")
        print("Run extract_features.py --save-embeddings first")
        sys.exit(1)

    # Load inventory
    inventory = load_inventory()
    inv_lookup = {}
    for rec in inventory:
        inv_lookup[(rec["dataset"], rec["speaker_id"])] = rec

    # Dataset → language mapping
    DATASET_LANG = {
        "SAP": "en", "COPAS": "nl", "TORGO": "en", "Neurovoz": "es",
        "YouTube_French": "fr", "MDSC": "zh", "IPVS": "it",
        "UASPEECH": "en", "UASPEECH_control": "en",
        "LibriSpeech_English": "en", "PC-GITA": "es",
    }

    # Load all speaker embeddings
    print("\nLoading embeddings...")
    speaker_data = {}  # (dataset, speaker_id) -> {phones, embeddings, durations}
    npz_files = sorted(EMB_DIR.glob("*.npz"))
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        ds = str(data["dataset"])
        spk = str(data["speaker_id"])
        speaker_data[(ds, spk)] = {
            "phones": list(data["phones"]),
            "embeddings": data["embeddings"],
            "durations": data["durations"],
        }
    print(f"Loaded {len(speaker_data)} speakers")

    # Load phone configs per language
    configs = {}
    for lang in set(DATASET_LANG.values()):
        try:
            configs[lang] = load_phone_config(lang)
        except FileNotFoundError:
            pass

    # === Step 1: Compute HC feature directions (using ALL tokens, not subsampled) ===
    print("\nComputing feature directions from HC (full token set)...")
    LANGUAGE_HC_DATASETS = {
        "en": ["LibriSpeech_English", "TORGO", "UASPEECH_control"],
        "nl": ["COPAS"], "fr": ["YouTube_French"],
        "es": ["Neurovoz", "PC-GITA"], "zh": ["MDSC"],
        "it": ["IPVS"],
    }

    directions_by_lang = {}
    for lang, hc_datasets in LANGUAGE_HC_DATASETS.items():
        if lang not in configs:
            continue
        config = configs[lang]
        ctrl_by_feat = defaultdict(lambda: defaultdict(list))

        for (ds, spk), data in speaker_data.items():
            if ds not in hc_datasets:
                continue
            rec = inv_lookup.get((ds, spk), {})
            is_ctrl = str(rec.get("is_control", "False")).lower() == "true"
            if not is_ctrl:
                continue
            for phone, emb in zip(data["phones"], data["embeddings"]):
                cats = classify_phone(phone, config)
                for feat, cls in cats.items():
                    ctrl_by_feat[feat][cls].append(emb)

        directions = {}
        for feat in CONSONANT_FEATURES:
            pos = np.array(ctrl_by_feat[feat].get("positive", []))
            neg = np.array(ctrl_by_feat[feat].get("negative", []))
            d = compute_direction(pos, neg)
            if d is not None:
                directions[feat] = d
        directions_by_lang[lang] = directions
        print(f"  {lang}: {len(directions)} directions")

    # === Step 2: For each speaker, classify phones and check eligibility ===
    print(f"\nClassifying phones and checking N≥{N} per class...")

    eligible_speakers = []  # list of (ds, spk, lang, sev_num, feat_classes)
    ineligible = 0

    for (ds, spk), data in speaker_data.items():
        lang = DATASET_LANG.get(ds)
        if lang is None or lang not in configs:
            continue
        rec = inv_lookup.get((ds, spk), {})
        sev_num = int(rec.get("severity_numeric", -1))
        if sev_num < 0:
            continue

        config = configs[lang]
        feat_classes = defaultdict(lambda: defaultdict(list))
        for i, phone in enumerate(data["phones"]):
            cats = classify_phone(phone, config)
            for feat, cls in cats.items():
                feat_classes[feat][cls].append(i)

        # Check: does this speaker have ≥N tokens per class for at least one feature?
        has_any = False
        for feat in CONSONANT_FEATURES:
            pos_n = len(feat_classes[feat].get("positive", []))
            neg_n = len(feat_classes[feat].get("negative", []))
            if pos_n >= N and neg_n >= N:
                has_any = True
                break

        if has_any:
            eligible_speakers.append((ds, spk, lang, sev_num, feat_classes, data["embeddings"]))
        else:
            ineligible += 1

    print(f"Eligible: {len(eligible_speakers)} speakers (≥{N} tokens in at least one feature)")
    print(f"Ineligible: {ineligible} speakers")

    # === Step 3: Subsampled d' computation (R repeats) ===
    print(f"\nRunning {R} subsampling iterations...")

    # For each repeat, for each speaker, for each feature: subsample N tokens, compute d'
    all_repeat_results = []  # list of dicts per repeat

    for r in range(R):
        repeat_results = []
        for (ds, spk, lang, sev_num, feat_classes, embeddings) in eligible_speakers:
            directions = directions_by_lang.get(lang, {})
            row = {"dataset": ds, "speaker_id": spk, "severity_numeric": sev_num}

            for feat in CONSONANT_FEATURES:
                direction = directions.get(feat)
                if direction is None:
                    row[f"{feat}_dprime"] = float("nan")
                    continue

                pos_idx = feat_classes[feat].get("positive", [])
                neg_idx = feat_classes[feat].get("negative", [])

                if len(pos_idx) < N or len(neg_idx) < N:
                    row[f"{feat}_dprime"] = float("nan")
                    continue

                # Subsample exactly N from each class
                pos_sample = np.random.choice(pos_idx, N, replace=False)
                neg_sample = np.random.choice(neg_idx, N, replace=False)

                pos_embs = embeddings[pos_sample]
                neg_embs = embeddings[neg_sample]

                row[f"{feat}_dprime"] = compute_dprime(pos_embs, neg_embs, direction)

            repeat_results.append(row)
        all_repeat_results.append(repeat_results)

        if (r + 1) % 20 == 0:
            print(f"  Repeat {r+1}/{R}")

    # === Step 4: Aggregate and compute correlations per repeat ===
    print("\nComputing correlations per repeat...")

    feat_cols = [f"{f}_dprime" for f in CONSONANT_FEATURES]
    repeat_correlations = {feat: [] for feat in feat_cols}

    for repeat_results in all_repeat_results:
        for feat in feat_cols:
            sevs = [r["severity_numeric"] for r in repeat_results if not np.isnan(r.get(feat, float("nan")))]
            vals = [r[feat] for r in repeat_results if not np.isnan(r.get(feat, float("nan")))]
            if len(sevs) >= 10:
                rho, _ = stats.spearmanr(sevs, vals)
                repeat_correlations[feat].append(rho)

    # === Step 5: Report ===
    print("\n" + "=" * 70)
    print(f"RESULTS: Fixed-Token Subsampling (N={N}, R={R})")
    print("=" * 70)

    print(f"\n{'Feature':<25} {'Mean rho':>10} {'95% CI':>22} {'SD':>8} {'N repeats':>10}")
    print("-" * 78)

    results = {}
    for feat in feat_cols:
        rhos = repeat_correlations[feat]
        if not rhos:
            continue
        mean_rho = np.mean(rhos)
        ci_low = np.percentile(rhos, 2.5)
        ci_high = np.percentile(rhos, 97.5)
        sd = np.std(rhos)
        print(f"{feat:<25} {mean_rho:>10.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {sd:>8.4f} {len(rhos):>10}")

        results[feat] = {
            "mean_rho": float(mean_rho),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "sd": float(sd),
            "n_repeats": len(rhos),
            "n_eligible_speakers": len(eligible_speakers),
            "n_tokens_per_class": N,
        }

    # Compare with unsubsampled
    print(f"\n{'Feature':<25} {'Subsampled':>12} {'Original':>12} {'Difference':>12}")
    print("-" * 65)

    # Load original results for comparison
    orig_csv = RESULTS_DIR / "track4_results_merged.csv"
    if orig_csv.exists():
        import pandas as pd
        df_orig = pd.read_csv(orig_csv)
        df_orig = df_orig[df_orig.severity_numeric >= 0]
        for feat in feat_cols:
            valid = df_orig[["severity_numeric", feat]].dropna()
            if len(valid) >= 10:
                orig_rho, _ = stats.spearmanr(valid.severity_numeric, valid[feat])
                sub_rho = results.get(feat, {}).get("mean_rho", float("nan"))
                diff = sub_rho - orig_rho
                print(f"{feat:<25} {sub_rho:>12.4f} {orig_rho:>12.4f} {diff:>+12.4f}")
                if feat in results:
                    results[feat]["original_rho"] = float(orig_rho)

    # Save
    output_path = OUTPUT_DIR / "fixed_token_subsampling.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    print("\nDONE")


if __name__ == "__main__":
    main()
