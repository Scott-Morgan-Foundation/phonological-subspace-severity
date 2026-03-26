#!/usr/bin/env python3
"""
Cross-corpus severity comparison after token-count normalization.

Tests whether equalized d' values enable cross-corpus severity prediction:
1. Fixed-token subsampled d' for all speakers
2. HC convergence check (same d' across corpora within language?)
3. Cross-corpus severity prediction (thresholds from one corpus applied to another)
4. Universal severity thresholds

Uses saved .npz embeddings from extract_features.py --save-embeddings.
"""

import numpy as np
import json, os, csv
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# Configuration
# =============================================================================

EMB_DIR = Path(os.environ.get("EMB_DIR", os.path.expanduser("~/dysarthria/results/track4/embeddings")))
INV_CSV = Path(os.environ.get("INV_CSV", os.path.expanduser("~/dysarthria/results/track4/speaker_inventory.csv")))
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", os.path.expanduser("~/dysarthria/config")))
DIRECTIONS_FILE = os.environ.get("DIRECTIONS", os.path.expanduser("~/dysarthria/en_feature_directions.npz"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", os.path.expanduser("~/dysarthria/results/track4/reviewer_experiments")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_LANG = {
    "SAP": "en", "COPAS": "nl", "TORGO": "en", "Neurovoz": "es",
    "YouTube_French": "fr", "MDSC": "zh", "IPVS": "it",
    "UASPEECH": "en", "UASPEECH_control": "en",
    "LibriSpeech_English": "en", "PC-GITA": "es",
}
FEATURES = ["nasal", "voicing", "strident", "sonorant", "manner"]
N_TOKENS = 30
N_REPEATS = 50
SEED = 42

# =============================================================================
# Helpers
# =============================================================================

def load_configs():
    configs = {}
    for lang in set(DATASET_LANG.values()):
        p = CONFIG_DIR / f"phone_features_{lang}.json"
        if p.exists():
            with open(p) as f:
                configs[lang] = json.load(f)
    return configs

def classify(phone, config):
    cats = {}
    for fn, fd in config.get("consonant_features", {}).items():
        if phone in set(fd["positive"]): cats[fn] = "positive"
        elif phone in set(fd["negative"]): cats[fn] = "negative"
    return cats

def compute_direction(embs_pos, embs_neg):
    if len(embs_pos) < 5 or len(embs_neg) < 5: return None
    d = embs_pos.mean(0) - embs_neg.mean(0)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-8 else None

def compute_dprime(pos, neg, direction):
    pp, pn = pos @ direction, neg @ direction
    gap = pp.mean() - pn.mean()
    ps = np.sqrt((pp.var() + pn.var()) / 2)
    return abs(gap / ps) if ps > 1e-8 else 0.0

def subsampled_dprime(pos_all, neg_all, direction, n=N_TOKENS, repeats=N_REPEATS):
    """Compute mean d' over repeated subsampling to fixed N tokens."""
    if len(pos_all) < n or len(neg_all) < n:
        return float("nan")
    dprimes = []
    for _ in range(repeats):
        pi = np.random.choice(len(pos_all), n, replace=False)
        ni = np.random.choice(len(neg_all), n, replace=False)
        dprimes.append(compute_dprime(pos_all[pi], neg_all[ni], direction))
    return np.mean(dprimes)


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(SEED)
    configs = load_configs()

    # Load inventory
    inv = {}
    with open(INV_CSV) as f:
        for row in csv.DictReader(f):
            inv[(row["dataset"], row["speaker_id"])] = row

    # Load HC directions per language
    # For English, use pre-computed. For others, compute from HC embeddings.
    en_dirs = {}
    if Path(DIRECTIONS_FILE).exists():
        data = np.load(DIRECTIONS_FILE)
        en_dirs = {k: data[k] for k in data.files}

    # We need per-language directions. For this experiment, compute from HC within each language.
    print("=" * 70)
    print("CROSS-CORPUS SEVERITY COMPARISON (Token-Normalized)")
    print(f"N={N_TOKENS} tokens/class, {N_REPEATS} repeats, seed={SEED}")
    print("=" * 70)

    # Load all embeddings
    print("\nLoading embeddings...")
    speakers = {}  # (ds, spk) -> {phones, embeddings, durations, lang, sev, is_ctrl}
    npz_files = sorted(EMB_DIR.glob("*.npz"))
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        ds, spk = str(data["dataset"]), str(data["speaker_id"])
        lang = DATASET_LANG.get(ds)
        if lang is None: continue
        rec = inv.get((ds, spk), {})
        sev = rec.get("severity_label", "unknown")
        sev_num = int(rec.get("severity_numeric", -1))
        is_ctrl = str(rec.get("is_control", "False")).lower() == "true"

        # Classify phones
        if lang not in configs: continue
        config = configs[lang]
        by_feat = defaultdict(lambda: defaultdict(list))
        for phone, emb in zip(list(data["phones"]), data["embeddings"]):
            for feat, cls in classify(phone, config).items():
                by_feat[feat][cls].append(emb)

        # Convert to arrays
        feat_data = {}
        for feat in FEATURES:
            pos = np.array(by_feat[feat].get("positive", []))
            neg = np.array(by_feat[feat].get("negative", []))
            feat_data[feat] = {"pos": pos, "neg": neg}

        speakers[(ds, spk)] = {
            "lang": lang, "sev": sev, "sev_num": sev_num,
            "is_ctrl": is_ctrl, "dataset": ds, "feat_data": feat_data,
        }
    print(f"Loaded {len(speakers)} speakers")

    # Compute per-language HC directions
    print("\nComputing per-language HC directions...")
    lang_dirs = {}
    LANG_HC = {
        "en": ["LibriSpeech_English", "TORGO", "UASPEECH_control"],
        "nl": ["COPAS"], "fr": ["YouTube_French"],
        "es": ["Neurovoz", "PC-GITA"], "zh": ["MDSC"], "it": ["IPVS"],
    }
    for lang, hc_datasets in LANG_HC.items():
        hc_by_feat = defaultdict(lambda: defaultdict(list))
        for (ds, spk), s in speakers.items():
            if ds not in hc_datasets or not s["is_ctrl"]: continue
            for feat in FEATURES:
                for cls in ["positive", "negative"]:
                    hc_by_feat[feat][cls].extend(list(s["feat_data"][feat][cls[:3]]))
        dirs = {}
        for feat in FEATURES:
            pos = np.array(hc_by_feat[feat].get("positive", []))
            neg = np.array(hc_by_feat[feat].get("negative", []))
            d = compute_direction(pos, neg)
            if d is not None: dirs[feat] = d
        lang_dirs[lang] = dirs
        print(f"  {lang}: {len(dirs)} directions")

    # =========================================================================
    # Step 1: Compute equalized d' for all speakers
    # =========================================================================
    print("\nComputing equalized d' (subsampled)...")
    for (ds, spk), s in speakers.items():
        dirs = lang_dirs.get(s["lang"], {})
        s["eq_dprime"] = {}
        for feat in FEATURES:
            d = dirs.get(feat)
            if d is None: s["eq_dprime"][feat] = float("nan"); continue
            pos = s["feat_data"][feat]["pos"]
            neg = s["feat_data"][feat]["neg"]
            s["eq_dprime"][feat] = subsampled_dprime(pos, neg, d)

    # =========================================================================
    # Step 2: HC convergence check
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: HC CONVERGENCE (do controls converge across corpora?)")
    print("=" * 70)

    for lang in ["en", "es"]:
        lang_name = {"en": "English", "es": "Spanish"}[lang]
        print(f"\n--- {lang_name} ---")
        # Group HC by corpus
        hc_by_corpus = defaultdict(lambda: defaultdict(list))
        for (ds, spk), s in speakers.items():
            if s["lang"] != lang or not s["is_ctrl"]: continue
            for feat in FEATURES:
                v = s["eq_dprime"].get(feat, float("nan"))
                if not np.isnan(v):
                    hc_by_corpus[ds][feat].append(v)

        if len(hc_by_corpus) < 2:
            print("  Not enough corpora with HC")
            continue

        for feat in FEATURES[:3]:  # top 3
            print(f"\n  {feat}:")
            corpus_means = {}
            for ds in sorted(hc_by_corpus.keys()):
                vals = hc_by_corpus[ds].get(feat, [])
                if vals:
                    m, sd = np.mean(vals), np.std(vals)
                    corpus_means[ds] = m
                    print(f"    {ds:<25} mean={m:.3f} sd={sd:.3f} n={len(vals)}")

            if len(corpus_means) >= 2:
                vals = list(corpus_means.values())
                spread = max(vals) - min(vals)
                mean_val = np.mean(vals)
                cv = spread / mean_val * 100 if mean_val > 0 else 0
                print(f"    Cross-corpus spread: {spread:.3f} ({cv:.1f}% of mean)")

    # =========================================================================
    # Step 3: Cross-corpus severity prediction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: CROSS-CORPUS SEVERITY PREDICTION")
    print("=" * 70)

    # For each pair of corpora (same language), train thresholds on one, predict on other
    # Use nasality d' as the primary feature (strongest correlate)
    feat = "nasal"

    for lang in ["en", "nl", "es", "zh"]:
        # Get corpora with severity labels in this language
        corpora = defaultdict(list)
        for (ds, spk), s in speakers.items():
            if s["lang"] != lang or s["sev_num"] < 0: continue
            v = s["eq_dprime"].get(feat)
            if v is not None and not np.isnan(v):
                corpora[ds].append({"dprime": v, "sev": s["sev"], "sev_num": s["sev_num"]})

        corpus_list = [ds for ds in corpora if len(corpora[ds]) >= 10]
        if len(corpus_list) < 2: continue

        print(f"\n--- {lang.upper()} (nasality d') ---")

        for train_ds in corpus_list:
            for test_ds in corpus_list:
                if train_ds == test_ds: continue

                # Learn thresholds from train corpus
                train = corpora[train_ds]
                test = corpora[test_ds]

                # Simple threshold: median d' of each severity group
                sev_medians = {}
                for sev_num in [0, 1, 2, 3]:
                    vals = [r["dprime"] for r in train if r["sev_num"] == sev_num]
                    if vals: sev_medians[sev_num] = np.median(vals)

                if len(sev_medians) < 2: continue

                # Create thresholds between adjacent severity levels
                sorted_sevs = sorted(sev_medians.keys())
                thresholds = []
                for i in range(len(sorted_sevs) - 1):
                    s1, s2 = sorted_sevs[i], sorted_sevs[i+1]
                    thresholds.append((sev_medians[s1] + sev_medians[s2]) / 2)

                # Predict severity on test corpus
                def predict(dprime):
                    for i, t in enumerate(thresholds):
                        if dprime >= t:
                            return sorted_sevs[i]
                    return sorted_sevs[-1]

                true_labels = [r["sev_num"] for r in test]
                pred_labels = [predict(r["dprime"]) for r in test]

                # Compute Spearman correlation between predicted and true
                rho, p = stats.spearmanr(true_labels, pred_labels)

                # Also compute within-corpus baseline
                train_true = [r["sev_num"] for r in train]
                train_pred = [predict(r["dprime"]) for r in train]
                train_rho, _ = stats.spearmanr(train_true, train_pred)

                print(f"  Train: {train_ds:<22} -> Test: {test_ds:<22} rho={rho:.3f} (within-corpus: {train_rho:.3f})")

    # =========================================================================
    # Step 4: Universal severity thresholds
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: UNIVERSAL SEVERITY THRESHOLDS (pooled across all corpora)")
    print("=" * 70)

    # Pool all speakers, compute d' percentiles per severity
    for feat in FEATURES[:3]:
        print(f"\n--- {feat} ---")
        by_sev = defaultdict(list)
        for (ds, spk), s in speakers.items():
            if s["sev_num"] < 0: continue
            v = s["eq_dprime"].get(feat)
            if v is not None and not np.isnan(v):
                by_sev[s["sev"]].append(v)

        print(f"  {'Severity':<12} {'N':>5} {'Mean':>8} {'SD':>8} {'Median':>8} {'Q25':>8} {'Q75':>8}")
        print(f"  {'-'*60}")
        for sev in ["control", "mild", "moderate", "severe"]:
            vals = by_sev.get(sev, [])
            if not vals: continue
            print(f"  {sev:<12} {len(vals):>5} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} "
                  f"{np.median(vals):>8.3f} {np.percentile(vals, 25):>8.3f} {np.percentile(vals, 75):>8.3f}")

        # Suggest thresholds (midpoint between adjacent medians)
        medians = {}
        for sev in ["control", "mild", "moderate", "severe"]:
            vals = by_sev.get(sev, [])
            if vals: medians[sev] = np.median(vals)

        if len(medians) >= 2:
            print(f"\n  Suggested thresholds:")
            sev_order = ["control", "mild", "moderate", "severe"]
            prev = None
            for sev in sev_order:
                if sev in medians:
                    if prev is not None:
                        threshold = (medians[prev] + medians[sev]) / 2
                        print(f"    {prev} / {sev} boundary: d' = {threshold:.3f}")
                    prev = sev

    # =========================================================================
    # Step 5: Overlap analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: SEVERITY GROUP OVERLAP (can we distinguish levels?)")
    print("=" * 70)

    feat = "nasal"
    by_sev = defaultdict(list)
    for (ds, spk), s in speakers.items():
        if s["sev_num"] < 0: continue
        v = s["eq_dprime"].get(feat)
        if v is not None and not np.isnan(v):
            by_sev[s["sev"]].append(v)

    # Pairwise Mann-Whitney between severity levels
    sev_order = ["control", "mild", "moderate", "severe"]
    print(f"\n  Pairwise Mann-Whitney U (nasality d', equalized N={N_TOKENS}):")
    for i in range(len(sev_order)):
        for j in range(i+1, len(sev_order)):
            a, b = by_sev.get(sev_order[i], []), by_sev.get(sev_order[j], [])
            if len(a) >= 5 and len(b) >= 5:
                U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                effect = abs(np.mean(a) - np.mean(b))
                print(f"    {sev_order[i]:>10} vs {sev_order[j]:<10} U={U:>8.0f}  p={p:.2e}  diff={effect:.3f} {star}")

    # Save results
    results = {
        "n_tokens": N_TOKENS, "n_repeats": N_REPEATS, "seed": SEED,
        "n_speakers": len(speakers),
    }

    # Collect equalized d' per speaker for output
    output_rows = []
    for (ds, spk), s in speakers.items():
        row = {"dataset": ds, "speaker_id": spk, "language": s["lang"],
               "severity": s["sev"], "severity_numeric": s["sev_num"],
               "is_control": s["is_ctrl"]}
        for feat in FEATURES:
            row[f"{feat}_eq_dprime"] = s["eq_dprime"].get(feat, float("nan"))
        output_rows.append(row)

    # Save equalized d' CSV
    eq_csv = OUTPUT_DIR / "cross_corpus_equalized_dprime.csv"
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(eq_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"\nSaved equalized d' to {eq_csv}")

    print("\nDONE")


if __name__ == "__main__":
    main()
