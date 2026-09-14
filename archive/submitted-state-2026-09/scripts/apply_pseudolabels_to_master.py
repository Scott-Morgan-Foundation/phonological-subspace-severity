#!/usr/bin/env python3
"""
Apply pseudo-labels to master CSV speakers with unknown severity.

Two approaches:
1. Manifest-based: SAP, EWA-DB, Hungarian already have pseudo_severity in their manifests
2. Feature-threshold: For remaining datasets (CDSD, Domotica, EasyCall, SVD, AVFAD),
   use d-prime thresholds calibrated from labelled speakers in the master CSV

Threshold method: per-language, uses mean of top-5 consonant d-primes as composite score.
Thresholds from labelled control/mild/moderate/severe centroids (midpoints).
"""

import csv
import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
MASTER = BASE / "results" / "track4" / "track4_master.csv"
OUTPUT = BASE / "results" / "track4" / "track4_master.csv"  # overwrite

SEVERITY_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
SEVERITY_ORDER = ["control", "mild", "moderate", "severe"]

# Datasets with pseudo-labels already in manifests
MANIFEST_DATASETS = {
    "SAP": BASE / "data" / "processed" / "SAP" / "manifest_v2.jsonl",
    "EWA-DB": BASE / "data" / "processed" / "EWA-DB" / "manifest_v2.jsonl",
    "Hungarian_Dysarthria": BASE / "data" / "processed" / "Hungarian_Dysarthria" / "manifest_v2.jsonl",
}

# Datasets needing feature-threshold pseudo-labels
THRESHOLD_DATASETS = ["CDSD", "Domotica", "EasyCall", "SVD", "AVFAD", "Hungarian_Toth_2026"]

# 5 consonant d-primes (strongest severity markers)
CONSONANT_DPRIMES = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]

# All 13 features for full threshold
FEATS_13 = CONSONANT_DPRIMES + [
    "high_dprime", "low_dprime", "back_dprime", "round_dprime",
    "vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv",
]

# Language mapping for per-language thresholds
DATASET_LANGUAGE = {
    "SAP": "en", "COPAS": "nl", "TORGO": "en", "Neurovoz": "es",
    "YouTube_French": "fr", "MDSC": "zh", "IPVS": "it",
    "VOC-ALS": "it", "PC-GITA": "es",
    "UASPEECH": "en", "UASPEECH_control": "en",
    "LibriSpeech_English": "en", "EWA-DB": "sk",
    "Hungarian_Dysarthria": "hu", "Hungarian_HC": "hu",
    "Hungarian_Toth_2026": "hu",
    "CDSD": "zh", "YouTube_German": "de", "SVD": "de",
    "EasyCall": "it", "Domotica": "nl",
    "SSNCE_Tamil": "ta", "SLR65_Tamil": "ta", "AVFAD": "pt",
}


def load_manifest_pseudo_labels(manifest_path):
    """Load per-speaker pseudo-severity from manifest."""
    speaker_labels = {}
    with open(manifest_path) as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("speaker_id", "")
            ps = d.get("pseudo_severity", "")
            src = d.get("severity_source", "")
            if ps and ps != "unknown" and "track4" in src:
                if sid not in speaker_labels:
                    speaker_labels[sid] = ps
    return speaker_labels


def get_composite_score(row):
    """Mean of available consonant d-primes (higher = better)."""
    vals = []
    for f in CONSONANT_DPRIMES:
        v = row.get(f, "")
        if v and v not in ("", "nan"):
            try:
                fv = float(v)
                if not np.isnan(fv):
                    vals.append(fv)
            except ValueError:
                pass
    return np.mean(vals) if len(vals) >= 3 else None


def compute_thresholds(rows):
    """Compute per-language severity thresholds from labelled speakers.

    Uses midpoints between severity-class centroids of the composite score.
    Falls back to global thresholds if a language has too few labelled speakers.
    """
    # Group labelled speakers by language and severity
    by_lang_sev = defaultdict(lambda: defaultdict(list))
    global_by_sev = defaultdict(list)

    for r in rows:
        sev = r.get("severity_label", "unknown")
        if sev not in SEVERITY_ORDER:
            continue
        score = get_composite_score(r)
        if score is None:
            continue
        lang = DATASET_LANGUAGE.get(r["dataset"], "en")
        by_lang_sev[lang][sev].append(score)
        global_by_sev[sev].append(score)

    # Compute centroids and thresholds per language
    thresholds = {}

    # Global thresholds first
    global_centroids = {}
    for sev in SEVERITY_ORDER:
        if global_by_sev[sev]:
            global_centroids[sev] = np.mean(global_by_sev[sev])

    if len(global_centroids) >= 2:
        global_thresh = _midpoint_thresholds(global_centroids)
        thresholds["global"] = global_thresh
        print(f"\n  Global thresholds (n={sum(len(v) for v in global_by_sev.values())} speakers):")
        print(f"    Centroids: " + ", ".join(f"{s}={global_centroids[s]:.2f}" for s in SEVERITY_ORDER if s in global_centroids))
        print(f"    Thresholds: severe<{global_thresh[0]:.2f}, moderate<{global_thresh[1]:.2f}, mild<{global_thresh[2]:.2f}, control>={global_thresh[2]:.2f}")

    # Per-language thresholds
    for lang in sorted(by_lang_sev):
        centroids = {}
        for sev in SEVERITY_ORDER:
            if by_lang_sev[lang][sev]:
                centroids[sev] = np.mean(by_lang_sev[lang][sev])
        n_total = sum(len(by_lang_sev[lang][s]) for s in SEVERITY_ORDER)
        if len(centroids) >= 2 and n_total >= 20:
            thresholds[lang] = _midpoint_thresholds(centroids)
            print(f"  {lang} thresholds (n={n_total}): " +
                  ", ".join(f"{s}={centroids[s]:.2f}" for s in SEVERITY_ORDER if s in centroids))
        else:
            print(f"  {lang}: too few labelled ({n_total}) — using global")

    return thresholds


def _midpoint_thresholds(centroids):
    """Compute 3 thresholds from up to 4 centroids (midpoints between adjacent classes).

    Returns (severe_upper, moderate_upper, mild_upper) — all in terms of composite score.
    Score decreases with severity, so: severe < moderate < mild < control.
    """
    ordered = [(sev, centroids[sev]) for sev in SEVERITY_ORDER if sev in centroids]
    # Ensure ordered by severity (severe=lowest score to control=highest)
    # Our d-primes decrease with severity, so control > mild > moderate > severe

    # Default fallbacks
    t_severe = 0.5   # below this = severe
    t_moderate = 1.0  # below this = moderate
    t_mild = 1.5      # below this = mild, above = control

    if "severe" in centroids and "moderate" in centroids:
        t_severe = (centroids["severe"] + centroids["moderate"]) / 2
    elif "severe" in centroids and "mild" in centroids:
        t_severe = centroids["severe"] + (centroids["mild"] - centroids["severe"]) * 0.33
    elif "severe" in centroids:
        t_severe = centroids["severe"] * 1.2

    if "moderate" in centroids and "mild" in centroids:
        t_moderate = (centroids["moderate"] + centroids["mild"]) / 2
    elif "moderate" in centroids and "control" in centroids:
        t_moderate = centroids["moderate"] + (centroids["control"] - centroids["moderate"]) * 0.33
    elif "moderate" in centroids:
        t_moderate = centroids["moderate"] * 1.15

    if "mild" in centroids and "control" in centroids:
        t_mild = (centroids["mild"] + centroids["control"]) / 2
    elif "mild" in centroids:
        t_mild = centroids["mild"] * 1.1

    return (t_severe, t_moderate, t_mild)


def classify_by_threshold(score, thresholds):
    """Classify a composite score into severity using thresholds."""
    t_severe, t_moderate, t_mild = thresholds
    if score < t_severe:
        return "severe"
    elif score < t_moderate:
        return "moderate"
    elif score < t_mild:
        return "mild"
    else:
        return "control"


def main():
    print("=" * 80)
    print("Pseudo-label application to Track 4 master CSV")
    print("=" * 80)

    # Step 1: Load pseudo-labels from manifests
    print("\n--- Step 1: Manifest-based pseudo-labels ---")
    manifest_pseudo = {}
    for ds, manifest_path in MANIFEST_DATASETS.items():
        if manifest_path.exists():
            labels = load_manifest_pseudo_labels(manifest_path)
            for sid, sev in labels.items():
                manifest_pseudo[(ds, sid)] = sev
            counts = defaultdict(int)
            for s in labels.values():
                counts[s] += 1
            print(f"  {ds}: {len(labels)} pseudo-labels — {dict(counts)}")
        else:
            print(f"  {ds}: manifest not found at {manifest_path}")

    # Step 2: Load master CSV
    print("\n--- Step 2: Load master CSV ---")
    with open(MASTER, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"  {len(rows)} speakers loaded")

    # Count current state
    n_unknown_before = sum(1 for r in rows if r.get("severity_label", "unknown") in ("unknown", ""))
    print(f"  {n_unknown_before} speakers with unknown severity")

    # Step 3: Apply manifest pseudo-labels
    print("\n--- Step 3: Apply manifest pseudo-labels ---")
    n_manifest = 0
    for row in rows:
        key = (row["dataset"], row["speaker_id"])
        if key in manifest_pseudo:
            if row.get("severity_label", "unknown") in ("unknown", ""):
                ps = manifest_pseudo[key]
                row["severity_label"] = ps
                row["severity_numeric"] = str(SEVERITY_MAP.get(ps, -1))
                n_manifest += 1
    print(f"  Applied {n_manifest} manifest pseudo-labels")

    # Step 4: Compute thresholds from labelled speakers for remaining unknowns
    print("\n--- Step 4: Feature-threshold pseudo-labels ---")
    thresholds = compute_thresholds(rows)

    # Apply to remaining unknowns in threshold datasets
    n_threshold = 0
    for row in rows:
        if row["dataset"] not in THRESHOLD_DATASETS:
            continue
        if row.get("severity_label", "unknown") not in ("unknown", ""):
            continue
        # Check if control (is_control flag)
        if row.get("is_control", "False").lower() == "true":
            row["severity_label"] = "control"
            row["severity_numeric"] = "0"
            n_threshold += 1
            continue

        score = get_composite_score(row)
        if score is None:
            continue

        lang = DATASET_LANGUAGE.get(row["dataset"], "en")
        thresh = thresholds.get(lang, thresholds.get("global"))
        if thresh is None:
            continue

        ps = classify_by_threshold(score, thresh)
        row["severity_label"] = ps
        row["severity_numeric"] = str(SEVERITY_MAP.get(ps, -1))
        n_threshold += 1

    print(f"  Applied {n_threshold} feature-threshold pseudo-labels")

    # Step 5: Summary
    print("\n--- Summary ---")
    n_unknown_after = sum(1 for r in rows if r.get("severity_label", "unknown") in ("unknown", ""))
    n_labelled = len(rows) - n_unknown_after
    print(f"  Before: {n_unknown_before} unknown")
    print(f"  After:  {n_unknown_after} unknown")
    print(f"  Total labelled: {n_labelled}/{len(rows)} ({100*n_labelled/len(rows):.1f}%)")

    # Severity distribution
    sev_counts = defaultdict(int)
    for r in rows:
        sev_counts[r.get("severity_label", "unknown")] += 1
    print(f"\n  Severity distribution:")
    for s in ["control", "mild", "moderate", "severe", "unknown"]:
        if s in sev_counts:
            print(f"    {s:>10}: {sev_counts[s]:>5}")

    # Per-dataset summary
    ds_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        ds_counts[r["dataset"]][r.get("severity_label", "unknown")] += 1

    print(f"\n  {'Dataset':>25} {'ctrl':>6} {'mild':>6} {'mod':>6} {'sev':>6} {'unk':>6} {'total':>6}")
    print("  " + "-" * 67)
    for ds in sorted(ds_counts):
        c = ds_counts[ds]
        total = sum(c.values())
        print(f"  {ds:>25} {c.get('control',0):>6} {c.get('mild',0):>6} {c.get('moderate',0):>6} {c.get('severe',0):>6} {c.get('unknown',0):>6} {total:>6}")

    # Write
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n  Wrote {len(rows)} speakers to {OUTPUT}")


if __name__ == "__main__":
    main()
