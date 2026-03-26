#!/usr/bin/env python3
"""
Track 4: Build master speaker inventory from all dataset manifests.

Parses manifest_v2.jsonl files for the 8 dysarthric datasets and HC datasets
used in the phonological subspace experiment. Outputs speaker_inventory.csv
with metadata needed for all downstream Track 4 analysis.

Runs on DGX Spark.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "track4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "speaker_inventory.csv"

# Datasets to include, with language and MFA model info
DATASET_CONFIG = {
    # --- Dysarthric datasets ---
    "SAP": {
        "language": "en",
        "mfa_model": "english_mfa",
        "severity_scale": "intelligibility_ordinal",
        "notes": "All 5 aetiologies; need SAPC2 CSV for text; HC from LibriSpeech",
    },
    "COPAS": {
        "language": "nl",
        "mfa_model": "dutch_mfa",
        "severity_scale": "intelligibility_pct",
        "notes": "Mixed dysarthria + matched controls",
    },
    "TORGO": {
        "language": "en",
        "mfa_model": "english_mfa",
        "severity_scale": "clinical_4class",
        "notes": "CP; matched controls",
    },
    "Neurovoz": {
        "language": "es",
        "mfa_model": "spanish_mfa",
        "severity_scale": "GRBAS_G",
        "notes": "PD; matched controls",
    },
    "YouTube_French": {
        "language": "fr",
        "mfa_model": "french_mfa",
        "severity_scale": "manual_3class",
        "notes": "ALS; longitudinal for 2 speakers",
    },
    "MDSC": {
        "language": "zh",
        "mfa_model": "mandarin_mfa",
        "severity_scale": "intelligibility_pct",
        "notes": "CP + Wilsons; matched controls",
    },
    "IPVS": {
        "language": "it",
        "mfa_model": "italian_mfa",
        "severity_scale": "none",
        "notes": "PD; matched controls; binary HC/PD only (no clinical severity available, CPS3 is reading speed not severity)",
        "binary_only": True,  # Flag: use HC vs PD, not 4-class severity
    },
    "VOC-ALS": {
        "language": "it",
        "mfa_model": "italian_mfa",
        "severity_scale": "clinical_4class",
        "notes": "ALS; matched controls; sustained vowels + syllables; vowel triangle only (no MFA)",
    },
    "PC-GITA": {
        "language": "es",
        "mfa_model": "spanish_mfa",
        "severity_scale": "UPDRS_speech",
        "notes": "PD; matched controls; has sustained /a/ /i/ /u/; vowel triangle + MFA sentences",
    },
    "UASPEECH": {
        "language": "en",
        "mfa_model": "english_mfa",
        "severity_scale": "clinical_binary",
        "notes": "CP; word-only — vowel triangle via MFA; no moderate class",
    },
    "UASPEECH_control": {
        "language": "en",
        "mfa_model": "english_mfa",
        "severity_scale": "none",
        "notes": "HC for UASPEECH; word-only — vowel triangle via MFA",
        "is_hc_only": True,
    },
    # --- Healthy control datasets (for languages without matched controls) ---
    "LibriSpeech_English": {
        "language": "en",
        "mfa_model": "english_mfa",
        "severity_scale": "none",
        "notes": "HC for SAP English reference",
        "is_hc_only": True,
        "max_speakers": 150,
    },
}

# Content types to include (skip single words, DDK, vowels for main analysis)
SENTENCE_CONTENT_TYPES = {"sentence", "sentences", "spontaneous", "read_speech",
                          "monologue", "paragraph", "connected_speech", "command",
                          "unknown"}

# Severity mapping helpers
SEVERITY_CANONICAL = {
    "control": "control", "healthy": "control", "normal": "control",
    "mild": "mild", "low": "mild",
    "moderate": "moderate", "medium": "moderate",
    "severe": "severe", "high": "severe",
    "very_low": "mild", "very_severe": "severe",
}


def stipancic_severity(intell_pct: float) -> str:
    """Map intelligibility % to severity using Stipancic et al. (2022) thresholds.

    Stipancic, K. L., Tjaden, K., & Wilding, G. (2022).
    Standard thresholds used universally across COPAS, MDSC, UASPEECH, etc.

    > 94%: control, 85-94%: mild, 70-84%: moderate, < 70%: severe
    """
    if intell_pct > 94:
        return "control"
    elif intell_pct >= 85:
        return "mild"
    elif intell_pct >= 70:
        return "moderate"
    else:
        return "severe"


def map_severity(raw_severity: str, intelligibility_pct: Optional[float] = None) -> str:
    """Map raw severity labels to canonical 4-class.

    If intelligibility_pct is available, uses Stipancic et al. (2022) thresholds
    for a universal, literature-grounded mapping.
    """
    # Prefer intelligibility-based mapping when available
    if intelligibility_pct is not None:
        return stipancic_severity(intelligibility_pct)
    if not raw_severity or raw_severity == "unknown":
        return "unknown"
    return SEVERITY_CANONICAL.get(raw_severity.lower().strip(), raw_severity.lower().strip())


def load_manifest(dataset_name: str) -> List[Dict]:
    """Load manifest_v2.jsonl for a dataset."""
    # Handle dir name mismatches
    dir_candidates = [
        dataset_name,
        dataset_name.replace("_", "-"),
        dataset_name.replace("-", "_"),
    ]

    manifest_path = None
    for candidate in dir_candidates:
        p = PROCESSED_DIR / candidate / "manifest_v2.jsonl"
        if p.exists():
            manifest_path = p
            break

    if manifest_path is None:
        print(f"  WARNING: No manifest found for {dataset_name} "
              f"(tried: {', '.join(dir_candidates)})")
        return []

    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                samples.append(record)
            except json.JSONDecodeError:
                continue

    return samples


def load_copas_intelligibility() -> Dict[str, float]:
    """Load per-speaker intelligibility % from COPAS TSV files.

    COPAS stores intelligibility in train.tsv/test.tsv but NOT in manifest_v2.jsonl.
    Returns {speaker_id: intelligibility_pct}.
    """
    copas_dir = PROCESSED_DIR / "COPAS"
    speaker_intel = {}

    for split in ["train", "test"]:
        tsv_path = copas_dir / f"{split}.tsv"
        if not tsv_path.exists():
            continue
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                spk_id = row.get("speaker_id", "")
                intel = row.get("intelligibility")
                if spk_id and intel:
                    try:
                        speaker_intel[spk_id] = float(intel)
                    except (ValueError, TypeError):
                        pass

    return speaker_intel


def build_speaker_records(dataset_name: str, config: dict, samples: List[Dict]) -> List[Dict]:
    """Aggregate samples into per-speaker records."""
    speakers = defaultdict(lambda: {
        "n_sentences": 0, "n_words": 0, "n_vowels": 0,
        "durations": [], "content_types": set(),
        "severity_raw": None, "aetiology": None,
        "sex": None, "age": None, "is_control": False,
        "intelligibility_pct": None, "clinical_score": None,
    })

    is_hc_only = config.get("is_hc_only", False)
    max_speakers = config.get("max_speakers", 0)

    for s in samples:
        spk_id = s.get("speaker_id", "unknown")
        spk = speakers[spk_id]

        # Severity
        sev_raw = s.get("severity", "unknown")
        if spk["severity_raw"] is None or spk["severity_raw"] == "unknown":
            spk["severity_raw"] = sev_raw

        # Aetiology
        aet = s.get("aetiology", "unknown")
        if spk["aetiology"] is None or spk["aetiology"] == "unknown":
            spk["aetiology"] = aet

        # Demographics
        if spk["sex"] is None:
            spk["sex"] = s.get("sex", s.get("gender", ""))
        if spk["age"] is None:
            spk["age"] = s.get("age", "")

        # Control status
        is_ctrl = s.get("is_dysarthric") is False or sev_raw in ("control", "healthy", "normal")
        if is_ctrl or is_hc_only:
            spk["is_control"] = True

        # Content type counting
        ct = s.get("content_type", "unknown").lower()
        spk["content_types"].add(ct)
        dur = s.get("duration", 0.0)
        spk["durations"].append(dur)

        if ct in ("word", "single_word", "words"):
            spk["n_words"] += 1
        elif ct in ("vowel", "sustained_vowel", "vowels"):
            spk["n_vowels"] += 1
        else:
            spk["n_sentences"] += 1

        # Intelligibility / clinical scores
        intel = s.get("intelligibility_pct", s.get("intelligibility"))
        if intel is not None and spk["intelligibility_pct"] is None:
            try:
                spk["intelligibility_pct"] = float(intel)
            except (ValueError, TypeError):
                pass
        clin = s.get("clinical_score", s.get("updrs_speech", s.get("grbas_g")))
        if clin is not None and spk["clinical_score"] is None:
            spk["clinical_score"] = str(clin)

    # Convert to list of records
    records = []
    speaker_ids = sorted(speakers.keys())

    # Limit HC-only datasets
    if max_speakers > 0 and len(speaker_ids) > max_speakers:
        speaker_ids = speaker_ids[:max_speakers]

    binary_only = config.get("binary_only", False)

    for spk_id in speaker_ids:
        spk = speakers[spk_id]
        # Use Stipancic thresholds when intelligibility is available
        sev_label = map_severity(spk["severity_raw"] or "unknown",
                                 intelligibility_pct=spk["intelligibility_pct"])

        # Update control status based on Stipancic mapping
        if sev_label == "control":
            spk["is_control"] = True

        # Binary-only datasets: keep control, set all dysarthric to "dysarthric" (no severity grading)
        if binary_only and not spk["is_control"]:
            sev_label = "dysarthric"

        # Numeric severity for ordering
        sev_numeric = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}.get(sev_label, -1)

        records.append({
            "dataset": dataset_name,
            "speaker_id": spk_id,
            "language": config["language"],
            "aetiology": spk["aetiology"] or "unknown",
            "severity_label": sev_label,
            "severity_scale": config["severity_scale"],
            "severity_numeric": sev_numeric,
            "sex": spk["sex"] or "",
            "age": spk["age"] or "",
            "is_control": spk["is_control"],
            "n_sentences": spk["n_sentences"],
            "n_words": spk["n_words"],
            "n_vowels": spk["n_vowels"],
            "intelligibility_pct": spk["intelligibility_pct"] if spk["intelligibility_pct"] is not None else "",
            "clinical_score": spk["clinical_score"] or "",
            "clinical_notes": config.get("notes", ""),
        })

    return records


def print_summary(all_records: List[Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SPEAKER INVENTORY SUMMARY")
    print("=" * 80)

    # By dataset
    by_dataset = defaultdict(list)
    for r in all_records:
        by_dataset[r["dataset"]].append(r)

    print(f"\n{'Dataset':<25} {'Speakers':>8} {'Dys':>5} {'Ctrl':>5} {'Sentences':>10} {'Language':<5}")
    print("-" * 70)
    total_spk = 0
    total_dys = 0
    total_ctrl = 0
    for ds in sorted(by_dataset.keys()):
        recs = by_dataset[ds]
        n_spk = len(recs)
        n_dys = sum(1 for r in recs if not r["is_control"])
        n_ctrl = sum(1 for r in recs if r["is_control"])
        n_sent = sum(r["n_sentences"] for r in recs)
        lang = recs[0]["language"]
        print(f"{ds:<25} {n_spk:>8} {n_dys:>5} {n_ctrl:>5} {n_sent:>10} {lang:<5}")
        total_spk += n_spk
        total_dys += n_dys
        total_ctrl += n_ctrl
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_spk:>8} {total_dys:>5} {total_ctrl:>5}")

    # By aetiology
    print(f"\n{'Aetiology':<20} {'Speakers':>8} {'Ctrl':>5} {'Mild':>5} {'Mod':>5} {'Sev':>5} {'Unk':>5}")
    print("-" * 55)
    by_aet = defaultdict(lambda: defaultdict(int))
    for r in all_records:
        aet = r["aetiology"] if not r["is_control"] else "control"
        by_aet[aet][r["severity_label"]] += 1
        by_aet[aet]["total"] += 1
    for aet in sorted(by_aet.keys()):
        counts = by_aet[aet]
        print(f"{aet:<20} {counts['total']:>8} {counts.get('control', 0):>5} "
              f"{counts.get('mild', 0):>5} {counts.get('moderate', 0):>5} "
              f"{counts.get('severe', 0):>5} {counts.get('unknown', 0):>5}")

    # By language
    print(f"\n{'Language':<10} {'Speakers':>8}")
    print("-" * 20)
    by_lang = defaultdict(int)
    for r in all_records:
        by_lang[r["language"]] += 1
    for lang in sorted(by_lang.keys()):
        print(f"{lang:<10} {by_lang[lang]:>8}")

    # By sex
    print(f"\n{'Sex':<10} {'Speakers':>8}")
    print("-" * 20)
    by_sex = defaultdict(int)
    for r in all_records:
        by_sex[r["sex"] or "unknown"] += 1
    for sex in sorted(by_sex.keys()):
        print(f"{sex:<10} {by_sex[sex]:>8}")


def main():
    print("=" * 80)
    print("Track 4: Building Speaker Inventory")
    print("=" * 80)
    print(f"Base dir: {BASE_DIR}")
    print(f"Processed dir: {PROCESSED_DIR}")

    all_records = []

    # Pre-load COPAS intelligibility from TSV (not in manifest)
    copas_intel = load_copas_intelligibility()
    if copas_intel:
        print(f"\nLoaded COPAS intelligibility for {len(copas_intel)} speakers from TSV")

    for dataset_name, config in DATASET_CONFIG.items():
        print(f"\nLoading {dataset_name}...")
        samples = load_manifest(dataset_name)
        if not samples:
            continue
        print(f"  {len(samples)} total samples")

        # Inject COPAS intelligibility into samples before building records
        if dataset_name == "COPAS" and copas_intel:
            for s in samples:
                spk_id = s.get("speaker_id", "")
                if spk_id in copas_intel and "intelligibility_pct" not in s:
                    s["intelligibility_pct"] = copas_intel[spk_id]

        records = build_speaker_records(dataset_name, config, samples)
        print(f"  {len(records)} speakers")
        all_records.extend(records)

    # Write CSV
    if all_records:
        fieldnames = [
            "dataset", "speaker_id", "language", "aetiology",
            "severity_label", "severity_scale", "severity_numeric",
            "sex", "age", "is_control", "n_sentences", "n_words", "n_vowels",
            "intelligibility_pct", "clinical_score", "clinical_notes",
        ]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\nWrote {len(all_records)} speaker records to {OUTPUT_CSV}")

    print_summary(all_records)


if __name__ == "__main__":
    main()
