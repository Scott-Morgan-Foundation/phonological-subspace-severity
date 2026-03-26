#!/usr/bin/env python3
"""
Track 4: MFA Alignment Pipeline for all datasets.

Prepares MFA corpus directories (speaker/file.wav + speaker/file.lab)
and runs Montreal Forced Aligner for each dataset in the Track 4 experiment.

Supports all 6 languages: English, French, Dutch, Spanish, Mandarin, Italian.

Usage:
    # Align a single dataset
    python scripts/track4_mfa_align.py --dataset COPAS

    # Align all datasets
    python scripts/track4_mfa_align.py --all

    # Prepare corpus only (no alignment)
    python scripts/track4_mfa_align.py --dataset SAP --prepare-only

    # Use specific MFA binary
    python scripts/track4_mfa_align.py --dataset TORGO --mfa-bin ~/miniforge3/envs/mfa/bin/mfa

Runs on DGX Spark.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(os.environ.get("DYSARTHRIA_BASE", os.path.expanduser("~/dysarthria")))
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "track4"
ALIGNED_DIR = RESULTS_DIR / "aligned"
CORPUS_DIR = RESULTS_DIR / "corpus"
INVENTORY_CSV = RESULTS_DIR / "speaker_inventory.csv"

# MFA binary — try miniforge env first, then system
MFA_BIN_CANDIDATES = [
    os.path.expanduser("~/miniforge3/envs/mfa/bin/mfa"),
    "mfa",  # system PATH
]

# Dataset → alignment configuration
DATASET_ALIGN_CONFIG = {
    "SAP": {
        "language": "en",
        "mfa_acoustic": "english_mfa",
        "mfa_dict": "english_mfa",
        "content_filter": {"sentence", "sentences", "read_speech", "spontaneous",
                           "monologue", "paragraph", "connected_speech", "unknown"},
        "text_source": "manifest",
        "severity_filter": {"mild", "moderate", "severe"},  # only rated speakers
        "notes": "Only 188 severity-rated speakers; skip 1045 unrated",
    },
    "COPAS": {
        "language": "nl",
        "mfa_acoustic": "dutch_cv",
        "mfa_dict": "dutch_cv",
        "content_filter": {"sentence", "sentences", "read_speech", "spontaneous",
                           "connected_speech", "unknown"},
        "text_source": "manifest",
    },
    "TORGO": {
        "language": "en",
        "mfa_acoustic": "english_mfa",
        "mfa_dict": "english_mfa",
        "content_filter": {"sentence", "sentences", "read_speech", "connected_speech", "unknown"},
        "text_source": "manifest",
    },
    "Neurovoz": {
        "language": "es",
        "mfa_acoustic": "spanish_mfa",
        "mfa_dict": "spanish_mfa",
        "content_filter": {"sentence", "sentences", "read_speech", "spontaneous",
                           "monologue", "connected_speech", "unknown"},
        "text_source": "manifest",
    },
    "YouTube_French": {
        "language": "fr",
        "mfa_acoustic": "french_mfa",
        "mfa_dict": "french_mfa",
        "content_filter": {"sentence", "sentences", "spontaneous", "connected_speech", "unknown"},
        "text_source": "manifest",
    },
    "MDSC": {
        "language": "zh",
        "mfa_acoustic": "mandarin_mfa",
        "mfa_dict": "mandarin_mfa",
        "content_filter": {"sentence", "sentences", "command", "connected_speech", "unknown"},
        "text_source": "manifest",
    },
    "IPVS": {
        "language": "it",
        "mfa_acoustic": "italian_cv",
        "mfa_dict": "italian_cv",
        "content_filter": {"sentence", "sentences", "spontaneous", "connected_speech", "unknown"},
        "text_source": "manifest",  # requires ASR pseudo-labelling first
        "notes": "PD; text empty in original manifest — pseudo-label with ASR before alignment",
    },
    # VOC-ALS: no MFA — sustained vowels only, text field empty.
    # Vowel triangle computed directly in track4_extract_features.py.
    "PC-GITA": {
        "language": "es",
        "mfa_acoustic": "spanish_mfa",
        "mfa_dict": "spanish_mfa",
        "content_filter": {"sentence", "sentences", "read_text", "monologue",
                           "connected_speech", "unknown"},
        "text_source": "manifest",
        "notes": "PD; also has sustained vowels for direct vowel triangle (no MFA needed)",
    },
    "UASPEECH": {
        "language": "en",
        "mfa_acoustic": "english_mfa",
        "mfa_dict": "english_mfa",
        "content_filter": {"word", "single_word", "words", "sentence", "sentences", "unknown"},
        "text_source": "manifest",
        "notes": "Word-only; used for vowel triangle analysis",
    },
    "UASPEECH_control": {
        "language": "en",
        "mfa_acoustic": "english_mfa",
        "mfa_dict": "english_mfa",
        "content_filter": {"word", "single_word", "words", "sentence", "sentences", "unknown"},
        "text_source": "manifest",
        "notes": "HC for UASPEECH; word-only",
    },
    "LibriSpeech_English": {
        "language": "en",
        "mfa_acoustic": "english_mfa",
        "mfa_dict": "english_mfa",
        "content_filter": {"sentence", "sentences", "read_speech", "connected_speech", "unknown"},
        "text_source": "manifest",
        "max_speakers": 150,
        "max_samples_per_speaker": 10,
    },
}


def find_mfa_binary(override: Optional[str] = None) -> str:
    """Find usable MFA binary."""
    if override:
        if os.path.isfile(override) or shutil.which(override):
            return override
        print(f"WARNING: Specified MFA binary not found: {override}")

    for candidate in MFA_BIN_CANDIDATES:
        expanded = os.path.expanduser(candidate)
        if os.path.isfile(expanded):
            return expanded
        if shutil.which(candidate):
            return candidate

    print("ERROR: MFA binary not found. Install with:")
    print("  conda create -n mfa -c conda-forge montreal-forced-aligner -y")
    sys.exit(1)


def load_manifest(dataset_name: str) -> List[Dict]:
    """Load manifest_v2.jsonl for a dataset."""
    dir_candidates = [dataset_name, dataset_name.replace("_", "-"),
                      dataset_name.replace("-", "_")]
    for candidate in dir_candidates:
        p = PROCESSED_DIR / candidate / "manifest_v2.jsonl"
        if p.exists():
            samples = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return samples
    return []


def load_inventory_speakers(dataset_name: str) -> Set[str]:
    """Load speaker IDs from inventory CSV for this dataset."""
    speakers = set()
    if not INVENTORY_CSV.exists():
        return speakers
    with open(INVENTORY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] == dataset_name:
                speakers.add(row["speaker_id"])
    return speakers


def resolve_audio_path(audio_filepath: str, dataset_dir: Path) -> Path:
    """Resolve audio path from manifest."""
    if os.path.isabs(audio_filepath):
        return Path(audio_filepath)
    return dataset_dir / audio_filepath


def prepare_corpus(dataset_name: str, config: dict, samples: List[Dict]) -> Path:
    """Create MFA corpus directory: speaker_dir/file.wav + speaker_dir/file.lab"""
    corpus_path = CORPUS_DIR / dataset_name
    corpus_path.mkdir(parents=True, exist_ok=True)

    content_filter = config.get("content_filter", set())
    severity_filter = config.get("severity_filter", set())
    max_speakers = config.get("max_speakers", 0)
    max_per_speaker = config.get("max_samples_per_speaker", 0)

    # Find dataset dir for path resolution
    dataset_dir = None
    for candidate in [dataset_name, dataset_name.replace("_", "-"),
                      dataset_name.replace("-", "_")]:
        p = PROCESSED_DIR / candidate
        if p.exists():
            dataset_dir = p
            break

    if dataset_dir is None:
        print(f"  ERROR: Dataset dir not found for {dataset_name}")
        return corpus_path

    # Group by speaker
    by_speaker = defaultdict(list)
    for s in samples:
        # Filter by severity (e.g., SAP: only rated speakers)
        if severity_filter and s.get("severity", "unknown") not in severity_filter:
            continue
        ct = s.get("content_type", "unknown").lower()
        if content_filter and ct not in content_filter:
            continue
        text = s.get("text", "").strip()
        if not text or len(text) < 3:
            continue
        spk_id = s.get("speaker_id", "unknown")
        by_speaker[spk_id].append(s)

    # Limit speakers if needed
    speaker_ids = sorted(by_speaker.keys())
    if max_speakers > 0:
        speaker_ids = speaker_ids[:max_speakers]

    total_prepared = 0
    total_skipped = 0

    for spk_id in speaker_ids:
        spk_samples = by_speaker[spk_id]
        if max_per_speaker > 0:
            spk_samples = spk_samples[:max_per_speaker]

        # Sanitize speaker dir name (remove problematic characters)
        safe_spk = spk_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        spk_dir = corpus_path / safe_spk
        spk_dir.mkdir(exist_ok=True)

        for i, s in enumerate(spk_samples):
            audio_filepath = s.get("audio_filepath", "")
            audio_path = resolve_audio_path(audio_filepath, dataset_dir)

            if not audio_path.exists():
                total_skipped += 1
                continue

            # Use a clean filename
            stem = audio_path.stem
            # Ensure unique names within speaker dir
            wav_dest = spk_dir / f"{stem}.wav"
            lab_dest = spk_dir / f"{stem}.lab"

            # Handle duplicates
            if wav_dest.exists():
                wav_dest = spk_dir / f"{stem}_{i}.wav"
                lab_dest = spk_dir / f"{stem}_{i}.lab"

            # Symlink or copy audio
            try:
                if not wav_dest.exists():
                    os.symlink(str(audio_path), str(wav_dest))
            except (OSError, NotImplementedError):
                if not wav_dest.exists():
                    shutil.copy2(str(audio_path), str(wav_dest))

            # Write text lab file
            text = s.get("text", "").strip()
            # Clean text for MFA: remove punctuation except apostrophes
            text_clean = text
            for ch in ".,;:!?\"()[]{}—–-":
                text_clean = text_clean.replace(ch, "")
            text_clean = " ".join(text_clean.split())  # normalize whitespace

            lab_dest.write_text(text_clean, encoding="utf-8")
            total_prepared += 1

    print(f"  Prepared {total_prepared} files for {len(speaker_ids)} speakers "
          f"(skipped {total_skipped} missing audio)")
    return corpus_path


def run_mfa_align(mfa_bin: str, corpus_path: Path, config: dict,
                  dataset_name: str, num_jobs: int = 4) -> Path:
    """Run MFA alignment on prepared corpus."""
    output_dir = ALIGNED_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    acoustic_model = config["mfa_acoustic"]
    dictionary = config["mfa_dict"]

    cmd = [
        mfa_bin, "align",
        str(corpus_path),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--num_jobs", str(num_jobs),
        "--clean",
        "--overwrite",
    ]

    # Add GPU flag if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.append("--use_gpu")
    except ImportError:
        pass

    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  MFA stderr: {result.stderr[:500]}")
            # Still continue — partial results are useful
        else:
            print(f"  MFA completed successfully")
    except subprocess.TimeoutExpired:
        print(f"  MFA timed out after 1 hour")
    except FileNotFoundError:
        print(f"  ERROR: MFA binary not found: {mfa_bin}")
        return output_dir

    # Count output TextGrids
    tg_count = len(list(output_dir.rglob("*.TextGrid")))
    print(f"  Output: {tg_count} TextGrid files in {output_dir}")
    return output_dir


def check_alignment_quality(dataset_name: str, corpus_path: Path, aligned_path: Path):
    """Report alignment success rate."""
    # Count input files
    input_wavs = set()
    for wav in corpus_path.rglob("*.wav"):
        input_wavs.add(wav.stem)

    # Count output TextGrids
    output_tgs = set()
    for tg in aligned_path.rglob("*.TextGrid"):
        output_tgs.add(tg.stem)

    aligned = len(input_wavs & output_tgs)
    failed = len(input_wavs - output_tgs)
    total = len(input_wavs)

    pct = (aligned / total * 100) if total > 0 else 0
    print(f"\n  Alignment quality for {dataset_name}:")
    print(f"    Input: {total} utterances")
    print(f"    Aligned: {aligned} ({pct:.1f}%)")
    print(f"    Failed: {failed}")

    if pct < 80:
        print(f"    WARNING: Below 80% threshold!")

    return {"dataset": dataset_name, "total": total, "aligned": aligned,
            "failed": failed, "pct": pct}


def main():
    parser = argparse.ArgumentParser(description="Track 4: MFA alignment pipeline")
    parser.add_argument("--dataset", type=str, help="Single dataset to align")
    parser.add_argument("--all", action="store_true", help="Align all datasets")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare corpus, don't align")
    parser.add_argument("--mfa-bin", type=str, default=None, help="Path to MFA binary")
    parser.add_argument("--num-jobs", type=int, default=4, help="MFA parallel jobs")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Specify --dataset NAME or --all")

    datasets = list(DATASET_ALIGN_CONFIG.keys()) if args.all else [args.dataset]

    # Validate datasets
    for ds in datasets:
        if ds not in DATASET_ALIGN_CONFIG:
            print(f"ERROR: Unknown dataset '{ds}'. Available: {list(DATASET_ALIGN_CONFIG.keys())}")
            sys.exit(1)

    mfa_bin = None
    if not args.prepare_only:
        mfa_bin = find_mfa_binary(args.mfa_bin)
        print(f"MFA binary: {mfa_bin}")

    print("=" * 80)
    print("Track 4: MFA Alignment Pipeline")
    print("=" * 80)
    print(f"Base dir: {BASE_DIR}")
    print(f"Datasets: {datasets}")

    quality_reports = []

    for dataset_name in datasets:
        config = DATASET_ALIGN_CONFIG[dataset_name]
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name} ({config['language']})")
        print(f"{'='*60}")

        # Load manifest
        samples = load_manifest(dataset_name)
        if not samples:
            print(f"  No samples found, skipping")
            continue
        print(f"  Loaded {len(samples)} samples from manifest")

        # Prepare corpus
        corpus_path = prepare_corpus(dataset_name, config, samples)

        if args.prepare_only:
            continue

        # Run alignment
        aligned_path = run_mfa_align(mfa_bin, corpus_path, config, dataset_name,
                                     num_jobs=args.num_jobs)

        # Check quality
        quality = check_alignment_quality(dataset_name, corpus_path, aligned_path)
        quality_reports.append(quality)

    # Summary
    if quality_reports:
        print("\n" + "=" * 80)
        print("ALIGNMENT SUMMARY")
        print("=" * 80)
        print(f"{'Dataset':<25} {'Total':>8} {'Aligned':>8} {'Failed':>8} {'Rate':>8}")
        print("-" * 60)
        for q in quality_reports:
            print(f"{q['dataset']:<25} {q['total']:>8} {q['aligned']:>8} "
                  f"{q['failed']:>8} {q['pct']:>7.1f}%")

        # Save quality report
        report_path = RESULTS_DIR / "alignment_quality.json"
        with open(report_path, "w") as f:
            json.dump(quality_reports, f, indent=2)
        print(f"\nQuality report saved to {report_path}")


if __name__ == "__main__":
    main()
