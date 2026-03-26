#!/usr/bin/env python3
"""
Track 4: Vowel triangle extraction from sustained vowel files.

Runs locally on Windows (RTX A4000). No MFA needed — each audio file
contains a single sustained vowel identified by filename or text field.

Datasets: VOC-ALS, PC-GITA, IPVS
"""

import csv
import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import soundfile as sf
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

VOWEL_DIR = Path(os.environ.get("VOWEL_DIR", str(Path(__file__).resolve().parent.parent / "data" / "vowels")))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = RESULTS_DIR / "vowel_triangle_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000

# Dataset configs: how to identify vowels
DATASETS = {
    "VOC-ALS": {
        "language": "it",
        "vowel_patterns": {"phonationA": "a", "phonationI": "i", "phonationU": "u"},
        "match_mode": "contains",  # pattern appears anywhere in filename
        "content_filter": None,  # use all matching files
    },
    "PC-GITA": {
        "language": "es",
        "vowel_patterns": {"a": "a", "i": "i", "u": "u"},
        "match_mode": "text",  # match on manifest text field
        "content_filter": "vowel",
    },
    "IPVS": {
        "language": "it",
        "vowel_patterns": {"VA": "a", "VI": "i", "VU": "u"},
        "match_mode": "startswith",  # filename starts with pattern
        "content_filter": "vowel",
    },
}


def load_hubert():
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    model_id = "facebook/hubert-base-ls960"
    print(f"Loading HuBERT from {model_id}...")
    model = HubertModel.from_pretrained(model_id).to(DEVICE).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    print(f"  Device: {DEVICE}")
    return model, extractor


def extract_embedding(model, extractor, audio_path: Path) -> np.ndarray:
    """Extract mean-pooled HuBERT embedding for entire audio file."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.to(DEVICE))
        frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    return frames.mean(axis=0)


def compute_vowel_triangle(embeddings: Dict[str, List[np.ndarray]]) -> float:
    """Compute vowel triangle area from /a/, /i/, /u/ embeddings."""
    centroids = {}
    for v in ["a", "i", "u"]:
        if v in embeddings and len(embeddings[v]) >= 1:
            centroids[v] = np.mean(embeddings[v], axis=0)
    if len(centroids) < 3:
        return float("nan")
    pts = [centroids["i"], centroids["a"], centroids["u"]]
    a = np.linalg.norm(pts[1] - pts[0])
    b = np.linalg.norm(pts[2] - pts[1])
    c = np.linalg.norm(pts[0] - pts[2])
    s = (a + b + c) / 2
    return float(np.sqrt(max(0, s * (s - a) * (s - b) * (s - c))))


def process_dataset(dataset_name: str, config: dict, model, extractor) -> List[Dict]:
    """Process one vowel dataset, return per-speaker results."""
    ds_dir = VOWEL_DIR / dataset_name
    manifest_path = ds_dir / "manifest_v2.jsonl"

    if not manifest_path.exists():
        print(f"  No manifest found for {dataset_name}")
        return []

    # Load manifest
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Group by speaker, filter to vowels
    speaker_vowels = defaultdict(lambda: defaultdict(list))  # spk -> vowel -> [embeddings]
    speaker_meta = {}
    skipped = 0

    vowel_samples = []
    for s in samples:
        stem = Path(s.get("audio_filepath", "")).stem
        text = s.get("text", "").strip().lower()
        ct = s.get("content_type", "").lower()

        # Filter by content type if specified
        if config["content_filter"] and ct != config["content_filter"]:
            continue

        # Match vowel identity
        vowel = None
        for pattern, ipa in config["vowel_patterns"].items():
            if config["match_mode"] == "contains" and pattern in stem:
                vowel = ipa
                break
            elif config["match_mode"] == "startswith" and stem.startswith(pattern):
                vowel = ipa
                break
            elif config["match_mode"] == "text" and text == pattern:
                vowel = ipa
                break

        if vowel is None:
            continue

        # Find audio file locally
        spk = s.get("speaker_id", "unknown")
        audio_name = Path(s.get("audio_filepath", "")).name
        audio_path = ds_dir / spk / audio_name

        if not audio_path.exists():
            # Try without speaker subdir
            matches = list(ds_dir.rglob(audio_name))
            if matches:
                audio_path = matches[0]
            else:
                skipped += 1
                continue

        vowel_samples.append((s, vowel, audio_path))
        if spk not in speaker_meta:
            speaker_meta[spk] = {
                "severity": s.get("severity", "unknown"),
                "aetiology": s.get("aetiology", "unknown"),
                "is_dysarthric": s.get("is_dysarthric", True),
            }

    print(f"  {len(vowel_samples)} vowel files to process, {skipped} skipped (not found)")

    # Extract embeddings
    for s, vowel, audio_path in tqdm(vowel_samples, desc=f"  {dataset_name}"):
        spk = s.get("speaker_id", "unknown")
        try:
            emb = extract_embedding(model, extractor, audio_path)
            speaker_vowels[spk][vowel].append(emb)
        except Exception as e:
            print(f"    Error {audio_path.name}: {e}")

    # Compute per-speaker vowel triangle
    results = []
    for spk in sorted(speaker_vowels.keys()):
        vowels = speaker_vowels[spk]
        area = compute_vowel_triangle(vowels)
        meta = speaker_meta.get(spk, {})

        results.append({
            "dataset": dataset_name,
            "speaker_id": spk,
            "language": config["language"],
            "aetiology": meta.get("aetiology", "unknown"),
            "severity_label": meta.get("severity", "unknown"),
            "is_control": not meta.get("is_dysarthric", True),
            "vowel_triangle_area": area,
            "n_a": len(vowels.get("a", [])),
            "n_i": len(vowels.get("i", [])),
            "n_u": len(vowels.get("u", [])),
        })

    return results


def main():
    print("=" * 60)
    print("Track 4: Vowel Triangle Extraction (Local)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    model, extractor = load_hubert()

    all_results = []
    for ds_name, config in DATASETS.items():
        print(f"\n--- {ds_name} ({config['language']}) ---")
        results = process_dataset(ds_name, config, model, extractor)
        all_results.extend(results)
        print(f"  {len(results)} speakers processed")

    # Write CSV
    if all_results:
        fieldnames = ["dataset", "speaker_id", "language", "aetiology",
                      "severity_label", "is_control", "vowel_triangle_area",
                      "n_a", "n_i", "n_u"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nWrote {len(all_results)} results to {OUTPUT_CSV}")

    # Summary
    print("\n" + "=" * 60)
    print("VOWEL TRIANGLE RESULTS BY SEVERITY")
    print("=" * 60)

    for ds_name in DATASETS:
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        if not ds_results:
            continue
        print(f"\n  {ds_name}:")
        print(f"  {'Severity':<12} {'N':>4} {'Mean Area':>12} {'Std':>10}")
        print(f"  {'-'*42}")
        for sev in ["control", "mild", "moderate", "severe"]:
            sev_r = [r for r in ds_results if r["severity_label"] == sev]
            if not sev_r:
                continue
            areas = [r["vowel_triangle_area"] for r in sev_r if not np.isnan(r["vowel_triangle_area"])]
            if areas:
                print(f"  {sev:<12} {len(areas):>4} {np.mean(areas):>12.4f} {np.std(areas):>10.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
