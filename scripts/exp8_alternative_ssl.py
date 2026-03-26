#!/usr/bin/env python3
"""
Experiment 8: Alternative SSL backbone.

Run the full Track 4 pipeline on TORGO using WavLM-base instead of HuBERT-base.
Shows the effect is not HuBERT-specific.

Also tests: wav2vec2-base for a third comparison point.

Uses existing TORGO TextGrids and corpus from C:/temp/track4_corpus/TORGO
and C:/temp/track4_aligned/TORGO (or the PoC MFA aligned data).

Runs locally on Windows RTX A4000.
"""

import csv
import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import soundfile as sf
from tqdm import tqdm
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "reviewer_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# TORGO data paths — override with env vars or adjust for your setup
CORPUS_DIR = Path(os.environ.get("TORGO_CORPUS", str(RESULTS_DIR.parent.parent / "results" / "track4" / "corpus" / "TORGO")))
ALIGNED_DIR = Path(os.environ.get("TORGO_ALIGNED", str(RESULTS_DIR.parent.parent / "results" / "track4" / "aligned" / "TORGO")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
HUBERT_FPS = 50.0
MIN_TOKENS = 5

# TORGO speaker metadata
SPEAKERS = {
    "FC01": {"severity": "control", "aetiology": "healthy", "sev_num": 0},
    "FC02": {"severity": "control", "aetiology": "healthy", "sev_num": 0},
    "FC03": {"severity": "control", "aetiology": "healthy", "sev_num": 0},
    "MC01": {"severity": "control", "aetiology": "healthy", "sev_num": 0},
    "MC02": {"severity": "control", "aetiology": "healthy", "sev_num": 0},
    "MC03": {"severity": "control", "aetiology": "healthy", "sev_num": 0},
    "MC04": {"severity": "control", "aetiology": "healthy", "sev_num": 0},
    "F01":  {"severity": "severe", "aetiology": "cerebral_palsy", "sev_num": 3},
    "F03":  {"severity": "mild", "aetiology": "cerebral_palsy", "sev_num": 1},
    "F04":  {"severity": "moderate", "aetiology": "cerebral_palsy", "sev_num": 2},
    "M01":  {"severity": "severe", "aetiology": "cerebral_palsy", "sev_num": 3},
    "M02":  {"severity": "severe", "aetiology": "cerebral_palsy", "sev_num": 3},
    "M03":  {"severity": "mild", "aetiology": "cerebral_palsy", "sev_num": 1},
    "M04":  {"severity": "severe", "aetiology": "cerebral_palsy", "sev_num": 3},
    "M05":  {"severity": "moderate", "aetiology": "cerebral_palsy", "sev_num": 2},
}

# Models to compare
MODELS = {
    "hubert-base": "facebook/hubert-base-ls960",
    "wavlm-base": "microsoft/wavlm-base",
    "wav2vec2-base": "facebook/wav2vec2-base-960h",
}


def load_phone_config():
    config_path = CONFIG_DIR / "phone_features_en.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_textgrid(path: Path) -> List[Dict]:
    content = path.read_text(encoding="utf-8")
    phones = []
    in_phones_tier = False
    current = {}
    for line in content.split("\n"):
        line = line.strip()
        if '"phones"' in line:
            in_phones_tier = True
            continue
        if in_phones_tier:
            if line.startswith("xmin ="):
                current["xmin"] = float(line.split("=")[1].strip())
            elif line.startswith("xmax ="):
                current["xmax"] = float(line.split("=")[1].strip())
            elif line.startswith("text ="):
                text = line.split("=")[1].strip().strip('"')
                current["text"] = text
                if text and text not in ("", "spn", "sil", "sp"):
                    phones.append(dict(current))
                current = {}
            elif '"' in line and "name" in line.lower() and phones:
                break
    return phones


def classify_phone(phone, config):
    cats = {}
    for feat_name, feat_def in config.get("consonant_features", {}).items():
        if phone in set(feat_def["positive"]):
            cats[feat_name] = "positive"
        elif phone in set(feat_def["negative"]):
            cats[feat_name] = "negative"
    return cats


def load_ssl_model(model_name, model_id):
    """Load any HuggingFace SSL model that outputs hidden states."""
    from transformers import AutoModel, AutoFeatureExtractor

    print(f"  Loading {model_name} ({model_id})...")
    model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
    try:
        extractor = AutoFeatureExtractor.from_pretrained(model_id)
    except Exception:
        from transformers import Wav2Vec2FeatureExtractor
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    return model, extractor


def extract_frames(model, extractor, audio_path):
    """Extract frame-level embeddings from any SSL model."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.to(DEVICE))
        frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    return frames


def compute_direction(embs_pos, embs_neg):
    if len(embs_pos) < MIN_TOKENS or len(embs_neg) < MIN_TOKENS:
        return None
    d = embs_pos.mean(0) - embs_neg.mean(0)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-8 else None


def compute_dprime(embs_pos, embs_neg, direction):
    if len(embs_pos) < MIN_TOKENS or len(embs_neg) < MIN_TOKENS:
        return float("nan")
    proj_pos = embs_pos @ direction
    proj_neg = embs_neg @ direction
    gap = proj_pos.mean() - proj_neg.mean()
    pooled_std = np.sqrt((proj_pos.var() + proj_neg.var()) / 2)
    return float(abs(gap / pooled_std)) if pooled_std > 1e-8 else 0.0


def run_pipeline(model_name, model_id, config):
    """Run the full Track 4 pipeline with a specific SSL model."""
    model, extractor = load_ssl_model(model_name, model_id)

    # Extract phone embeddings for all speakers
    speaker_phones = {}
    for spk_name in SPEAKERS:
        spk_aligned = ALIGNED_DIR / spk_name
        spk_corpus = CORPUS_DIR / spk_name
        if not spk_aligned.exists():
            continue

        textgrids = list(spk_aligned.glob("*.TextGrid"))
        all_embs = []
        for tg in textgrids:
            phones = parse_textgrid(tg)
            wav = spk_corpus / tg.with_suffix(".wav").name
            if not wav.exists():
                continue
            try:
                frames = extract_frames(model, extractor, wav)
            except Exception:
                continue

            for phone in phones:
                start = int(phone["xmin"] * HUBERT_FPS)
                end = int(phone["xmax"] * HUBERT_FPS)
                if end <= start:
                    end = start + 1
                if start >= len(frames):
                    continue
                end = min(end, len(frames))
                emb = frames[start:end].mean(axis=0)
                all_embs.append({"phone": phone["text"], "embedding": emb})

        if all_embs:
            speaker_phones[spk_name] = all_embs

    # Compute feature directions from controls
    ctrl_by_feat = defaultdict(lambda: defaultdict(list))
    for spk, phone_embs in speaker_phones.items():
        if SPEAKERS[spk]["severity"] != "control":
            continue
        for pe in phone_embs:
            cats = classify_phone(pe["phone"], config)
            for feat, cls in cats.items():
                ctrl_by_feat[feat][cls].append(pe["embedding"])

    directions = {}
    for feat in sorted(ctrl_by_feat.keys()):
        pos = np.array(ctrl_by_feat[feat].get("positive", []))
        neg = np.array(ctrl_by_feat[feat].get("negative", []))
        d = compute_direction(pos, neg)
        if d is not None:
            directions[feat] = d

    # Compute per-speaker d-prime
    results = []
    for spk in sorted(SPEAKERS):
        if spk not in speaker_phones:
            continue
        meta = SPEAKERS[spk]
        phone_embs = speaker_phones[spk]

        by_feat = defaultdict(lambda: defaultdict(list))
        for pe in phone_embs:
            cats = classify_phone(pe["phone"], config)
            for feat, cls in cats.items():
                by_feat[feat][cls].append(pe["embedding"])

        metrics = {}
        for feat, direction in directions.items():
            pos = np.array(by_feat[feat].get("positive", [])) if by_feat[feat].get("positive") else np.zeros((0, direction.shape[0]))
            neg = np.array(by_feat[feat].get("negative", [])) if by_feat[feat].get("negative") else np.zeros((0, direction.shape[0]))
            metrics[feat + "_dprime"] = compute_dprime(pos, neg, direction)

        results.append({
            "speaker": spk,
            "severity": meta["severity"],
            "sev_num": meta["sev_num"],
            **metrics,
        })

    # Compute Spearman correlations
    correlations = {}
    for feat in directions:
        col = feat + "_dprime"
        vals = [(r["sev_num"], r[col]) for r in results if not np.isnan(r.get(col, float("nan")))]
        if len(vals) >= 5:
            sevs, dps = zip(*vals)
            rho, p = stats.spearmanr(sevs, dps)
            correlations[feat] = {"rho": float(rho), "p": float(p), "n": len(vals)}

    return results, correlations


def main():
    print("=" * 70)
    print("EXPERIMENT 8: Alternative SSL Backbone Comparison")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Aligned: {ALIGNED_DIR}")

    if not CORPUS_DIR.exists():
        print(f"ERROR: TORGO corpus not found at {CORPUS_DIR}")
        print("Need to copy from DGX first")
        sys.exit(1)

    if not ALIGNED_DIR.exists():
        print(f"ERROR: TORGO aligned data not found at {ALIGNED_DIR}")
        sys.exit(1)

    config = load_phone_config()

    all_correlations = {}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        results, correlations = run_pipeline(model_name, model_id, config)
        all_correlations[model_name] = correlations

        print(f"\n  Speakers processed: {len(results)}")
        print(f"  Features computed: {list(correlations.keys())}")
        print(f"\n  {'Feature':<20} {'rho':>8} {'p':>12}")
        print(f"  {'-'*45}")
        for feat, c in sorted(correlations.items()):
            star = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
            print(f"  {feat:<20} {c['rho']:>8.4f} {c['p']:>12.4e} {star}")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: Spearman rho across SSL models (TORGO, n=15)")
    print("=" * 70)
    features = sorted(set(f for c in all_correlations.values() for f in c))

    header = f"{'Feature':<20}"
    for model_name in MODELS:
        header += f" {model_name:>15}"
    print(header)
    print("-" * 70)

    for feat in features:
        row = f"{feat:<20}"
        for model_name in MODELS:
            c = all_correlations.get(model_name, {}).get(feat)
            if c:
                star = "*" if c["p"] < 0.05 else ""
                row += f" {c['rho']:>14.4f}{star}"
            else:
                row += f" {'--':>15}"
        print(row)

    # Save
    output_path = RESULTS_DIR / "exp8_alternative_ssl.json"
    with open(output_path, "w") as f:
        json.dump(all_correlations, f, indent=2)
    print(f"\nSaved to {output_path}")
    print("\nDONE")


if __name__ == "__main__":
    main()
