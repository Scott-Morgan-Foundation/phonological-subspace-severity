#!/usr/bin/env python3
"""
Track 4: Extract phonological feature metrics from HuBERT + MFA TextGrids.

For each speaker, computes 12 metrics:
  Consonant features (5): nasality, voicing, strident, sonorant, manner d-prime
  Vowel features (4): high, low, back, round d-prime
  Structural metrics (3): boundary_sharpness, cross_position_cosim, vowel_triangle_area

Feature directions are computed from healthy controls within each language.

Usage:
    # Extract features for a single dataset
    python scripts/track4_extract_features.py --dataset COPAS

    # Extract for all aligned datasets
    python scripts/track4_extract_features.py --all

    # Use specific HuBERT model path
    python scripts/track4_extract_features.py --all --hubert-path ~/dysarthria/models/hubert-base-ls960

Runs on DGX Spark (GPU recommended for HuBERT).
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import soundfile as sf
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parent.parent))
RESULTS_DIR = BASE_DIR / "results" / "track4"
ALIGNED_DIR = RESULTS_DIR / "aligned"
CORPUS_DIR = RESULTS_DIR / "corpus"
CONFIG_DIR = Path(os.environ.get("TRACK4_CONFIG_DIR",
                                  str(Path(__file__).resolve().parent.parent / "config")))
INVENTORY_CSV = RESULTS_DIR / "speaker_inventory.csv"
OUTPUT_CSV = RESULTS_DIR / "track4_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
HUBERT_FPS = 50.0  # HuBERT frame rate: 20ms per frame
MIN_TOKENS_PER_FEATURE = 5  # Minimum tokens per class for a valid measurement

# Language → dataset mapping for HC feature direction computation
LANGUAGE_HC_DATASETS = {
    "en": ["LibriSpeech_English", "TORGO", "UASPEECH_control"],
    "nl": ["COPAS"],
    "fr": ["YouTube_French"],
    "es": ["Neurovoz", "PC-GITA"],
    "zh": ["MDSC"],
    "it": ["IPVS", "VOC-ALS"],
    "sk": ["EWA-DB"],
    "hu": ["Hungarian_HC", "CV_Hungarian"],
    "de": ["YouTube_German", "SVD"],
    "ta": ["SLR65_Tamil", "SSNCE_Tamil"],
    "pt": ["AVFAD"],
    "sw": ["CDLI_Kenyan_Swahili"],
    "el": ["Kostas_Greek_reading", "Kostas_Greek_v3_reading"],
}

# Dataset → language mapping
DATASET_LANGUAGE = {
    "SAP": "en", "COPAS": "nl", "TORGO": "en", "Neurovoz": "es",
    "YouTube_French": "fr", "MDSC": "zh", "IPVS": "it",
    "VOC-ALS": "it", "PC-GITA": "es",
    "UASPEECH": "en", "UASPEECH_control": "en",
    "LibriSpeech_English": "en",
    "EWA-DB": "sk",
    "Hungarian_Dysarthria": "hu",
    "Hungarian_HC": "hu",
    "CDSD": "zh",
    "YouTube_German": "de",
    "SVD": "de",
    "EasyCall": "it",
    "Domotica": "nl",
    "SSNCE_Tamil": "ta",
    "SLR65_Tamil": "ta",
    "AVFAD": "pt",
    "CDLI_Kenyan_Swahili": "sw",
    "CHASING": "nl",
    "TreasureHunters1": "nl",
    "SAP_all": "en",
    "SAP_unlabeled": "en",
    "TORGO_FC01": "en",
    "CV_Hungarian": "hu",
    "Kostas_Greek": "el",
    "Kostas_Greek_v2": "el",
    "Kostas_Greek_reading": "el",
    "Kostas_Greek_v3_reading": "el",
}

# Datasets where vowel triangle is computed directly from sustained vowel files
# (no MFA needed — one vowel per audio file, identified by filename pattern)
DIRECT_VOWEL_DATASETS = {
    "VOC-ALS": {
        # filename pattern -> IPA vowel
        "phonationA": "a", "phonationI": "i", "phonationU": "u",
    },
    "PC-GITA": {
        # text field contains the vowel identity
        "a": "a", "i": "i", "u": "u",
    },
    "IPVS": {
        # filename prefix -> IPA vowel (VA=a, VI=i, VU=u)
        "VA": "a", "VI": "i", "VU": "u",
    },
}


# =============================================================================
# TextGrid Parsing
# =============================================================================

def parse_textgrid(path: Path) -> List[Dict]:
    """Parse Praat TextGrid file, return phone intervals."""
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
            # Stop if we hit another tier
            elif '"' in line and "name" in line.lower() and in_phones_tier and phones:
                break

    return phones


# =============================================================================
# Phone Feature Mapping
# =============================================================================

class PhoneFeatureMapper:
    """Maps IPA phones to phonological features using language-specific configs."""

    def __init__(self, config_dir: Path):
        self.configs = {}
        self.loaded_langs = set()
        self.config_dir = config_dir

    def load_language(self, lang: str):
        """Load phone feature config for a language."""
        if lang in self.loaded_langs:
            return
        config_path = self.config_dir / f"phone_features_{lang}.json"
        if not config_path.exists():
            print(f"  WARNING: No phone feature config for '{lang}' at {config_path}")
            self.configs[lang] = {"consonant_features": {}, "vowel_features": {},
                                  "all_consonants": [], "all_vowels": []}
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                self.configs[lang] = json.load(f)
        self.loaded_langs.add(lang)

    def classify_phone(self, phone: str, lang: str) -> Dict[str, str]:
        """Classify a phone into feature categories for a given language."""
        self.load_language(lang)
        config = self.configs[lang]
        cats = {}

        # Consonant features
        for feat_name, feat_def in config.get("consonant_features", {}).items():
            pos = set(feat_def["positive"])
            neg = set(feat_def["negative"])
            if phone in pos:
                cats[feat_name] = "positive"
            elif phone in neg:
                cats[feat_name] = "negative"

        # Vowel features
        for feat_name, feat_def in config.get("vowel_features", {}).items():
            pos = set(feat_def["positive"])
            neg = set(feat_def["negative"])
            if phone in pos:
                cats[f"vowel_{feat_name}"] = "positive"
            elif phone in neg:
                cats[f"vowel_{feat_name}"] = "negative"

        return cats

    def is_consonant(self, phone: str, lang: str) -> bool:
        self.load_language(lang)
        return phone in set(self.configs[lang].get("all_consonants", []))

    def is_vowel(self, phone: str, lang: str) -> bool:
        self.load_language(lang)
        return phone in set(self.configs[lang].get("all_vowels", []))

    def get_vowel_triangle_phones(self, lang: str) -> List[str]:
        self.load_language(lang)
        return self.configs[lang].get("vowel_triangle", ["i", "a", "u"])


# =============================================================================
# HuBERT
# =============================================================================

SSL_MODELS = {
    "hubert": "facebook/hubert-base-ls960",
    "hubert-large": "facebook/hubert-large-ls960-ft",
    "wav2vec2": "facebook/wav2vec2-base",
    "xlsr": "facebook/wav2vec2-xls-r-300m",
    "mms": "facebook/mms-300m",
    "wavlm": "microsoft/wavlm-base",
}


def load_ssl_model(model_name: str = "hubert", model_path: Optional[str] = None):
    """Load SSL model and feature extractor."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    if model_path is None:
        # Check local cache first for HuBERT
        if model_name == "hubert":
            local_path = BASE_DIR / "models" / "hubert-base-ls960"
            if local_path.exists():
                model_path = str(local_path)
        if model_path is None:
            model_path = SSL_MODELS.get(model_name, model_name)

    print(f"Loading {model_name} from {model_path}...")
    model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    print(f"  Device: {DEVICE}, hidden_size: {hidden_size}")
    return model, extractor


def load_hubert(model_path: Optional[str] = None):
    """Load HuBERT model — backward compatibility wrapper."""
    return load_ssl_model("hubert", model_path)


def extract_frame_embeddings(model, extractor, audio_path: Path) -> Optional[np.ndarray]:
    """Extract HuBERT frame-level embeddings for an audio file."""
    try:
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import torchaudio
            audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(
                torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()

        inputs = extractor(audio, sampling_rate=SAMPLE_RATE,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(inputs.input_values.to(DEVICE))
            frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (T, 768)
        return frames
    except Exception as e:
        print(f"    Error extracting {audio_path.name}: {e}")
        return None


def get_phone_embedding(frames: np.ndarray, xmin: float, xmax: float) -> Optional[np.ndarray]:
    """Extract mean-pooled embedding for a phone interval."""
    start_frame = int(xmin * HUBERT_FPS)
    end_frame = int(xmax * HUBERT_FPS)
    if end_frame <= start_frame:
        end_frame = start_frame + 1
    if start_frame >= len(frames):
        return None
    end_frame = min(end_frame, len(frames))
    return frames[start_frame:end_frame].mean(axis=0)


# =============================================================================
# Feature Extraction per Speaker
# =============================================================================

def extract_speaker_phone_embeddings(
    model, extractor, speaker_id: str, dataset_name: str
) -> List[Dict]:
    """Extract phone embeddings for all aligned utterances of a speaker."""
    safe_spk = speaker_id.replace("/", "_").replace("\\", "_").replace(" ", "_")

    # Find TextGrids
    aligned_spk_dir = ALIGNED_DIR / dataset_name / safe_spk
    corpus_spk_dir = CORPUS_DIR / dataset_name / safe_spk

    if not aligned_spk_dir.exists():
        return []

    textgrids = list(aligned_spk_dir.glob("*.TextGrid"))
    if not textgrids:
        return []

    all_phone_embs = []

    for tg_path in textgrids:
        phones = parse_textgrid(tg_path)
        wav_path = corpus_spk_dir / tg_path.with_suffix(".wav").name
        if not wav_path.exists():
            # Try resolving symlink target
            continue

        frames = extract_frame_embeddings(model, extractor, wav_path)
        if frames is None:
            continue

        for i, phone in enumerate(phones):
            emb = get_phone_embedding(frames, phone["xmin"], phone["xmax"])
            if emb is not None:
                all_phone_embs.append({
                    "phone": phone["text"],
                    "embedding": emb,
                    "xmin": phone["xmin"],
                    "xmax": phone["xmax"],
                    "duration": phone["xmax"] - phone["xmin"],
                    "utterance": tg_path.stem,
                    "phone_idx": i,
                    "n_phones_in_utt": len(phones),
                })

    return all_phone_embs


def extract_direct_vowel_embeddings(
    model, extractor, speaker_id: str, dataset_name: str
) -> List[Dict]:
    """Extract vowel embeddings directly from sustained vowel files (no MFA).

    For datasets like VOC-ALS (phonationA/I/U) and PC-GITA (text='a'/'i'/'u')
    where each file contains a single sustained vowel.
    """
    if dataset_name not in DIRECT_VOWEL_DATASETS:
        return []

    vowel_map = DIRECT_VOWEL_DATASETS[dataset_name]
    dataset_dir = None
    for candidate in [dataset_name, dataset_name.replace("-", "_"),
                      dataset_name.replace("_", "-")]:
        p = BASE_DIR / "data" / "processed" / candidate
        if p.exists():
            dataset_dir = p
            break
    if dataset_dir is None:
        return []

    manifest_path = dataset_dir / "manifest_v2.jsonl"
    if not manifest_path.exists():
        return []

    # Load samples for this speaker
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("speaker_id") != speaker_id:
                continue
            samples.append(r)

    vowel_embs = []
    for s in samples:
        audio_filepath = s.get("audio_filepath", "")
        audio_path = dataset_dir / audio_filepath if not os.path.isabs(audio_filepath) else Path(audio_filepath)
        if not audio_path.exists():
            continue

        # Determine vowel identity from filename or text
        filename_stem = audio_path.stem  # e.g., PZ068_phonationA
        text = s.get("text", "").strip().lower()
        vowel = None

        # Check filename patterns
        # VOC-ALS: PZ068_phonationA -> pattern 'phonationA' in stem
        # IPVS: VA1APGANRET... -> stem starts with 'VA'
        for pattern, ipa in vowel_map.items():
            if filename_stem.startswith(pattern) or pattern in filename_stem:
                vowel = ipa
                break

        # Check text field (PC-GITA: text='a' -> 'a')
        if vowel is None and text in vowel_map:
            vowel = vowel_map[text]

        if vowel is None:
            continue

        # Extract full-file mean embedding (sustained vowel = entire file is one phone)
        frames = extract_frame_embeddings(model, extractor, audio_path)
        if frames is None:
            continue

        emb = frames.mean(axis=0)
        vowel_embs.append({
            "phone": vowel,
            "embedding": emb,
            "xmin": 0.0,
            "xmax": s.get("duration", 0.0),
            "duration": s.get("duration", 0.0),
            "utterance": filename_stem,
            "phone_idx": 0,
            "n_phones_in_utt": 1,
        })

    return vowel_embs


# =============================================================================
# Metric Computation
# =============================================================================

def compute_dprime(embs_pos: np.ndarray, embs_neg: np.ndarray,
                   direction: np.ndarray) -> Dict:
    """Compute d-prime along a feature direction."""
    if len(embs_pos) < MIN_TOKENS_PER_FEATURE or len(embs_neg) < MIN_TOKENS_PER_FEATURE:
        return {"d_prime": float("nan"), "accuracy": float("nan"),
                "n_pos": len(embs_pos), "n_neg": len(embs_neg)}

    proj_pos = embs_pos @ direction
    proj_neg = embs_neg @ direction
    gap = proj_pos.mean() - proj_neg.mean()
    pooled_std = np.sqrt((proj_pos.var() + proj_neg.var()) / 2)
    d_prime = gap / pooled_std if pooled_std > 1e-8 else 0.0

    threshold = (proj_pos.mean() + proj_neg.mean()) / 2
    correct = (proj_pos > threshold).sum() + (proj_neg <= threshold).sum()
    accuracy = correct / (len(proj_pos) + len(proj_neg))

    return {"d_prime": float(abs(d_prime)), "accuracy": float(accuracy),
            "n_pos": len(embs_pos), "n_neg": len(embs_neg)}


def compute_feature_direction(embs_pos: np.ndarray, embs_neg: np.ndarray) -> Optional[np.ndarray]:
    """Compute normalized direction vector from positive to negative class."""
    if len(embs_pos) < MIN_TOKENS_PER_FEATURE or len(embs_neg) < MIN_TOKENS_PER_FEATURE:
        return None
    d = embs_pos.mean(0) - embs_neg.mean(0)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return None
    return d / norm


def compute_boundary_sharpness(phone_embs: List[Dict]) -> float:
    """Compute average cosine similarity at phone transitions within utterances.

    Low similarity at boundaries = sharp transitions (healthy).
    High similarity = blurred boundaries (dysarthric).
    """
    # Group by utterance
    by_utt = defaultdict(list)
    for pe in phone_embs:
        by_utt[pe["utterance"]].append(pe)

    boundary_sims = []
    for utt, phones in by_utt.items():
        phones_sorted = sorted(phones, key=lambda p: p["xmin"])
        for i in range(len(phones_sorted) - 1):
            emb_a = phones_sorted[i]["embedding"]
            emb_b = phones_sorted[i + 1]["embedding"]
            sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
            boundary_sims.append(sim)

    return float(np.mean(boundary_sims)) if boundary_sims else float("nan")


def compute_cross_position_cosim(phone_embs: List[Dict]) -> float:
    """Compute cross-position cosine similarity.

    For each phone, compare embedding at position i with embeddings at i-1 and i+1.
    In healthy speech, these should be orthogonal (~0 cosine sim).
    In dysarthric speech, they collapse toward each other.
    """
    by_utt = defaultdict(list)
    for pe in phone_embs:
        by_utt[pe["utterance"]].append(pe)

    cross_sims = []
    for utt, phones in by_utt.items():
        phones_sorted = sorted(phones, key=lambda p: p["xmin"])
        for i in range(1, len(phones_sorted) - 1):
            prev_emb = phones_sorted[i - 1]["embedding"]
            curr_emb = phones_sorted[i]["embedding"]
            next_emb = phones_sorted[i + 1]["embedding"]

            sim_prev = np.dot(prev_emb, curr_emb) / (
                np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb) + 1e-8)
            sim_next = np.dot(curr_emb, next_emb) / (
                np.linalg.norm(curr_emb) * np.linalg.norm(next_emb) + 1e-8)
            cross_sims.extend([sim_prev, sim_next])

    return float(np.mean(cross_sims)) if cross_sims else float("nan")


def compute_vowel_triangle_area(phone_embs: List[Dict], triangle_phones: List[str],
                                lang: str, mapper: PhoneFeatureMapper) -> float:
    """Compute vowel triangle area in HuBERT embedding space.

    Uses the 3 corner vowels (typically /i/, /a/, /u/) and computes the
    Euclidean triangle area from their mean embeddings.
    Larger area = better vowel distinction (healthy).
    """
    vowel_embs = defaultdict(list)
    for pe in phone_embs:
        if pe["phone"] in triangle_phones:
            vowel_embs[pe["phone"]].append(pe["embedding"])

    # Need all 3 corners
    centroids = {}
    for v in triangle_phones:
        if v in vowel_embs and len(vowel_embs[v]) >= 3:
            centroids[v] = np.mean(vowel_embs[v], axis=0)

    if len(centroids) < 3:
        return float("nan")

    # Triangle area via cross product (in high-dim, use 2D projection or Heron's formula)
    pts = [centroids[v] for v in triangle_phones[:3]]
    a = np.linalg.norm(pts[1] - pts[0])
    b = np.linalg.norm(pts[2] - pts[1])
    c = np.linalg.norm(pts[0] - pts[2])
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    area = np.sqrt(max(0, area_sq))

    return float(area)


def compute_speaker_metrics(
    phone_embs: List[Dict],
    feature_directions: Dict[str, np.ndarray],
    lang: str,
    mapper: PhoneFeatureMapper,
) -> Dict[str, float]:
    """Compute all 12 metrics for a speaker."""
    metrics = {}

    # Classify all phone embeddings
    by_feature_class = defaultdict(lambda: defaultdict(list))
    for pe in phone_embs:
        cats = mapper.classify_phone(pe["phone"], lang)
        for feat, cls in cats.items():
            by_feature_class[feat][cls].append(pe["embedding"])

    # d-prime for each feature that has a direction
    for feat_name, direction in feature_directions.items():
        pos_embs = by_feature_class.get(feat_name, {}).get("positive", [])
        neg_embs = by_feature_class.get(feat_name, {}).get("negative", [])

        pos_arr = np.array(pos_embs) if pos_embs else np.zeros((0, 768))
        neg_arr = np.array(neg_embs) if neg_embs else np.zeros((0, 768))

        result = compute_dprime(pos_arr, neg_arr, direction)
        col_name = feat_name.replace("vowel_", "") + "_dprime"
        metrics[col_name] = result["d_prime"]

    # Structural metrics
    metrics["boundary_sharpness"] = compute_boundary_sharpness(phone_embs)
    metrics["cross_position_cosim"] = compute_cross_position_cosim(phone_embs)

    triangle_phones = mapper.get_vowel_triangle_phones(lang)
    metrics["vowel_triangle_area"] = compute_vowel_triangle_area(
        phone_embs, triangle_phones, lang, mapper)

    return metrics


# =============================================================================
# Feature Direction Computation (from healthy controls)
# =============================================================================

def compute_hc_feature_directions(
    hc_phone_embs: List[Dict],
    lang: str,
    mapper: PhoneFeatureMapper,
) -> Dict[str, np.ndarray]:
    """Compute feature directions from healthy control phone embeddings."""
    by_feature_class = defaultdict(lambda: defaultdict(list))

    for pe in hc_phone_embs:
        cats = mapper.classify_phone(pe["phone"], lang)
        for feat, cls in cats.items():
            by_feature_class[feat][cls].append(pe["embedding"])

    directions = {}
    for feat_name in sorted(by_feature_class.keys()):
        pos_embs = by_feature_class[feat_name].get("positive", [])
        neg_embs = by_feature_class[feat_name].get("negative", [])

        pos_arr = np.array(pos_embs) if pos_embs else np.zeros((0, 768))
        neg_arr = np.array(neg_embs) if neg_embs else np.zeros((0, 768))

        direction = compute_feature_direction(pos_arr, neg_arr)
        if direction is not None:
            directions[feat_name] = direction
            print(f"    {feat_name}: {len(pos_embs)} pos, {len(neg_embs)} neg tokens")

    return directions


# =============================================================================
# Main
# =============================================================================

def load_inventory() -> List[Dict]:
    """Load speaker inventory CSV."""
    if not INVENTORY_CSV.exists():
        print(f"ERROR: Speaker inventory not found at {INVENTORY_CSV}")
        print("Run track4_speaker_inventory.py first")
        sys.exit(1)

    records = []
    with open(INVENTORY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description="Track 4: Extract phonological features")
    parser.add_argument("--dataset", type=str, help="Single dataset")
    parser.add_argument("--all", action="store_true", help="All datasets")
    parser.add_argument("--hubert-path", type=str, default=None, help="HuBERT model path")
    parser.add_argument("--model", type=str, default="hubert",
                        choices=["hubert", "hubert-large", "wav2vec2", "xlsr", "mms", "wavlm"],
                        help="SSL backbone: hubert (default), xlsr (XLS-R-300M), mms (MMS-1B), wavlm")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output CSV filename (e.g., '_xlsr')")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Save per-speaker phone embeddings as .npz for downstream analysis")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Specify --dataset NAME or --all")

    print("=" * 80)
    print(f"Track 4: Phonological Feature Extraction ({args.model})")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # Load SSL model
    model, extractor = load_ssl_model(args.model, args.hubert_path)

    # Load inventory
    inventory = load_inventory()
    print(f"Loaded {len(inventory)} speakers from inventory")

    # Phone feature mapper
    mapper = PhoneFeatureMapper(CONFIG_DIR)

    # Determine which datasets to process
    available_datasets = set()
    for d in ALIGNED_DIR.iterdir():
        if d.is_dir():
            available_datasets.add(d.name)

    if args.all:
        datasets = sorted(available_datasets)
    else:
        if args.dataset not in available_datasets:
            print(f"ERROR: No aligned data for '{args.dataset}'. Available: {available_datasets}")
            sys.exit(1)
        datasets = [args.dataset]
        # Also include HC datasets for the target language (needed for feature directions)
        lang = DATASET_LANGUAGE.get(args.dataset, "en")
        hc_datasets = LANGUAGE_HC_DATASETS.get(lang, [])
        for hc_ds in hc_datasets:
            if hc_ds in available_datasets and hc_ds not in datasets:
                datasets.append(hc_ds)
                print(f"  Auto-adding HC dataset '{hc_ds}' for {lang} feature directions")

    print(f"Datasets to process: {datasets}")

    # Group inventory by dataset
    inv_by_dataset = defaultdict(list)
    for rec in inventory:
        inv_by_dataset[rec["dataset"]].append(rec)

    # === Phase 1: Extract phone embeddings for all speakers ===
    all_phone_embs = {}  # (dataset, speaker_id) -> list of phone embs

    for dataset_name in datasets:
        lang = DATASET_LANGUAGE.get(dataset_name, "en")
        print(f"\n--- Extracting phone embeddings: {dataset_name} ({lang}) ---")

        spk_records = inv_by_dataset.get(dataset_name, [])
        if not spk_records:
            # Try to find speakers from aligned dir
            aligned_ds = ALIGNED_DIR / dataset_name
            spk_dirs = [d.name for d in aligned_ds.iterdir() if d.is_dir()]
            for spk_id in spk_dirs:
                spk_records.append({"speaker_id": spk_id, "dataset": dataset_name})

        for rec in tqdm(spk_records, desc=f"  {dataset_name}"):
            spk_id = rec["speaker_id"]
            phone_embs = extract_speaker_phone_embeddings(
                model, extractor, spk_id, dataset_name)
            if phone_embs:
                all_phone_embs[(dataset_name, spk_id)] = phone_embs

        n_extracted = sum(1 for k in all_phone_embs if k[0] == dataset_name)
        print(f"  Extracted for {n_extracted}/{len(spk_records)} speakers")

    # === Phase 1b: Save phone embeddings if requested ===
    if args.save_embeddings:
        emb_dir = RESULTS_DIR / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for (ds, spk_id), phone_embs in all_phone_embs.items():
            safe_spk = spk_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
            out_path = emb_dir / f"{ds}__{safe_spk}.npz"
            phones = [pe["phone"] for pe in phone_embs]
            embeddings = np.array([pe["embedding"] for pe in phone_embs])
            durations = np.array([pe["duration"] for pe in phone_embs])
            np.savez_compressed(out_path,
                                phones=phones,
                                embeddings=embeddings,
                                durations=durations,
                                dataset=ds,
                                speaker_id=spk_id)
            n_saved += 1
        print(f"\nSaved embeddings for {n_saved} speakers to {emb_dir}")

    # === Phase 2: Compute feature directions from controls per language ===
    print("\n--- Computing feature directions from healthy controls ---")

    feature_directions_by_lang = {}

    for lang in set(DATASET_LANGUAGE.values()):
        print(f"\n  Language: {lang}")
        hc_datasets = LANGUAGE_HC_DATASETS.get(lang, [])

        # Collect all HC phone embeddings for this language
        hc_phone_embs = []
        for ds in hc_datasets:
            ds_records = inv_by_dataset.get(ds, [])
            for rec in ds_records:
                is_ctrl = rec.get("is_control", "False")
                if isinstance(is_ctrl, str):
                    is_ctrl = is_ctrl.lower() == "true"
                if not is_ctrl:
                    continue
                key = (ds, rec["speaker_id"])
                if key in all_phone_embs:
                    hc_phone_embs.extend(all_phone_embs[key])

        if not hc_phone_embs:
            print(f"    WARNING: No HC phone embeddings for {lang}")
            feature_directions_by_lang[lang] = {}
            continue

        print(f"    {len(hc_phone_embs)} HC phone tokens from {hc_datasets}")
        directions = compute_hc_feature_directions(hc_phone_embs, lang, mapper)
        feature_directions_by_lang[lang] = directions
        print(f"    Computed {len(directions)} feature directions")

    # === Phase 3: Compute per-speaker metrics ===
    print("\n--- Computing per-speaker metrics ---")

    results = []
    for dataset_name in datasets:
        lang = DATASET_LANGUAGE.get(dataset_name, "en")
        directions = feature_directions_by_lang.get(lang, {})

        if not directions:
            print(f"  Skipping {dataset_name}: no feature directions for {lang}")
            continue

        spk_records = inv_by_dataset.get(dataset_name, [])
        for rec in spk_records:
            spk_id = rec["speaker_id"]
            key = (dataset_name, spk_id)
            if key not in all_phone_embs:
                continue

            phone_embs = all_phone_embs[key]
            metrics = compute_speaker_metrics(phone_embs, directions, lang, mapper)

            result_row = {
                "dataset": dataset_name,
                "speaker_id": spk_id,
                "language": lang,
                "aetiology": rec.get("aetiology", "unknown"),
                "severity_label": rec.get("severity_label", "unknown"),
                "severity_numeric": rec.get("severity_numeric", -1),
                "is_control": rec.get("is_control", False),
                "n_phones": len(phone_embs),
            }
            result_row.update(metrics)
            results.append(result_row)

    # === Write output CSV ===
    if results:
        # Collect all metric columns
        metric_cols = set()
        for r in results:
            for k in r:
                if k not in {"dataset", "speaker_id", "language", "aetiology",
                             "severity_label", "severity_numeric", "is_control", "n_phones"}:
                    metric_cols.add(k)
        metric_cols = sorted(metric_cols)

        fieldnames = ["dataset", "speaker_id", "language", "aetiology",
                      "severity_label", "severity_numeric", "is_control",
                      "n_phones"] + metric_cols

        out_csv = RESULTS_DIR / f"track4_results{args.output_suffix}.csv"

        if args.dataset and out_csv.exists():
            # Single dataset: load existing, remove old rows for this dataset, append new
            existing = []
            with open(out_csv, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["dataset"] == args.dataset:
                        continue
                    # Normalize column names
                    if "nasality_dprime" in r and "nasal_dprime" not in r:
                        r["nasal_dprime"] = r.pop("nasality_dprime")
                    existing.append(r)
            print(f"  Merged with existing ({len(existing)} kept + {len(results)} new)")
        else:
            existing = []

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in existing:
                for col in metric_cols:
                    if col not in r:
                        r[col] = ""
                writer.writerow(r)
            for r in results:
                for col in metric_cols:
                    if col not in r:
                        r[col] = float("nan")
                writer.writerow(r)

        print(f"\nWrote {len(existing) + len(results)} speaker results to {out_csv}")
    else:
        print("\nNo results to write!")

    # === Print summary ===
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 80)

    if results:
        by_dataset = defaultdict(list)
        for r in results:
            by_dataset[r["dataset"]].append(r)

        for ds in sorted(by_dataset.keys()):
            recs = by_dataset[ds]
            print(f"\n  {ds}: {len(recs)} speakers")
            for col in metric_cols:
                vals = [r[col] for r in recs if not (isinstance(r[col], float) and np.isnan(r[col]))]
                if vals:
                    print(f"    {col}: mean={np.mean(vals):.4f}, "
                          f"std={np.std(vals):.4f}, n={len(vals)}")


if __name__ == "__main__":
    main()
