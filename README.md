# Phonological Subspace Severity Assessment

**Training-free, cross-lingual dysarthria severity assessment via phonological subspace analysis in self-supervised speech representations.**

> B. Muller, A.A. Ortiz Barrañón, and L. Roberts, "Training-Free Cross-Lingual Dysarthria Severity Assessment via Phonological Subspace Analysis in Self-Supervised Speech Representations," submitted to *PLOS Digital Health*, 2026. Preprint: [arXiv:TBD]

## Overview

Dysarthria reduces a speaker's ability to maintain phonological contrasts (e.g., nasal vs. oral, voiced vs. voiceless). This repository provides a **training-free** pipeline that quantifies this degradation by measuring the separability (d-prime) of phonological feature directions in frozen [HuBERT](https://arxiv.org/abs/2106.07447) embeddings.

The method requires **no dysarthric training data and no model adaptation** -- feature directions are computed entirely from healthy control speech. It generalises across languages and aetiologies without modification.

### Key results

- **890 speakers**, 10 corpora, 5 languages (English, Spanish, Dutch, Mandarin, French)
- All 5 consonant d-primes correlate with severity: rho = -0.47 to -0.55 (p < 10^-48)
- Effect replicates within every individual corpus (rho up to -0.92)
- Cross-lingual: identical pipeline, no language-specific tuning
- 8 robustness experiments (bootstrap CIs, LOCO, meta-analysis, alternative SSL backbones, etc.)

## Method

```
Audio + Transcript
    --> Montreal Forced Aligner (phone-level alignment)
    --> HuBERT-base frame embeddings (768-dim, 50 fps)
    --> Mean-pool per phone interval
    --> Compute feature directions from healthy controls
         (e.g., nasal centroid - oral centroid, normalised)
    --> Project each speaker's phone embeddings onto directions
    --> d-prime per speaker per feature --> 12-metric severity profile
```

### 12 metrics per speaker

| Category | Features |
|----------|----------|
| Consonant d-prime (5) | nasality, voicing, stridency, sonorance, manner |
| Vowel d-prime (4) | height, backness, lowness, rounding |
| Structural (3) | boundary sharpness, cross-position cosine similarity, vowel triangle area |

## Installation

```bash
pip install -r requirements.txt
```

Requires Python >= 3.9, PyTorch >= 2.0, and a CUDA GPU for efficient HuBERT inference.

For phone-level alignment you also need [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (v2.7+).

## Usage

### 1. Prepare MFA corpus

Organise audio and transcripts into per-speaker directories:

```bash
python scripts/mfa_align.py --dataset TORGO --prepare-only
```

This creates `corpus/<DATASET>/<speaker_id>/` with paired `.wav` and `.lab` files.

### 2. Run MFA alignment

```bash
mfa align corpus/<DATASET> english_mfa english_mfa aligned/<DATASET> --num_jobs 8
```

Replace `english_mfa` with the appropriate language model (see table below).

| Language | Acoustic model | Dictionary |
|----------|---------------|------------|
| English | `english_mfa` | `english_mfa` |
| Dutch | `dutch_cv` | `dutch_cv` |
| Spanish | `spanish_mfa` | `spanish_mfa` |
| French | `french_mfa` | `french_mfa` |
| Mandarin | `mandarin_mfa` | `mandarin_mfa` |
| Italian | `italian_cv` | `italian_cv` |

### 3. Extract features

```bash
python scripts/extract_features.py --dataset TORGO
# or for all aligned datasets:
python scripts/extract_features.py --all --save-embeddings
```

Output: one CSV row per speaker with 12 metrics + metadata.

### 4. Statistical analysis

```bash
python scripts/analyze.py --results-csv results/track4_results_merged.csv
```

Produces Spearman correlations, Mann-Whitney U tests, severity group comparisons, and publication-ready figures.

## Phone feature configurations

Each language has a JSON config mapping IPA phones to binary phonological features. For example, English nasality:

```json
{
  "nasality": {
    "positive": ["m", "n", "ng"],
    "negative": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", ...]
  }
}
```

To add a new language, create `config/phone_features_<lang>.json` using the IPA symbols from the corresponding MFA dictionary.

## Aggregate Results

Aggregate statistics are provided in `results/` for verification of the paper's tables and figures without requiring access to restricted datasets:

| File | Description |
|------|-------------|
| `results/aggregate_pooled_correlations.csv` | Spearman rho for all 12 features vs ordinal severity (Table 5 / Fig 5) |
| `results/aggregate_group_means.csv` | Mean d' by severity group for all 12 features (Table 6) |
| `results/aggregate_corpus_correlations.csv` | Per-corpus Spearman rho for consonant features (Table 4) |
| `results/aggregate_speaker_counts.csv` | Speaker counts by dataset and severity (Table 1) |

Per-speaker results are available from the corresponding author upon reasonable request, subject to the data sharing agreements of the underlying corpora.

## Datasets

The paper validates on these corpora (not included in this repository due to licensing):

| Dataset | Language | Aetiology | Speakers | Reference |
|---------|----------|-----------|----------|-----------|
| TORGO | English | Cerebral palsy | 15 | Rudzicz et al., 2012 |
| UA-Speech | English | Cerebral palsy | 28 | Kim et al., 2008 |
| SAP / SAPC2 | English | Mixed (5 aetiologies) | 188 | Laaridh et al., 2024 |
| LibriSpeech | English | Healthy controls | 150 | Panayotov et al., 2015 |
| COPAS | Dutch | Mixed | 218 | Middag et al., 2009 |
| Neurovoz | Spanish | Parkinson's | 111 | Moro-Velazquez et al., 2019 |
| PC-GITA | Spanish | Parkinson's | 100 | Orozco-Arroyave et al., 2014 |
| MDSC | Mandarin | Cerebral palsy | 56 | Wang et al., 2022 |
| YouTube French | French | ALS | 24 | Muller, Ortiz Barrañón & Roberts, 2026 |
| VOC-ALS | Italian | ALS | 153 | Turrisi et al., 2024 |

## Citation

If you use this code, please cite:

```bibtex
@article{muller2026phonological,
  title={Training-Free Cross-Lingual Dysarthria Severity Assessment via Phonological Subspace Analysis in Self-Supervised Speech Representations},
  author={Muller, Bernard and Ortiz Barra{\~n}{\'o}n, Antonio Armando and Roberts, LaVonne},
  journal={PLOS Digital Health},
  year={2026},
  note={The Scott-Morgan Foundation}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work is part of the [CANDOR](https://scottmorganfoundation.org) research programme at The Scott-Morgan Foundation, building cross-lingual adaptive speech technology for people with motor neurone disease and other neurological conditions.

We thank the creators and custodians of all datasets used in this study.
