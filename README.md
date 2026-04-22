# Phonological Subspace Severity Assessment

**Training-free, cross-lingual dysarthria severity assessment via phonological subspace analysis in self-supervised speech representations.**

This repository supports two companion papers:

- **Paper 1 — method introduction:**
  B. Muller, A.A. Ortiz Barrañón, and L. Roberts (2026). *Training-Free Cross-Lingual Dysarthria Severity Assessment via Phonological Subspace Analysis in Self-Supervised Speech Representations.* arXiv preprint arXiv:2604.10123. https://arxiv.org/abs/2604.10123
- **Paper 2 — scale-up and phenotyping extension:**
  B. Muller, A.A. Ortiz Barrañón, and L. Roberts (2026). *Phonological Subspace Collapse Is Aetiology-Specific and Cross-Lingually Stable: Evidence from 3,374 Speakers.* Under review, *Computer Speech & Language*. arXiv preprint: TBD.

## Overview

Dysarthria reduces a speaker's ability to maintain phonological contrasts (e.g., nasal vs. oral, voiced vs. voiceless). This repository provides a **training-free** pipeline that quantifies this degradation by measuring the separability (d-prime) of phonological feature directions in frozen [HuBERT](https://arxiv.org/abs/2106.07447) embeddings (and five other SSL backbones in Paper 2).

The method requires **no dysarthric training data and no model adaptation** — feature directions are computed entirely from healthy control speech. It generalises across languages, aetiologies, and SSL architectures without modification.

### Key results

**Paper 1 (890 speakers, 10 corpora, 5 languages):**
- All 5 consonant d-primes correlate with severity: rho = −0.47 to −0.55 (p < 10⁻⁴⁸)
- Effect replicates within every individual corpus (rho up to −0.92)
- 8 robustness experiments (bootstrap CIs, LOCO, meta-analysis, alternative SSL backbones)

**Paper 2 (3,374 speakers, 25 corpora, 12 languages, 6 SSL backbones):**
- Aetiology-specific degradation profiles distinguishable at the group level (10 of 13 features: Large effect sizes, epsilon-squared > 0.14, FDR-corrected p < 0.001)
- Cross-lingual profile-shape stability (cosine similarity > 0.95 across available languages per aetiology)
- Architecture-independent (inter-model rho > 0.77 across HuBERT-base, HuBERT-large, WavLM, wav2vec2, XLS-R, MMS)
- Fixed-token d-prime estimation preserves severity signal (rho = −0.733 at 200 tokens/class)
- SAP-excluded sensitivity confirms PD/CP/ALS claims are not SAP-specific

## Method

```
Audio + Transcript
    → Montreal Forced Aligner (phone-level alignment)
    → Frozen SSL frame embeddings (50 fps)
    → Mean-pool per phone interval
    → Compute feature directions from healthy controls
         (e.g., nasal centroid − oral centroid, normalised)
    → Project each speaker's phone embeddings onto directions
    → d-prime per speaker per feature → 15-metric phonological profile
```

### 15 metrics per speaker (Paper 2)

| Category | Features |
|----------|----------|
| Consonant d-prime (5) | nasality, voicing, stridency, sonorance, manner |
| Vowel d-prime (4) | height, backness, lowness, rounding |
| Structural (3) | boundary sharpness, cross-position cosine similarity, vowel triangle area |
| Prosodic (3) | speech rate, pause rate, vowel duration CV |

Paper 1 used the 12 segmental + structural metrics. Paper 2 adds the 3 prosodic metrics and uses a 13-feature main-analysis subset (excluding boundary sharpness and cross-position cosine, which are content-type dependent).

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, and a CUDA GPU for efficient HuBERT inference.

For phone-level alignment you also need [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (v3.1+).

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

Replace `english_mfa` with the appropriate language model. Paper 2 covers 12 languages:

| Language | Acoustic model |
|----------|---------------|
| English | `english_mfa` |
| Dutch | `dutch_cv` |
| Spanish | `spanish_mfa` |
| French | `french_mfa` |
| Mandarin | `mandarin_mfa` |
| Italian | `italian_cv` |
| Slovak | `czech_mfa` (fallback) |
| Hungarian | `hungarian_cv` |
| Portuguese | `portuguese_mfa` |
| German | `german_mfa` |
| Swahili | `swahili_mfa` |
| Tamil | `tamil_cv` (+ torchaudio MMS fallback) |

### 3. Extract features

```bash
python scripts/extract_features.py --dataset TORGO
# or for all aligned datasets:
python scripts/extract_features.py --all --save-embeddings
# or for non-HuBERT backbones (Paper 2):
python scripts/extract_features.py --all --model wavlm  # {hubert-large, wavlm, wav2vec2, xlsr, mms}
```

Output: one CSV row per speaker with 15 metrics + metadata.

### 4. Statistical analysis (Paper 1)

```bash
python scripts/analyze.py --results-csv results/track4_results_merged.csv
```

## Reproducing Paper 2 results

Each script maps to a specific table, figure, or section. Assumes `track4_master.csv` and the five backbone CSVs are present in `results/track4/` (released on Zenodo upon Paper 2 acceptance).

| Script | Produces |
|--------|----------|
| `scripts/aetiology_discrimination.py` | Table 3 (Kruskal–Wallis, epsilon-squared, Cohen's d across aetiologies) |
| `scripts/cross_lingual_aetiology.py` | Table 4 and Section 4.2 (cross-lingual cosine similarity) |
| `scripts/cross_model_comparison.py` | Table 5 and Section 4.3 (multi-backbone inter-model agreement) |
| `scripts/adjust_token_count.py` | Section 3.4 (token-count regression adjustment) |
| `scripts/robustness_analyses.py` | Section 4.5 (bootstrap CIs, leave-one-dataset-out) |
| `scripts/reviewer_experiments.py` | Section 4.5 (severity-source ablation, minimum-n, Dunn post hoc, token-matched) |
| `scripts/fixed_token_dprime.py` | Section 4.5 and 5.1 (fixed-token d-prime + cross-lingual permutation baseline) |
| `scripts/sap_excluded_sensitivity.py` | Section 4.5 (SAP-excluded sensitivity) |
| `scripts/robustness_pass5.py` | Section 4.5 (common-speaker fixed-token + HC minimum-n) |

## Phone feature configurations

Each language has a JSON config mapping IPA phones to binary phonological features. For example, English nasality:

```json
{
  "consonant_features": {
    "nasal": {
      "positive": ["m", "n", "ŋ"],
      "negative": ["p", "b", "t", "d", "k", "g"]
    }
  }
}
```

To add a new language, create `config/phone_features_<lang>.json` using the IPA symbols from the corresponding MFA dictionary.

## Data availability

**Paper 1 aggregate statistics** are included in `results/` for reproducibility without access to restricted datasets:

| File | Description |
|------|-------------|
| `results/aggregate_pooled_correlations.csv` | Spearman rho for all 12 features vs ordinal severity |
| `results/aggregate_group_means.csv` | Mean d' by severity group for all 12 features |
| `results/aggregate_corpus_correlations.csv` | Per-corpus Spearman rho for consonant features |
| `results/aggregate_speaker_counts.csv` | Speaker counts by dataset and severity |

**Paper 2 speaker-level master CSV** (3,374 speakers × 15 features + metadata) will be released via a Zenodo archive upon acceptance of Paper 2; the DOI will be added here. Per-speaker results for unreleased datasets remain subject to the data-sharing agreements of the underlying corpora and are available from the corresponding author upon reasonable request.

## Figures

Paper 2 figures (PNG and TIFF at 300 DPI) are in `figures/`:

| File | Figure |
|------|--------|
| `figures/fig1_aetiology_heatmap.*` | Figure 1: Deviation from healthy controls by aetiology (Cohen's d) |
| `figures/fig2_aetiology_radars.*` | Figure 2: Pairwise aetiology comparison radar plots |
| `figures/fig3_crosslingual_PD.*` | Figure 3: HC-normalised Parkinson's profiles across 6 languages |
| `figures/fig4_severity_gradient.*` | Figure 4: Severity gradient across 6 SSL backbones with bootstrap CIs |
| `figures/fig5_intermodel_agreement.*` | Figure 5: Inter-model Spearman rho on composite consonant d-prime |

## Citation

If you use this code, please cite both papers:

```bibtex
@article{muller2026phonological,
  title={Training-Free Cross-Lingual Dysarthria Severity Assessment via Phonological Subspace Analysis in Self-Supervised Speech Representations},
  author={Muller, Bernard and Ortiz Barra{\~n}{\'o}n, Antonio Armando and Roberts, LaVonne},
  journal={arXiv preprint arXiv:2604.10123},
  year={2026},
  doi={10.48550/arXiv.2604.10123},
  note={The Scott-Morgan Foundation}
}

@article{muller2026aetiology,
  title={Phonological Subspace Collapse Is Aetiology-Specific and Cross-Lingually Stable: Evidence from 3,374 Speakers},
  author={Muller, Bernard and Ortiz Barra{\~n}{\'o}n, Antonio Armando and Roberts, LaVonne},
  journal={Computer Speech and Language},
  year={2026},
  note={Under review. The Scott-Morgan Foundation}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work is part of the [CANDOR](https://scottmorganfoundation.org) research programme at The Scott-Morgan Foundation, building cross-lingual adaptive speech technology for people with motor neurone disease and other neurological conditions.

We thank the creators and custodians of all datasets used in these studies.
