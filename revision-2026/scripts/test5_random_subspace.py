#!/usr/bin/env python
"""Test 5b — matched random-subspace control (ANALYSIS_PLAN §6b). DGX, cache-only.

For each language: R random unit directions per feature (768-dim), same phone classes
and token counts as the real pipeline — only the DIRECTION is random. Per-speaker d'
per draw. Establishes the floor for (i) HC-vs-dys separation and (ii) cell-profile
cosine statistics under directions with no phonological content.

Output: ~/dysarthria/results/track4/test4_rebuilds/test5_random_dprime.csv
Seed 42, R=50.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EMB_DIR = Path.home() / "dysarthria" / "results" / "track4" / "embeddings"
OUT = Path.home() / "dysarthria" / "results" / "track4" / "test4_rebuilds" / "test5_random_dprime.csv"
CONFIG_DIR = Path.home() / "dysarthria" / "track4" / "config"

SEED = 42
R = 50
FEATS = ["nasal", "voicing", "sonorant", "strident", "manner"]
MIN_TOKENS = 5
LANGS = {"en": ["LibriSpeech_English", "TORGO", "UASPEECH_control", "SAP", "UASPEECH"],
         "es": ["Neurovoz", "PC-GITA"], "it": ["IPVS"], "nl": ["COPAS"],
         "sk": ["EWA-DB"], "zh": ["MDSC"], "fr": ["YouTube_French"]}


def load_config(lang):
    d = json.loads((CONFIG_DIR / f"phone_features_{lang}.json").read_text(encoding="utf-8"))
    cf = d["consonant_features"]
    if "nasality" in cf and "nasal" not in cf:   # sk names the feature 'nasality'
        cf["nasal"] = cf["nasality"]
    return {f: (set(cf[f]["positive"]), set(cf[f]["negative"])) for f in FEATS if f in cf}


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for lang, datasets in LANGS.items():
        feat_sets = load_config(lang)
        D = rng.standard_normal((768, 5 * R))
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        files = []
        for ds in datasets:
            files += [(ds, f) for f in sorted(EMB_DIR.glob(f"{ds}__*.npz"))]
        print(f"{lang}: {len(files)} speakers", flush=True)
        for ds, f in files:
            spk = f.stem.split("__", 1)[1]
            z = np.load(f, allow_pickle=True)
            phones = [str(p) for p in z["phones"]]
            embs = z["embeddings"].astype(np.float64)
            proj = embs @ D                     # tokens x 5R
            idx = {ft: {"pos": [], "neg": []} for ft in FEATS}
            for i, ph in enumerate(phones):
                for ft, (pos, neg) in feat_sets.items():
                    if ph in pos:
                        idx[ft]["pos"].append(i)
                    elif ph in neg:
                        idx[ft]["neg"].append(i)
            for r in range(R):
                rec = {"language": lang, "dataset": ds, "speaker_id": spk, "draw": r}
                for fi, ft in enumerate(FEATS):
                    p_, n_ = idx[ft]["pos"], idx[ft]["neg"]
                    if len(p_) < MIN_TOKENS or len(n_) < MIN_TOKENS:
                        rec[ft] = ""
                        continue
                    pp, pn = proj[p_, r * 5 + fi], proj[n_, r * 5 + fi]
                    sp = np.sqrt((pp.var() + pn.var()) / 2)
                    rec[ft] = round(float(abs(pp.mean() - pn.mean()) / sp), 6) if sp > 1e-8 else 0.0
                rows.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["language", "dataset", "speaker_id", "draw"] + FEATS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
