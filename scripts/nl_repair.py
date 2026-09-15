#!/usr/bin/env python
"""Repair step 2 (LaVonne v1.3, L-2): rebuild Dutch feature directions from
HEALTHY-ONLY controls and re-score every nl speaker. DGX, GPU.

The published nl directions pooled COPAS speakers flagged is_control=True,
115 of whom are pathological (cleft palate, laryngectomy, voice disorder,
mixed, unknown). Corrected pool = COPAS speakers with aetiology 'healthy'
(list shipped as copas_healthy.tsv).

Embeddings: loaded from the cache where present (COPAS: 218 of 227), else
extracted from aligned TextGrids + corpus audio (CHASING, TreasureHunters1,
Domotica, and uncached COPAS) with hubert-base, identical recipe.

Output: ~/dysarthria/results/track4/test4_rebuilds/nl_repair_dprime.csv
(9 d-prime columns in master naming) + a token-count line for the new pool.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE = Path.home() / "dysarthria"
EMB = BASE / "results" / "track4" / "embeddings"
ALIGNED = BASE / "results" / "track4" / "aligned"
CORPUS = BASE / "results" / "track4" / "corpus"
CFG = BASE / "track4" / "config" / "phone_features_nl.json"
HC_LIST = Path(__file__).resolve().parent / "copas_healthy.tsv"
OUT = BASE / "results" / "track4" / "test4_rebuilds" / "nl_repair_dprime.csv"

DATASETS = ["COPAS", "CHASING", "TreasureHunters1", "Domotica"]
FPS = 50.0
MIN_TOKENS = 5
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLMAP = {"nasal": "nasal_dprime", "voicing": "voicing_dprime",
          "sonorant": "sonorant_dprime", "strident": "strident_dprime",
          "manner": "manner_dprime", "vowel_high": "high_dprime",
          "vowel_low": "low_dprime", "vowel_back": "back_dprime",
          "vowel_round": "round_dprime"}


def feat_sets():
    d = json.loads(CFG.read_text(encoding="utf-8"))
    fs = {}
    for k, v in d["consonant_features"].items():
        fs[k if k != "nasality" else "nasal"] = (set(v["positive"]), set(v["negative"]))
    for k, v in d["vowel_features"].items():
        fs[f"vowel_{k}"] = (set(v["positive"]), set(v["negative"]))
    return fs


def parse_textgrid(path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'name = "phones".*?(?=item \[|\Z)', txt, re.S)
    if not m:
        return []
    out = []
    for iv in re.finditer(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "([^"]*)"',
                          m.group(0)):
        t = iv.group(3).strip()
        if t and t not in ("sil", "sp", "spn"):
            out.append((float(iv.group(1)), float(iv.group(2)), t))
    return out


def main():
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    mp = BASE / "models" / "hubert-base-ls960"
    mp = str(mp) if mp.exists() else "facebook/hubert-base-ls960"
    model = AutoModel.from_pretrained(mp).to(DEV).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(mp)
    fs = feat_sets()

    hc = set()
    for line in HC_LIST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            hc.add(line.strip())
    print(f"healthy COPAS controls: {len(hc)}")

    def get_embs(ds, spk):
        npz = EMB / f"{ds}__{spk}.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            return list(zip([str(p) for p in z["phones"]],
                            z["embeddings"].astype(np.float64)))
        spkdir = ALIGNED / ds / spk
        if not spkdir.exists():
            return []
        out = []
        for tg in sorted(spkdir.glob("*.TextGrid")):
            wav = CORPUS / ds / spk / tg.with_suffix(".wav").name
            if not wav.exists():
                continue
            phones = parse_textgrid(tg)
            if not phones:
                continue
            audio, sr = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import torchaudio
                audio = torchaudio.transforms.Resample(sr, 16000)(
                    torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
            inputs = extractor(audio, sampling_rate=16000, return_tensors="pt",
                               padding=True)
            with torch.no_grad():
                frames = model(inputs.input_values.to(DEV)).last_hidden_state \
                    .squeeze(0).cpu().numpy()
            for xmin, xmax, ph in phones:
                a, b = int(xmin * FPS), int(xmax * FPS)
                if b <= a:
                    b = a + 1
                if a >= len(frames):
                    continue
                out.append((ph, frames[a:min(b, len(frames))].mean(axis=0)))
        return out

    speakers = {}
    for ds in DATASETS:
        d = ALIGNED / ds
        if not d.exists():
            continue
        for spkdir in sorted(p.name for p in d.iterdir() if p.is_dir()):
            embs = get_embs(ds, spkdir)
            if embs:
                speakers[(ds, spkdir)] = embs
            print(f"{ds}/{spkdir}: {len(embs)} phones", flush=True)

    # healthy-only pooled directions
    sums = {f_: {"pos": [None, 0], "neg": [None, 0]} for f_ in fs}
    n_hc_used = 0
    for (ds, spk), embs in speakers.items():
        if ds != "COPAS" or spk not in hc:
            continue
        n_hc_used += 1
        for ph, e in embs:
            for f_, (pos, neg) in fs.items():
                cls = "pos" if ph in pos else ("neg" if ph in neg else None)
                if cls:
                    s_ = sums[f_][cls]
                    s_[0] = e.copy() if s_[0] is None else s_[0] + e
                    s_[1] += 1
    dirs = {}
    tok = {}
    for f_, s_ in sums.items():
        tok[f_] = (s_["pos"][1], s_["neg"][1])
        if s_["pos"][1] >= MIN_TOKENS and s_["neg"][1] >= MIN_TOKENS:
            d = s_["pos"][0] / s_["pos"][1] - s_["neg"][0] / s_["neg"][1]
            n = np.linalg.norm(d)
            if n > 1e-8:
                dirs[f_] = d / n
    print(f"directions from {n_hc_used} healthy controls; tokens: {tok}")

    rows = []
    for (ds, spk), embs in sorted(speakers.items()):
        rec = {"dataset": ds, "speaker_id": spk, "n_phones": len(embs)}
        for f_, (pos, neg) in fs.items():
            col = COLMAP[f_]
            if f_ not in dirs:
                rec[col] = ""
                continue
            p_ = np.array([e for ph, e in embs if ph in pos])
            n_ = np.array([e for ph, e in embs if ph in neg])
            if len(p_) < MIN_TOKENS or len(n_) < MIN_TOKENS:
                rec[col] = ""
                continue
            pp, pn = p_ @ dirs[f_], n_ @ dirs[f_]
            sp = np.sqrt((pp.var() + pn.var()) / 2)
            rec[col] = round(float(abs(pp.mean() - pn.mean()) / sp), 6) if sp > 1e-8 else 0.0
        rows.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "speaker_id", "n_phones"]
                           + list(COLMAP.values()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
