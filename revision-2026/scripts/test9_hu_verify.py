#!/usr/bin/env python
"""Test 9 / F-2 verification — can the schema-translated Hungarian config reproduce
the submitted hu d-prime values?

Extracts hubert-base phone embeddings for ALL aligned Hungarian speakers
(Hungarian_HC + CV_Hungarian = HC pool; Hungarian_Dysarthria = dys), builds pooled
directions per the published construction using the surviving hu config translated
consonants->consonant_features / vowels->vowel_features, scores every speaker,
writes CSV for comparison against the frozen master. DGX, GPU.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE = Path.home() / "dysarthria"
ALIGNED = BASE / "results" / "track4" / "aligned"
CORPUS = BASE / "results" / "track4" / "corpus"
CFG = BASE / "track4" / "config" / "phone_features_hu.json"
OUT = BASE / "results" / "track4" / "test4_rebuilds" / "test9_hu_dprime.csv"

DATASETS = ["Hungarian_HC", "CV_Hungarian", "Hungarian_Dysarthria"]
HC_DS = {"Hungarian_HC", "CV_Hungarian"}
FPS = 50.0
MIN_TOKENS = 5
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feat_sets():
    d = json.loads(CFG.read_text(encoding="utf-8"))
    cons = d.get("consonant_features") or d["consonants"]
    vows = d.get("vowel_features") or d["vowels"]
    fs = {}
    for k, v in cons.items():
        fs[k if k != "nasality" else "nasal"] = (set(v["positive"]), set(v["negative"]))
    for k, v in vows.items():
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
    print("features:", sorted(fs))

    spk_embs = {}
    for ds in DATASETS:
        for spkdir in sorted((ALIGNED / ds).iterdir()):
            if not spkdir.is_dir():
                continue
            embs = []
            for tg in sorted(spkdir.glob("*.TextGrid")):
                wav = CORPUS / ds / spkdir.name / tg.with_suffix(".wav").name
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
                    embs.append((ph, frames[a:min(b, len(frames))].mean(axis=0)))
            if embs:
                spk_embs[(ds, spkdir.name)] = embs
            print(f"{ds}/{spkdir.name}: {len(embs)} phones", flush=True)

    # pooled directions from HC
    sums = {f: {"pos": [None, 0], "neg": [None, 0]} for f in fs}
    for (ds, spk), embs in spk_embs.items():
        if ds not in HC_DS:
            continue
        for ph, e in embs:
            for f_, (pos, neg) in fs.items():
                cls = "pos" if ph in pos else ("neg" if ph in neg else None)
                if cls:
                    s = sums[f_][cls]
                    s[0] = e.astype(np.float64) if s[0] is None else s[0] + e
                    s[1] += 1
    dirs = {}
    for f_, s in sums.items():
        if s["pos"][1] >= MIN_TOKENS and s["neg"][1] >= MIN_TOKENS:
            d = s["pos"][0] / s["pos"][1] - s["neg"][0] / s["neg"][1]
            n = np.linalg.norm(d)
            if n > 1e-8:
                dirs[f_] = d / n
    print("directions:", sorted(dirs))

    rows = []
    for (ds, spk), embs in sorted(spk_embs.items()):
        rec = {"dataset": ds, "speaker_id": spk, "n_phones": len(embs)}
        for f_, (pos, neg) in fs.items():
            if f_ not in dirs:
                rec[f_] = ""
                continue
            p_ = np.array([e for ph, e in embs if ph in pos])
            n_ = np.array([e for ph, e in embs if ph in neg])
            if len(p_) < MIN_TOKENS or len(n_) < MIN_TOKENS:
                rec[f_] = ""
                continue
            pp, pn = p_ @ dirs[f_], n_ @ dirs[f_]
            sp = np.sqrt((pp.var() + pn.var()) / 2)
            rec[f_] = round(float(abs(pp.mean() - pn.mean()) / sp), 6) if sp > 1e-8 else 0.0
        rows.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "speaker_id", "n_phones"] + sorted(fs)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
