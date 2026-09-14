#!/usr/bin/env python
"""Test 5a — generic-degradation control (ANALYSIS_PLAN §6a). DGX, GPU.

20 LibriSpeech English HC speakers (seed 42), up to 20 aligned utterances each,
processed through the IDENTICAL pipeline (hubert-base last_hidden_state, 50 fps
phone mean-pooling, published pooled English directions rebuilt from cached HC)
under six conditions:
  clean            sanity check vs the cache
  noise_snr10      additive gaussian at 10 dB SNR
  noise_snr0       additive gaussian at 0 dB SNR
  lowpass3k        3 kHz biquad low-pass
  speed0.85        resample-based speed change (pitch shifts too — sox unavailable
  speed1.15        on this box; deviation from 'tempo' documented in the report)
Phone interval times are rescaled by 1/speed for the speed conditions.

Output: ~/dysarthria/results/track4/test4_rebuilds/test5_degradation_dprime.csv
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE = Path.home() / "dysarthria"
EMB_DIR = BASE / "results" / "track4" / "embeddings"
ALIGNED = BASE / "results" / "track4" / "aligned" / "LibriSpeech_English"
CORPUS = BASE / "results" / "track4" / "corpus" / "LibriSpeech_English"
CONFIG = BASE / "track4" / "config" / "phone_features_en.json"
OUT = BASE / "results" / "track4" / "test4_rebuilds" / "test5_degradation_dprime.csv"
HC_LIST = Path.home() / "dysarthria" / "scripts" / "oneoff" / "hc_speakers.tsv"

SEED = 42
N_SPK = 20
N_UTT = 20
FPS = 50.0
MIN_TOKENS = 5
FEATS = ["nasal", "voicing", "sonorant", "strident", "manner"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_textgrid(path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'name = "phones".*?(?=item \[|\Z)', txt, re.S)
    if not m:
        return []
    seg = m.group(0)
    out = []
    for iv in re.finditer(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "([^"]*)"', seg):
        t = iv.group(3).strip()
        if t and t not in ("sil", "sp", "spn", ""):
            out.append((float(iv.group(1)), float(iv.group(2)), t))
    return out


def load_feat_sets():
    cf = json.loads(CONFIG.read_text(encoding="utf-8"))["consonant_features"]
    return {f: (set(cf[f]["positive"]), set(cf[f]["negative"])) for f in FEATS}


def build_en_directions(feat_sets):
    """Published pooled construction from all cached English HC speakers."""
    hc = set()
    for line in HC_LIST.read_text(encoding="utf-8").splitlines():
        ds, spk = line.split("\t")
        if ds in ("LibriSpeech_English", "TORGO", "UASPEECH_control"):
            hc.add((ds, spk))
    sums = {f: {"pos": [None, 0], "neg": [None, 0]} for f in FEATS}
    n = 0
    for ds, spk in sorted(hc):
        f = EMB_DIR / f"{ds}__{spk}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        phones = [str(p) for p in z["phones"]]
        embs = z["embeddings"].astype(np.float64)
        n += 1
        for i, ph in enumerate(phones):
            for ft, (pos, neg) in feat_sets.items():
                cls = "pos" if ph in pos else ("neg" if ph in neg else None)
                if cls:
                    s = sums[ft][cls]
                    s[0] = embs[i].copy() if s[0] is None else s[0] + embs[i]
                    s[1] += 1
    dirs = {}
    for ft, s in sums.items():
        if s["pos"][1] >= MIN_TOKENS and s["neg"][1] >= MIN_TOKENS:
            d = s["pos"][0] / s["pos"][1] - s["neg"][0] / s["neg"][1]
            dirs[ft] = d / np.linalg.norm(d)
    print(f"directions from {n} HC speakers: {sorted(dirs)}", flush=True)
    return dirs


def degrade(audio, sr, cond, rng):
    import torchaudio.functional as F
    x = torch.from_numpy(audio).float()
    if cond == "clean":
        return audio, 1.0
    if cond.startswith("noise_snr"):
        snr = float(cond.replace("noise_snr", ""))
        rms = float(np.sqrt(np.mean(audio ** 2))) or 1e-8
        nrms = rms / (10 ** (snr / 20))
        return audio + rng.standard_normal(len(audio)).astype(np.float32) * nrms, 1.0
    if cond == "lowpass3k":
        return F.lowpass_biquad(x, sr, 3000.0).numpy(), 1.0
    if cond.startswith("speed"):
        s = float(cond.replace("speed", ""))
        import torchaudio
        y = torchaudio.transforms.Resample(int(sr * s), sr)(x.unsqueeze(0)).squeeze(0)
        return y.numpy(), s
    raise ValueError(cond)


def main():
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    mp = BASE / "models" / "hubert-base-ls960"
    mp = str(mp) if mp.exists() else "facebook/hubert-base-ls960"
    model = AutoModel.from_pretrained(mp).to(DEV).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(mp)

    feat_sets = load_feat_sets()
    dirs = build_en_directions(feat_sets)

    rng = np.random.default_rng(SEED)
    spks = sorted(d.name for d in ALIGNED.iterdir() if d.is_dir())
    pick = list(rng.choice(spks, size=min(N_SPK, len(spks)), replace=False))
    conds = ["clean", "noise_snr10", "noise_snr0", "lowpass3k", "speed0.85", "speed1.15"]
    rows = []
    for spk in pick:
        tgs = sorted((ALIGNED / spk).glob("*.TextGrid"))[:N_UTT]
        utt = []
        for tg in tgs:
            wav = CORPUS / spk / tg.with_suffix(".wav").name
            if wav.exists():
                utt.append((tg, wav))
        print(f"{spk}: {len(utt)} utts", flush=True)
        for cond in conds:
            cls_emb = {ft: {"pos": [], "neg": []} for ft in FEATS}
            nph = 0
            for tg, wav in utt:
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
                    sr = 16000
                aud, s = degrade(audio, sr, cond, rng)
                inputs = extractor(aud, sampling_rate=16000, return_tensors="pt",
                                   padding=True)
                with torch.no_grad():
                    frames = model(inputs.input_values.to(DEV)).last_hidden_state \
                        .squeeze(0).cpu().numpy()
                for xmin, xmax, ph in phones:
                    a, b = int(xmin / s * FPS), int(xmax / s * FPS)
                    if b <= a:
                        b = a + 1
                    if a >= len(frames):
                        continue
                    e = frames[a:min(b, len(frames))].mean(axis=0)
                    nph += 1
                    for ft, (pos, neg) in feat_sets.items():
                        if ph in pos:
                            cls_emb[ft]["pos"].append(e)
                        elif ph in neg:
                            cls_emb[ft]["neg"].append(e)
            rec = {"speaker_id": spk, "condition": cond, "n_phones": nph}
            for ft in FEATS:
                p_, n_ = cls_emb[ft]["pos"], cls_emb[ft]["neg"]
                if ft not in dirs or len(p_) < MIN_TOKENS or len(n_) < MIN_TOKENS:
                    rec[ft] = ""
                    continue
                pp = np.array(p_) @ dirs[ft]
                pn = np.array(n_) @ dirs[ft]
                sp = np.sqrt((pp.var() + pn.var()) / 2)
                rec[ft] = round(float(abs(pp.mean() - pn.mean()) / sp), 6) if sp > 1e-8 else 0.0
            rows.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker_id", "condition", "n_phones"] + FEATS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
