#!/usr/bin/env python3
"""Confidence-weighted d-prime via duration filtering."""
import numpy as np
import json, os, csv
from pathlib import Path
from collections import defaultdict
from scipy import stats

EMB_DIR = Path(os.environ.get("EMB_DIR", os.path.expanduser("~/dysarthria/results/track4/embeddings")))
INV_CSV = Path(os.environ.get("INV_CSV", os.path.expanduser("~/dysarthria/results/track4/speaker_inventory.csv")))
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", os.path.expanduser("~/dysarthria/config")))
DIRECTIONS_FILE = os.environ.get("DIRECTIONS", os.path.expanduser("~/dysarthria/en_feature_directions.npz"))

DATASET_LANG = {
    "SAP": "en", "COPAS": "nl", "TORGO": "en", "Neurovoz": "es",
    "YouTube_French": "fr", "MDSC": "zh", "IPVS": "it",
    "UASPEECH": "en", "UASPEECH_control": "en",
    "LibriSpeech_English": "en", "PC-GITA": "es",
}
FEATURES = ["nasal", "voicing", "strident", "sonorant", "manner"]
THRESHOLDS = [0.0, 0.02, 0.03, 0.05]

configs = {}
for lang in set(DATASET_LANG.values()):
    p = CONFIG_DIR / f"phone_features_{lang}.json"
    if p.exists():
        with open(p) as f:
            configs[lang] = json.load(f)

def classify(phone, config):
    cats = {}
    for fn, fd in config.get("consonant_features", {}).items():
        if phone in set(fd["positive"]): cats[fn] = "positive"
        elif phone in set(fd["negative"]): cats[fn] = "negative"
    return cats

inv = {}
with open(INV_CSV) as f:
    for row in csv.DictReader(f):
        inv[(row["dataset"], row["speaker_id"])] = row

dirs_data = np.load(DIRECTIONS_FILE)
en_dirs = {k: dirs_data[k] for k in dirs_data.files}

print("=" * 70)
print("CONFIDENCE-WEIGHTED D-PRIME (Duration Filtering)")
print("=" * 70)

npz_files = sorted(EMB_DIR.glob("*.npz"))
print(f"Loading {len(npz_files)} speakers...")

all_results = {}

for min_dur in THRESHOLDS:
    results = []
    n_removed, n_kept = 0, 0

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        ds, spk = str(data["dataset"]), str(data["speaker_id"])
        phones, embs, durs = list(data["phones"]), data["embeddings"], data["durations"]

        lang = DATASET_LANG.get(ds, "en")
        if lang not in configs: continue
        rec = inv.get((ds, spk), {})
        sev_num = int(rec.get("severity_numeric", -1))
        if sev_num < 0: continue

        mask = durs >= min_dur
        n_removed += int((~mask).sum())
        n_kept += int(mask.sum())
        if mask.sum() < 10: continue

        fp = [p for p, m in zip(phones, mask) if m]
        fe = embs[mask]

        by_feat = defaultdict(lambda: defaultdict(list))
        for phone, emb in zip(fp, fe):
            for feat, cls in classify(phone, configs[lang]).items():
                by_feat[feat][cls].append(emb)

        row = {"sev": sev_num}
        for feat in FEATURES:
            d = en_dirs.get(feat)
            if d is None: row[feat] = float("nan"); continue
            pos = np.array(by_feat[feat].get("positive", []))
            neg = np.array(by_feat[feat].get("negative", []))
            if len(pos) < 5 or len(neg) < 5: row[feat] = float("nan"); continue
            pp, pn = pos @ d, neg @ d
            ps = np.sqrt((pp.var() + pn.var()) / 2)
            row[feat] = abs((pp.mean() - pn.mean()) / ps) if ps > 1e-8 else 0.0
        results.append(row)

    pct = n_kept / (n_kept + n_removed) * 100 if (n_kept + n_removed) > 0 else 0
    print(f"\nThreshold: {min_dur*1000:.0f}ms | Speakers: {len(results)} | Kept: {pct:.1f}%")
    print("  " + "-" * 40)
    for feat in FEATURES:
        vals = [(r["sev"], r[feat]) for r in results if not np.isnan(r.get(feat, float("nan")))]
        if len(vals) >= 10:
            sevs, dps = zip(*vals)
            rho, p = stats.spearmanr(sevs, dps)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:<15} rho={rho:>8.4f}  p={p:.2e} {star}")
            all_results.setdefault(f"{min_dur*1000:.0f}ms", {})[feat] = {"rho": float(rho), "p": float(p)}

print("\nSUMMARY: Effect of duration filtering on correlations")
print("  " + "-" * 60)
header = f"  {'Feature':<15}"
for t in THRESHOLDS:
    header += f" {t*1000:.0f}ms".rjust(10)
print(header)
print("  " + "-" * 60)
for feat in FEATURES:
    row = f"  {feat:<15}"
    for t in THRESHOLDS:
        key = f"{t*1000:.0f}ms"
        r = all_results.get(key, {}).get(feat, {})
        rho = r.get("rho", float("nan"))
        row += f" {rho:>9.4f}"
    print(row)

print("\nDONE")
