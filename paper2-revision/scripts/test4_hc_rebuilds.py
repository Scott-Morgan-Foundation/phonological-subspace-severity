#!/usr/bin/env python
"""Test 4 Steps 2-4 — HC feature-direction rebuilds (ANALYSIS_PLAN §5, v1.x).

Runs on the DGX against the phone-embedding cache
(~/dysarthria/results/track4/embeddings/*.npz, one npz per dataset__speaker with
phones / embeddings / durations / dataset / speaker_id).

Modes:
  --audit    coverage table only (cache vs required speakers), fast
  --run      full rebuild + re-score

Direction variants per language (consonant features only, 5 feats):
  pooled        published construction: token-pooled class means (faithfulness check)
  eq_speaker    mean over per-speaker class means (speaker needs >=1 token in class)
  eq_dataset    mean over per-dataset token-pooled class means
  eq_phone      mean over per-phone-type token-pooled means (phone type needs >=5 tokens)
  combined      phone-type means within speaker -> speaker mean -> dataset mean -> mean
English extras (Step 3+4):
  loco_<ds>     directions from HC excluding dataset <ds> (LibriSpeech_English, TORGO,
                UASPEECH_control)
  only_libri    directions from LibriSpeech_English HC only
  only_clinical directions from TORGO + UASPEECH_control HC only

Every variant direction: L2-normalised difference of class means, requires >=5 pos and
>=5 neg tokens overall (as published). d' per speaker per feature: |mean gap| / pooled
std of projections, >=5 tokens per class else NaN (identical to frozen compute_dprime).

Output: test4_speaker_dprime.csv (speaker x feature x variant) + test4_summary.json.
Seed irrelevant (deterministic). Reads the cache and the language->HC-dataset map only.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EMB_DIR = Path.home() / "dysarthria" / "results" / "track4" / "embeddings"
OUT_DIR = Path.home() / "dysarthria" / "results" / "track4" / "test4_rebuilds"
CONFIG_DIR = Path.home() / "dysarthria" / "track4" / "config"

FEATS = ["nasal", "voicing", "sonorant", "strident", "manner"]
MIN_TOKENS = 5

LANGUAGE_HC_DATASETS = {
    "en": ["LibriSpeech_English", "TORGO", "UASPEECH_control"],
    "nl": ["COPAS"], "fr": ["YouTube_French"], "es": ["Neurovoz", "PC-GITA"],
    "zh": ["MDSC"], "it": ["IPVS", "VOC-ALS"], "sk": ["EWA-DB"],
    "hu": ["Hungarian_HC", "CV_Hungarian"], "de": ["YouTube_German", "SVD"],
    "ta": ["SLR65_Tamil", "SSNCE_Tamil"], "pt": ["AVFAD"],
    "sw": ["CDLI_Kenyan_Swahili"],
}
DATASET_LANGUAGE = {
    "SAP": "en", "COPAS": "nl", "TORGO": "en", "Neurovoz": "es",
    "YouTube_French": "fr", "MDSC": "zh", "IPVS": "it", "VOC-ALS": "it",
    "PC-GITA": "es", "UASPEECH": "en", "UASPEECH_control": "en",
    "LibriSpeech_English": "en", "EWA-DB": "sk", "Hungarian_Dysarthria": "hu",
    "Hungarian_HC": "hu", "CDSD": "zh", "YouTube_German": "de", "SVD": "de",
    "EasyCall": "it", "Domotica": "nl", "SSNCE_Tamil": "ta", "SLR65_Tamil": "ta",
    "AVFAD": "pt", "CDLI_Kenyan_Swahili": "sw", "CHASING": "nl",
    "TreasureHunters1": "nl", "SAP_all": "en", "SAP_unlabeled": "en",
    "TORGO_FC01": "en", "CV_Hungarian": "hu",
}

# HC membership comes from hc_speakers.tsv (dataset<TAB>safe_speaker_id), generated
# from the frozen master's aetiology=='healthy' rows (1,445 speakers).
HC_LIST = Path(__file__).resolve().parent / "hc_speakers.tsv"
HC_SET = set()
if HC_LIST.exists():
    for line in HC_LIST.read_text(encoding="utf-8").splitlines():
        ds, spk = line.split("\t")
        HC_SET.add((ds, spk))


def load_config(lang):
    import json as _json
    p = CONFIG_DIR / f"phone_features_{lang}.json"
    if not p.exists():
        return None
    d = _json.loads(p.read_text(encoding="utf-8"))
    cf = d.get("consonant_features")
    if not cf:
        return None
    return {f: (set(cf[f]["positive"]), set(cf[f]["negative"])) for f in FEATS if f in cf}


def scan_cache():
    files = sorted(EMB_DIR.glob("*.npz"))
    by_ds = defaultdict(list)
    for f in files:
        ds, spk = f.stem.split("__", 1)
        by_ds[ds].append((spk, f))
    return by_ds


def is_hc(ds, spk):
    return (ds, spk) in HC_SET


def speaker_class_stats(npz_path, feat_sets):
    """Per feature: dict phone -> (sum_vec, count) split by class, from one speaker."""
    z = np.load(npz_path, allow_pickle=True)
    phones = [str(p) for p in z["phones"]]
    embs = z["embeddings"]
    out = {f: {"pos": defaultdict(lambda: [None, 0]), "neg": defaultdict(lambda: [None, 0])}
           for f in feat_sets}
    for ph, e in zip(phones, embs):
        for f, (pos, neg) in feat_sets.items():
            cls = "pos" if ph in pos else ("neg" if ph in neg else None)
            if cls is None:
                continue
            slot = out[f][cls][ph]
            slot[0] = e.astype(np.float64) if slot[0] is None else slot[0] + e
            slot[1] += 1
    return out, len(phones)


def norm_dir(vpos, vneg):
    d = vpos - vneg
    n = np.linalg.norm(d)
    return d / n if n > 1e-8 else None


def build_directions(hc_stats):
    """hc_stats: list of (dataset, speaker, stats). Returns {variant: {feat: dir}}."""
    variants = defaultdict(dict)
    feats = list(hc_stats[0][2].keys()) if hc_stats else []
    for f in feats:
        # token-pooled sums, plus nested structures
        tot = {"pos": [None, 0], "neg": [None, 0]}
        per_spk, per_ds, per_phone = defaultdict(lambda: {"pos": [None, 0], "neg": [None, 0]}), \
            defaultdict(lambda: {"pos": [None, 0], "neg": [None, 0]}), \
            {"pos": defaultdict(lambda: [None, 0]), "neg": defaultdict(lambda: [None, 0])}
        spk_phone = defaultdict(lambda: {"pos": defaultdict(lambda: [None, 0]),
                                         "neg": defaultdict(lambda: [None, 0])})
        spk_ds = {}
        for ds, spk, st in hc_stats:
            spk_ds[(ds, spk)] = ds
            for cls in ("pos", "neg"):
                for ph, (s, c) in st[f][cls].items():
                    if c == 0:
                        continue
                    for target in (tot[cls], per_spk[(ds, spk)][cls], per_ds[ds][cls],
                                   per_phone[cls][ph]):
                        target[0] = s.copy() if target[0] is None else target[0] + s
                        target[1] += c
                    slot = spk_phone[(ds, spk)][cls][ph]
                    slot[0] = s.copy() if slot[0] is None else slot[0] + s
                    slot[1] += c
        if tot["pos"][1] < MIN_TOKENS or tot["neg"][1] < MIN_TOKENS:
            continue
        mean = lambda sc: sc[0] / sc[1]
        # pooled
        d = norm_dir(mean(tot["pos"]), mean(tot["neg"]))
        if d is not None:
            variants["pooled"][f] = d
        # eq_speaker
        ps = {c: [mean(v[c]) for v in per_spk.values() if v[c][1] >= 1] for c in ("pos", "neg")}
        if len(ps["pos"]) and len(ps["neg"]):
            d = norm_dir(np.mean(ps["pos"], 0), np.mean(ps["neg"], 0))
            if d is not None:
                variants["eq_speaker"][f] = d
        # eq_dataset
        pdm = {c: [mean(v[c]) for v in per_ds.values() if v[c][1] >= 1] for c in ("pos", "neg")}
        if len(pdm["pos"]) and len(pdm["neg"]):
            d = norm_dir(np.mean(pdm["pos"], 0), np.mean(pdm["neg"], 0))
            if d is not None:
                variants["eq_dataset"][f] = d
        # eq_phone (phone type needs >=5 tokens)
        pph = {c: [mean(v) for v in per_phone[c].values() if v[1] >= MIN_TOKENS]
               for c in ("pos", "neg")}
        if len(pph["pos"]) and len(pph["neg"]):
            d = norm_dir(np.mean(pph["pos"], 0), np.mean(pph["neg"], 0))
            if d is not None:
                variants["eq_phone"][f] = d
        # combined: phone means within speaker -> speaker mean -> dataset mean -> global
        ds_acc = {c: defaultdict(list) for c in ("pos", "neg")}
        for key, st in spk_phone.items():
            for c in ("pos", "neg"):
                pm = [v[0] / v[1] for v in st[c].values() if v[1] >= 1]
                if pm:
                    ds_acc[c][spk_ds[key]].append(np.mean(pm, 0))
        cm = {c: [np.mean(v, 0) for v in ds_acc[c].values() if v] for c in ("pos", "neg")}
        if len(cm["pos"]) and len(cm["neg"]):
            d = norm_dir(np.mean(cm["pos"], 0), np.mean(cm["neg"], 0))
            if d is not None:
                variants["combined"][f] = d
    return variants


def dprime(embs_pos, embs_neg, direction):
    if len(embs_pos) < MIN_TOKENS or len(embs_neg) < MIN_TOKENS:
        return np.nan
    pp, pn = embs_pos @ direction, embs_neg @ direction
    sp = np.sqrt((pp.var() + pn.var()) / 2)
    return float(abs(pp.mean() - pn.mean()) / sp) if sp > 1e-8 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--langs", default="en,es,nl,zh,it,sk,pt,de,ta,fr")
    args = ap.parse_args()

    by_ds = scan_cache()
    if args.audit or not args.run:
        print(f"{'dataset':30s} {'lang':4s} {'cached':>6s} {'hc':>4s}")
        for ds in sorted(by_ds):
            lang = DATASET_LANGUAGE.get(ds, "?")
            n = len(by_ds[ds])
            nhc = sum(1 for s, _ in by_ds[ds] if is_hc(ds, s))
            print(f"{ds:30s} {lang:4s} {n:6d} {nhc:4d}")
        if not args.run:
            return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    langs = args.langs.split(",")
    rows = []
    summary = {}
    for lang in langs:
        feat_sets = load_config(lang)
        if not feat_sets:
            summary[lang] = "no usable config"
            continue
        hc_specs = []
        for ds in LANGUAGE_HC_DATASETS.get(lang, []):
            for spk, f in by_ds.get(ds, []):
                if is_hc(ds, spk):
                    hc_specs.append((ds, spk, f))
        if not hc_specs:
            summary[lang] = "no cached HC speakers"
            continue
        print(f"\n=== {lang}: {len(hc_specs)} cached HC speakers", flush=True)
        hc_stats = []
        for ds, spk, f in hc_specs:
            st, _ = speaker_class_stats(f, feat_sets)
            hc_stats.append((ds, spk, st))
        variants = build_directions(hc_stats)
        if lang == "en":
            for drop in LANGUAGE_HC_DATASETS["en"]:
                sub = [x for x in hc_stats if x[0] != drop]
                for v, dirs in build_directions(sub).items():
                    if v == "pooled":
                        variants[f"loco_{drop}"] = dirs
            only_l = [x for x in hc_stats if x[0] == "LibriSpeech_English"]
            only_c = [x for x in hc_stats if x[0] != "LibriSpeech_English"]
            for name, sub in (("only_libri", only_l), ("only_clinical", only_c)):
                for v, dirs in build_directions(sub).items():
                    if v == "pooled":
                        variants[name] = dirs
        summary[lang] = {v: sorted(d.keys()) for v, d in variants.items()}
        # re-score every cached speaker of this language
        targets = [(ds, spk, f) for ds, lst in by_ds.items()
                   if DATASET_LANGUAGE.get(ds) == lang for spk, f in lst]
        print(f"    scoring {len(targets)} speakers x {len(variants)} variants", flush=True)
        for ds, spk, f in targets:
            z = np.load(f, allow_pickle=True)
            phones = [str(p) for p in z["phones"]]
            embs = z["embeddings"].astype(np.float64)
            cls_idx = {feat: {"pos": [], "neg": []} for feat in feat_sets}
            for i, ph in enumerate(phones):
                for feat, (pos, neg) in feat_sets.items():
                    if ph in pos:
                        cls_idx[feat]["pos"].append(i)
                    elif ph in neg:
                        cls_idx[feat]["neg"].append(i)
            for vname, dirs in variants.items():
                rec = {"language": lang, "dataset": ds, "speaker_id": spk,
                       "variant": vname}
                for feat in FEATS:
                    if feat not in dirs or feat not in cls_idx:
                        rec[feat] = ""
                        continue
                    dp = dprime(embs[cls_idx[feat]["pos"]], embs[cls_idx[feat]["neg"]],
                                dirs[feat])
                    rec[feat] = "" if dp != dp else round(dp, 6)
                rows.append(rec)

    import csv as _csv
    outcsv = OUT_DIR / "test4_speaker_dprime.csv"
    with open(outcsv, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["language", "dataset", "speaker_id",
                                           "variant"] + FEATS)
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "test4_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {len(rows)} rows -> {outcsv}")


if __name__ == "__main__":
    main()
