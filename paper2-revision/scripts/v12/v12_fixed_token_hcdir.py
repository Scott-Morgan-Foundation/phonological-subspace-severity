#!/usr/bin/env python3
"""v12 (CSL R2, Reviewer 2 point 2): fixed-token d-prime along the HC-derived directions.

Extends frozen_package/scripts/fixed_token_dprime.py Experiment A. Loop order, filters,
token budgets, repeats and the RNG draw sequence (RandomState(42), one choice() for pos then
neg per repeat) are identical, so every draw is scored on the same subsampled tokens two ways:
  own : the original estimator (axis joining the speaker's two subsampled class means,
        ddof=1) -> must reproduce the v9 corrected log exactly
  hc  : the main-method estimator (Section 3.1): projection on the language's healthy-control
        direction (token-pooled HC class means, L2-normalised, >=5 tokens per class),
        |mean gap| / sqrt((var_pos + var_neg)/2), ddof=0
plus, for the same speakers, 'hc_full': the HC-direction d' from all tokens (no budget).
So own -> hc isolates the direction; hc -> hc_full isolates the token budget.

HC directions are rebuilt from the embedding cache per language from LANGUAGE_HC_DATASETS
(extract_features.py), HC = master aetiology 'healthy'; phone classes from the same config the
fixed-token script uses (LANG_MAP). A faithfulness check compares hc_full against the master's
published per-contrast d' columns.

Usage (DGX01): DYSARTHRIA_BASE=~/p2v9_originals/corrected python v12_fixed_token_hcdir.py OUT_DIR
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(os.environ["DYSARTHRIA_BASE"])
MASTER = BASE / "results" / "track4" / "track4_master.csv"
EMB_DIR = BASE / "results" / "track4" / "embeddings"
CONFIG_DIR = BASE / "config"
OUT = Path(sys.argv[1])

SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
MAIN_AETS = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}
CONS_CONTRASTS = ["nasal", "voicing", "sonorant", "strident", "manner"]
TOKEN_BUDGETS = [20, 50, 100, 200]
N_REPEATS = 50
RNG = np.random.RandomState(42)
MIN_TOKENS = 5
VARIANTS = ["own", "hc", "hc_full"]

LANG_MAP = {"en": "en", "nl": "nl", "es": "es", "fr": "fr", "zh": "zh", "it": "it", "de": "de",
            "pt": "pt", "hu": "hu", "ta": "ta", "sk": "de", "sw": "en"}
LANGUAGE_HC_DATASETS = {
    "en": ["LibriSpeech_English", "TORGO", "UASPEECH_control"],
    "nl": ["COPAS"], "fr": ["YouTube_French"], "es": ["Neurovoz", "PC-GITA"],
    "zh": ["MDSC"], "it": ["IPVS", "VOC-ALS"], "sk": ["EWA-DB"],
    "hu": ["Hungarian_HC", "CV_Hungarian"], "de": ["YouTube_German", "SVD"],
    "ta": ["SLR65_Tamil", "SSNCE_Tamil"], "pt": ["AVFAD"], "sw": ["CDLI_Kenyan_Swahili"],
}


def load_phone_features(language):
    p = CONFIG_DIR / f"phone_features_{language}.json"
    return json.loads(p.read_text()) if p.exists() else None


def get_phone_classes(pf, contrast):
    m = pf.get("consonant_features", {}).get(contrast, {})
    return set(m.get("positive", [])), set(m.get("negative", []))


def dprime_own(ep, en):
    """Verbatim estimator of fixed_token_dprime.py."""
    if len(ep) < 2 or len(en) < 2:
        return np.nan
    mp, mn = np.mean(ep, axis=0), np.mean(en, axis=0)
    d = mp - mn
    nrm = np.linalg.norm(d)
    if nrm < 1e-10:
        return 0.0
    d = d / nrm
    pp, pn = ep @ d, en @ d
    s = np.sqrt((np.var(pp, ddof=1) + np.var(pn, ddof=1)) / 2)
    return 0.0 if s < 1e-10 else (np.mean(pp) - np.mean(pn)) / s


def dprime_hc(ep, en, direction):
    """Main-method estimator (extract_features.compute_dprime)."""
    if len(ep) < MIN_TOKENS or len(en) < MIN_TOKENS:
        return np.nan
    pp, pn = ep @ direction, en @ direction
    s = np.sqrt((pp.var() + pn.var()) / 2)
    return float(abs(pp.mean() - pn.mean()) / s) if s > 1e-8 else 0.0


def build_hc_directions(rows, phone_feats):
    hc_keys = {f"{r['dataset']}__{r['speaker_id']}" for r in rows if r["aetiology"] == "healthy"}
    dirs, info = {}, {}
    for lang, dsets in LANGUAGE_HC_DATASETS.items():
        pf = phone_feats.get(LANG_MAP.get(lang, "en"))
        if pf is None:
            continue
        acc = {c: {"pos": [None, 0], "neg": [None, 0]} for c in CONS_CONTRASTS}
        n_spk = 0
        for f in sorted(EMB_DIR.glob("*.npz")):
            ds = f.stem.split("__", 1)[0]
            if ds not in dsets or f.stem not in hc_keys:
                continue
            z = np.load(f, allow_pickle=True)
            phones, embs = z["phones"], z["embeddings"]
            n_spk += 1
            for c in CONS_CONTRASTS:
                pos, neg = get_phone_classes(pf, c)
                for cls, s in (("pos", pos), ("neg", neg)):
                    idx = [i for i, p in enumerate(phones) if p in s]
                    if not idx:
                        continue
                    v = embs[idx].astype(np.float64).sum(0)
                    a = acc[c][cls]
                    a[0] = v if a[0] is None else a[0] + v
                    a[1] += len(idx)
        dirs[lang], info[lang] = {}, {"hc_speakers": n_spk, "tokens": {}}
        for c in CONS_CONTRASTS:
            p, n = acc[c]["pos"], acc[c]["neg"]
            info[lang]["tokens"][c] = [p[1], n[1]]
            if p[1] < MIN_TOKENS or n[1] < MIN_TOKENS:
                continue
            d = p[0] / p[1] - n[0] / n[1]
            nrm = np.linalg.norm(d)
            if nrm > 1e-8:
                dirs[lang][c] = d / nrm
        print(f"  {lang}: {n_spk} HC speakers, directions {sorted(dirs[lang])}", flush=True)
    return dirs, info


def group_stats(recs, var):
    ok = [r for r in recs if r[var] is not None]
    out = {"n": len(ok)}
    if len(ok) < 20:
        return out
    rho, p = stats.spearmanr([r["severity"] for r in ok], [r[var] for r in ok])
    out.update(rho=float(rho), p=float(p))
    by = defaultdict(list)
    for r in ok:
        by[r["severity"]].append(r[var])
    out["sev_means"] = {["control", "mild", "moderate", "severe"][k]: [float(np.mean(v)), len(v)]
                        for k, v in sorted(by.items())}
    m = [np.mean(by[k]) for k in sorted(by)]
    out["monotonic"] = bool(all(a > b for a, b in zip(m, m[1:])))
    ag = defaultdict(list)
    for r in ok:
        ag[r["aetiology"]].append(r[var])
    valid = {AET_SHORT[a]: g for a, g in ag.items() if len(g) >= 5}
    if len(valid) >= 2:
        H, pk = stats.kruskal(*valid.values())
        N, k = sum(map(len, valid.values())), len(valid)
        out.update(kw_groups={a: len(g) for a, g in valid.items()}, kw_H=float(H), kw_p=float(pk),
                   kw_N=N, eps2=float(max((H - k + 1) / (N - k), 0)))
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
    meta = {f"{r['dataset']}__{r['speaker_id']}": r for r in rows}
    phone_feats = {}
    for lang in LANG_MAP.values():
        if lang not in phone_feats:
            pf = load_phone_features(lang)
            if pf:
                phone_feats[lang] = pf
    print("Building HC directions", flush=True)
    hc_dirs, hc_info = build_hc_directions(rows, phone_feats)

    per_budget, speaker_rows, faith = {}, [], defaultdict(list)
    npz_files = sorted(EMB_DIR.glob("*.npz"))
    for budget in TOKEN_BUDGETS:
        print(f"--- budget {budget} ---", flush=True)
        recs, n_skipped, n_no_hcdir = [], 0, 0
        for f in npz_files:
            m = meta.get(f.stem)
            if m is None:
                continue
            lang = m.get("language", "en")
            pf = phone_feats.get(LANG_MAP.get(lang, "en"))
            if pf is None:
                n_skipped += 1
                continue
            sev, aet = m.get("severity_label", "unknown"), m.get("aetiology", "unknown")
            if sev not in SEV_MAP or aet not in MAIN_AETS:
                continue
            z = np.load(f, allow_pickle=True)
            phones, embs = z["phones"], z["embeddings"]
            ldirs = hc_dirs.get(lang, {})
            vals = {v: [] for v in VARIANTS}
            for c in CONS_CONTRASTS:
                pos, neg = get_phone_classes(pf, c)
                if not pos or not neg:
                    for v in VARIANTS:
                        vals[v].append(np.nan)
                    continue
                pi = [i for i, p in enumerate(phones) if p in pos]
                ni = [i for i, p in enumerate(phones) if p in neg]
                if len(pi) < budget or len(ni) < budget:
                    for v in VARIANTS:
                        vals[v].append(np.nan)
                    continue
                d = ldirs.get(c)
                own, hc = [], []
                for _ in range(N_REPEATS):
                    sp = RNG.choice(pi, size=budget, replace=False)
                    sn = RNG.choice(ni, size=budget, replace=False)
                    own.append(dprime_own(embs[sp], embs[sn]))
                    if d is not None:
                        hc.append(dprime_hc(embs[sp], embs[sn], d))
                vals["own"].append(np.mean(own))
                if d is None:
                    n_no_hcdir += 1
                    vals["hc"].append(np.nan)
                    vals["hc_full"].append(np.nan)
                else:
                    vals["hc"].append(np.mean(hc))
                    full = dprime_hc(embs[pi], embs[ni], d)
                    vals["hc_full"].append(full)
                    pub = m.get(f"{c}_dprime", "")
                    if budget == TOKEN_BUDGETS[0] and pub not in ("", "nan"):
                        faith[lang].append((full, float(pub)))
            rec = {"speaker": f.stem, "language": lang, "aetiology": aet, "severity": SEV_MAP[sev]}
            for v in VARIANTS:
                good = [x for x in vals[v] if not np.isnan(x)]
                rec[v] = float(np.mean(good)) if len(good) >= 3 else None
                rec[f"{v}_ncontrasts"] = len(good)
            if rec["own"] is not None:
                recs.append(rec)
        res = {"processed_own": len(recs), "skipped": n_skipped, "contrasts_without_hc_direction": n_no_hcdir}
        for v in VARIANTS:
            res[v] = group_stats(recs, v)
        both = [r for r in recs if r["hc"] is not None]
        if len(both) > 2:
            res["spearman_own_vs_hc"] = float(stats.spearmanr([r["own"] for r in both], [r["hc"] for r in both])[0])
            res["spearman_hc_vs_hcfull"] = float(stats.spearmanr([r["hc"] for r in both], [r["hc_full"] for r in both])[0])
        per_budget[budget] = res
        for r in recs:
            speaker_rows.append({"budget": budget, **r})
        print(json.dumps({v: {k: res[v].get(k) for k in ("n", "rho", "eps2")} for v in VARIANTS}), flush=True)

    # common speaker set across all budgets, per variant (robustness_pass5 Exp 1 analogue)
    common = {}
    for v in VARIANTS:
        sets = [{r["speaker"] for r in speaker_rows if r["budget"] == b and r[v] is not None} for b in TOKEN_BUDGETS]
        keep = set.intersection(*sets)
        common[v] = {"n": len(keep)}
        for b in TOKEN_BUDGETS:
            sub = [r for r in speaker_rows if r["budget"] == b and r["speaker"] in keep]
            rho, p = stats.spearmanr([r["severity"] for r in sub], [r[v] for r in sub])
            common[v][b] = {"rho": float(rho), "p": float(p)}

    faithfulness = {}
    for lang, pairs in faith.items():
        a, b = np.array(pairs).T
        faithfulness[lang] = {"n_values": len(pairs), "spearman": float(stats.spearmanr(a, b)[0]),
                              "median_abs_diff": float(np.median(np.abs(a - b)))}

    out = {"master_sha_note": "staged corrected master", "hc_directions": hc_info,
           "faithfulness_hc_full_vs_published": faithfulness, "per_budget": per_budget,
           "common_speaker_set": common, "elapsed_s": round(time.time() - t0)}
    (OUT / "v12_fixed_token_hcdir.json").write_text(json.dumps(out, indent=2))
    with open(OUT / "v12_fixed_token_hcdir_speakers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(speaker_rows[0].keys()))
        w.writeheader()
        w.writerows(speaker_rows)
    print(f"DONE in {out['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
