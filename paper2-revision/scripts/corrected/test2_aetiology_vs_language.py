#!/usr/bin/env python
"""Test 2 — same-aetiology vs different-aetiology cross-language profile similarity.

Pre-registered in ANALYSIS_PLAN.md §3 as amended by §15 (v1.2, no-manner sensitivity)
and §16 (v1.3, bootstrap-primary inference after the permutation-degeneracy finding).

Reads ONLY the frozen master. Seed 42. All draws persisted.

Outputs:
  results/test2_delta.json   — every statistic, parameter and diagnostic
  results/test2_draws.npz    — bootstrap and permutation draws
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
MASTER = BASE / "corrected" / "track4_master.csv"
OUT_JSON = BASE / "results" / "test2_delta_corrected.json"
OUT_NPZ = BASE / "results" / "test2_draws_corrected.npz"

SEED = 42
N_BOOT = 10_000          # primary bootstrap draws
N_BOOT_SENS = 2_000      # bootstrap draws for sensitivity CIs
N_PERM = 10_000          # permutation draws
N_REPRO = 1_000          # draws for reproduction of the submitted permutation

FEATS5 = ["nasal_dprime", "voicing_dprime", "sonorant_dprime",
          "strident_dprime", "manner_dprime"]
FEATS4 = [f for f in FEATS5 if f != "manner_dprime"]          # v1.2 sensitivity 6
FEATS9 = FEATS5 + ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]

AET_MAP = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS"}
MIN_CELL = 3             # cell rule, mirrors frozen cross_lingual_aetiology.py
MIN_HC = 3               # HC pool rule, mirrors frozen script line 224
MIN_FEAT_OVERLAP = 3     # cosine NaN-mask rule, mirrors frozen cosine_sim()


def safe_float(v):
    try:
        f = float(v)
        return f if f == f else np.nan
    except (TypeError, ValueError):
        return np.nan


def load(feats):
    """Return dys[(lang,aet)] -> (matrix n x k, dataset list), hc[lang] -> matrix."""
    dys, hc = defaultdict(list), defaultdict(list)
    dys_ds = defaultdict(list)
    with open(MASTER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vec = [safe_float(r[c]) for c in feats]
            if r["aetiology"] == "healthy":
                hc[r["language"]].append(vec)
            elif r["aetiology"] in AET_MAP and r["is_control"].lower() != "true":
                key = (r["language"], AET_MAP[r["aetiology"]])
                dys[key].append(vec)
                dys_ds[key].append(r["dataset"])
    dys = {k: (np.array(v, float), dys_ds[k]) for k, v in dys.items()}
    hc = {k: np.array(v, float) for k, v in hc.items()}
    return dys, hc


def nanmean_profile(mat):
    with np.errstate(invalid="ignore"):
        return np.nanmean(mat, axis=0)


def cos(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < MIN_FEAT_OVERLAP:
        return np.nan
    a, b = a[m], b[m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(a @ b / (na * nb))


def corr_centred(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < MIN_FEAT_OVERLAP:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(a @ b / (na * nb))


def build_profiles(dys, hc, normalise, min_cell=MIN_CELL, drop_lang=None,
                   drop_dataset=None, sim=None):
    """Cell profiles. normalise: per-feature ratio to language HC mean.
    Returns dict[(lang,aet)] -> profile vector."""
    profiles = {}
    hc_means = {}
    for lang, mat in hc.items():
        if len(mat) >= MIN_HC:
            m = nanmean_profile(mat)
            m = np.where(m == 0, 1.0, m)     # zero guard, mirrors frozen script
            hc_means[lang] = m
    for (lang, aet), (mat, ds) in dys.items():
        if drop_lang and lang == drop_lang:
            continue
        if drop_dataset:
            keep = [i for i, d in enumerate(ds) if d != drop_dataset]
            if not keep:
                continue
            mat = mat[keep]
        if len(mat) < min_cell:
            continue
        p = nanmean_profile(mat)
        if normalise:
            if lang not in hc_means:
                continue                     # drops sw (single-speaker HC)
            p = p / hc_means[lang]
        profiles[(lang, aet)] = p
    return profiles


def delta(profiles, simfun=cos):
    """Δ = mean within-aet diff-lang sim − mean between-aet diff-lang sim."""
    keys = sorted(profiles)
    within, between = [], []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            if k1[0] == k2[0]:
                continue                     # same language: excluded (v1.3 §4)
            s = simfun(profiles[k1], profiles[k2])
            if s != s:
                continue
            (within if k1[1] == k2[1] else between).append(s)
    if not within or not between:
        return np.nan, np.nan, np.nan, 0, 0
    return (float(np.mean(within)) - float(np.mean(between)),
            float(np.mean(within)), float(np.mean(between)),
            len(within), len(between))


def resample(rng, dys, hc):
    """Speaker-level bootstrap resample of every cell and HC pool."""
    d2 = {}
    for k, (mat, ds) in dys.items():
        idx = rng.integers(0, len(mat), len(mat))
        d2[k] = (mat[idx], ds)
    h2 = {}
    for k, mat in hc.items():
        idx = rng.integers(0, len(mat), len(mat))
        h2[k] = mat[idx]
    return d2, h2


def resample_two_stage(rng, dys, hc):
    """Dataset-then-speaker resample for multi-dataset cells (sensitivity 7)."""
    d2 = {}
    for k, (mat, ds) in dys.items():
        uds = sorted(set(ds))
        if len(uds) == 1:
            idx = rng.integers(0, len(mat), len(mat))
            d2[k] = (mat[idx], ds)
        else:
            pick = rng.integers(0, len(uds), len(uds))
            rows = []
            for j in pick:
                rows_j = [i for i, d in enumerate(ds) if d == uds[j]]
                rows.extend(rng.choice(rows_j, size=len(rows_j), replace=True))
            d2[k] = (mat[rows], [ds[i] for i in rows])
    h2 = {}
    for k, mat in hc.items():
        idx = rng.integers(0, len(mat), len(mat))
        h2[k] = mat[idx]
    return d2, h2


def boot_ci(rng, dys, hc, normalise, n_draws, two_stage=False, min_cell=MIN_CELL,
            simfun=cos):
    draws = np.empty(n_draws)
    for b in range(n_draws):
        d2, h2 = (resample_two_stage if two_stage else resample)(rng, dys, hc)
        pr = build_profiles(d2, h2, normalise, min_cell=min_cell)
        draws[b] = delta(pr, simfun)[0]
    ok = draws[~np.isnan(draws)]
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(lo), float(hi), draws


def permute(rng, dys, hc, normalise, blocks, n_draws):
    """Permute PD/CP/ALS labels among dysarthric speakers within blocks.
    blocks: 'lang_dataset' (plan §3) or 'lang' (v1.3 tertiary).
    Returns draws + mean fraction of labels moved (degeneracy diagnostic)."""
    # flatten speakers
    recs = []          # (lang, dataset, aet, vec)
    for (lang, aet), (mat, ds) in dys.items():
        for i in range(len(mat)):
            recs.append((lang, ds[i], aet, mat[i]))
    by_block = defaultdict(list)
    for j, r in enumerate(recs):
        key = (r[0], r[1]) if blocks == "lang_dataset" else r[0]
        by_block[key].append(j)
    draws = np.empty(n_draws)
    moved = np.empty(n_draws)
    labels0 = np.array([r[2] for r in recs])
    for b in range(n_draws):
        labels = labels0.copy()
        for _, js in by_block.items():
            js = np.array(js)
            labels[js] = labels0[js[rng.permutation(len(js))]]
        moved[b] = float(np.mean(labels != labels0))
        d2 = defaultdict(list)
        d2ds = defaultdict(list)
        for j, r in enumerate(recs):
            d2[(r[0], labels[j])].append(r[3])
            d2ds[(r[0], labels[j])].append(r[1])
        d2 = {k: (np.array(v), d2ds[k]) for k, v in d2.items()}
        pr = build_profiles(d2, hc, normalise)
        draws[b] = delta(pr)[0]
    return draws, float(np.mean(moved))


def repro_original(rng, dys_raw, hc, feats_label, n_draws):
    """Reproduce the submitted permutation: PD observed mean cross-lang cosine vs
    same-size random subsets of pooled dysarthric speakers within each language.
    Raw profiles (no HC normalisation), mirroring frozen script's Table-2 block."""
    pd_cells = {k: v for k, v in dys_raw.items() if k[1] == "PD" and len(v[0]) >= MIN_CELL}
    prof = {k: nanmean_profile(v[0]) for k, v in pd_cells.items()}
    keys = sorted(prof)
    obs = [cos(prof[k1], prof[k2]) for i, k1 in enumerate(keys) for k2 in keys[i+1:]]
    obs = float(np.nanmean(obs))
    # pooled dysarthric speakers per language (all three aetiologies)
    pool = defaultdict(list)
    for (lang, aet), (mat, _) in dys_raw.items():
        pool[lang].append(mat)
    pool = {k: np.vstack(v) for k, v in pool.items()}
    null = np.empty(n_draws)
    for b in range(n_draws):
        prs = {}
        for (lang, _), (mat, _) in pd_cells.items():
            p = pool[lang]
            idx = rng.choice(len(p), size=len(mat), replace=False) \
                if len(p) >= len(mat) else rng.integers(0, len(p), len(mat))
            prs[lang] = nanmean_profile(p[idx])
        ks = sorted(prs)
        sims = [cos(prs[k1], prs[k2]) for i, k1 in enumerate(ks) for k2 in ks[i+1:]]
        null[b] = np.nanmean(sims)
    p = float(np.mean(null >= obs))
    return {"feats": feats_label, "observed_pd_mean_cosine": obs,
            "null_mean": float(np.mean(null)), "p_null_ge_obs": p,
            "n_draws": n_draws}, null


def main():
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM,
           "master": str(MASTER), "min_cell": MIN_CELL, "min_hc": MIN_HC}

    dys5, hc5 = load(FEATS5)
    out["cells"] = {f"{l}|{a}": len(m[0]) for (l, a), m in sorted(dys5.items())}
    out["hc_pools"] = {l: len(m) for l, m in sorted(hc5.items())}

    npz = {}
    for label, normalise in [("hcnorm", True), ("raw", False)]:
        pr = build_profiles(dys5, hc5, normalise)
        d, w, btw, nw, nb = delta(pr)
        lo, hi, draws = boot_ci(rng, dys5, hc5, normalise, N_BOOT)
        out[f"primary_{label}"] = {
            "cells_used": sorted(f"{l}|{a}" for (l, a) in pr),
            "delta": d, "mean_within": w, "mean_between": btw,
            "n_within_pairs": nw, "n_between_pairs": nb,
            "ci95": [lo, hi], "n_boot": N_BOOT,
        }
        npz[f"boot_{label}"] = draws
        for blocks in ["lang_dataset", "lang"]:
            pdraws, frac = permute(rng, dys5, hc5, normalise, blocks, N_PERM)
            ok = pdraws[~np.isnan(pdraws)]
            out[f"perm_{blocks}_{label}"] = {
                "p_null_ge_obs": float(np.mean(ok >= d)),
                "null_mean": float(np.mean(ok)), "n_valid": int(len(ok)),
                "degeneracy_frac_labels_moved": frac,
            }
            npz[f"perm_{blocks}_{label}"] = pdraws

    # sensitivities on the primary (hcnorm)
    pr0 = build_profiles(dys5, hc5, True)
    langs = sorted({l for (l, a) in pr0})
    sens = {}
    for li, L in enumerate(langs):
        pr = build_profiles(dys5, hc5, True, drop_lang=L)
        d, *_ = delta(pr)
        lo, hi, _ = boot_ci(np.random.default_rng(SEED + 100 + li),
                            {k: v for k, v in dys5.items() if k[0] != L},
                            hc5, True, N_BOOT_SENS)
        sens[f"LOLO_drop_{L}"] = {"delta": d, "ci95": [lo, hi]}
    datasets = sorted({d for (mat, ds) in dys5.values() for d in ds})
    for D in datasets:
        pr = build_profiles(dys5, hc5, True, drop_dataset=D)
        d, *_ = delta(pr)
        sens[f"LODO_drop_{D}"] = {"delta": d}
    pr = build_profiles(dys5, hc5, True, min_cell=5)
    sens["min_cell_5"] = {"delta": delta(pr)[0],
                          "cells_used": sorted(f"{l}|{a}" for (l, a) in pr)}
    sens["corr_centred"] = {"delta": delta(pr0, corr_centred)[0]}
    dys4, hc4 = load(FEATS4)
    pr = build_profiles(dys4, hc4, True)
    d4, w4, b4, *_ = delta(pr)
    lo4, hi4, _ = boot_ci(np.random.default_rng(SEED + 4), dys4, hc4, True, N_BOOT_SENS)
    sens["no_manner_4dim"] = {"delta": d4, "mean_within": w4, "mean_between": b4,
                              "ci95": [lo4, hi4]}
    lo7, hi7, _ = boot_ci(np.random.default_rng(SEED + 7), dys5, hc5, True,
                          N_BOOT_SENS, two_stage=True)
    sens["two_stage_bootstrap"] = {"ci95": [lo7, hi7]}
    out["sensitivities"] = sens

    # reproduction of the submitted permutation (raw, 5 and 9 features)
    rep5, null5 = repro_original(np.random.default_rng(SEED + 5), dys5, hc5,
                                 "5-consonant", N_REPRO)
    dys9, hc9 = load(FEATS9)
    rep9, null9 = repro_original(np.random.default_rng(SEED + 9), dys9, hc9,
                                 "9-dprime", N_REPRO)
    out["repro_submitted_permutation"] = {"feats5": rep5, "feats9": rep9,
        "submitted_values": {"observed": 0.979, "null": 0.982, "p": 0.84}}
    npz["repro_null5"], npz["repro_null9"] = null5, null9

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez_compressed(OUT_NPZ, **npz)
    print(json.dumps(out, indent=1)[:4000])
    print("...\nwritten:", OUT_JSON, OUT_NPZ)


if __name__ == "__main__":
    main()
