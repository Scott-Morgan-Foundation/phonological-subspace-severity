#!/usr/bin/env python
"""Test 3 — per-feature HC-normalised effects by aetiology x language.

Pre-registered in ANALYSIS_PLAN.md §4. For each consonant feature x aetiology x
language cell: SMD (Cohen's d, pooled SD) of HC minus dysarthric speaker-level
d-prime (positive = reduced under dysarthria), with bootstrap 95% CI
(1,000 draws, seed 42; two-stage dataset->speaker where the dys cell or the HC
pool spans multiple datasets, per plan wording). Also the HC-ratio
(dys_mean / hc_mean) for comparability with the submitted Figure 3.

Reads ONLY the frozen master. Outputs:
  results/test3_effects.csv
  results/test3_report.md (forest-style table + sign-consistency counts)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MASTER = BASE / "frozen_package" / "results" / "track4_master.csv"
OUT_CSV = BASE / "results" / "test3_effects.csv"
OUT_MD = BASE / "results" / "test3_report.md"

SEED = 42
N_BOOT = 1_000
FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime",
         "strident_dprime", "manner_dprime"]
SHORT = {f: f.replace("_dprime", "") for f in FEATS}
AET_MAP = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS"}
MIN_CELL = 3
MIN_HC = 3


def safe(v):
    try:
        f = float(v)
        return f if f == f else np.nan
    except (TypeError, ValueError):
        return np.nan


def load():
    dys = defaultdict(lambda: ([], []))   # (lang,aet) -> (rows, datasets)
    hc = defaultdict(lambda: ([], []))    # lang -> (rows, datasets)
    with open(MASTER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vec = [safe(r[c]) for c in FEATS]
            if r["aetiology"] == "healthy":
                hc[r["language"]][0].append(vec)
                hc[r["language"]][1].append(r["dataset"])
            elif r["aetiology"] in AET_MAP and r["is_control"].lower() != "true":
                k = (r["language"], AET_MAP[r["aetiology"]])
                dys[k][0].append(vec)
                dys[k][1].append(r["dataset"])
    dys = {k: (np.array(v[0], float), v[1]) for k, v in dys.items() if len(v[0]) >= MIN_CELL}
    hc = {k: (np.array(v[0], float), v[1]) for k, v in hc.items() if len(v[0]) >= MIN_HC}
    return dys, hc


def smd(hc_vals, dys_vals):
    """Cohen's d with pooled SD: positive = reduced under dysarthria."""
    h = hc_vals[~np.isnan(hc_vals)]
    d = dys_vals[~np.isnan(dys_vals)]
    if len(h) < 2 or len(d) < 2:
        return np.nan
    sp = np.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(d) - 1) * d.var(ddof=1))
                 / (len(h) + len(d) - 2))
    if sp < 1e-12:
        return np.nan
    return float((h.mean() - d.mean()) / sp)


def two_stage_idx(rng, ds_list):
    uds = sorted(set(ds_list))
    if len(uds) == 1:
        return rng.integers(0, len(ds_list), len(ds_list))
    pick = rng.integers(0, len(uds), len(uds))
    rows = []
    for j in pick:
        rows_j = [i for i, d in enumerate(ds_list) if d == uds[j]]
        rows.extend(rng.choice(rows_j, size=len(rows_j), replace=True))
    return np.array(rows)


def main():
    dys, hc = load()
    rows_out = []
    rng = np.random.default_rng(SEED)

    for (lang, aet) in sorted(dys):
        if lang not in hc:
            rows_out.append({"aetiology": aet, "language": lang, "feature": "ALL",
                             "note": "no HC pool >=3 (excluded)", "n_dys": len(dys[(lang, aet)][0]),
                             "n_hc": 0, "smd": "", "ci_lo": "", "ci_hi": "", "hc_ratio": ""})
            continue
        dmat, dds = dys[(lang, aet)]
        hmat, hds = hc[lang]
        multi = len(set(dds)) > 1 or len(set(hds)) > 1
        for fi, f in enumerate(FEATS):
            obs = smd(hmat[:, fi], dmat[:, fi])
            hm = np.nanmean(hmat[:, fi])
            ratio = float(np.nanmean(dmat[:, fi]) / hm) if hm not in (0,) and hm == hm else np.nan
            draws = np.empty(N_BOOT)
            for b in range(N_BOOT):
                di = two_stage_idx(rng, dds) if multi else rng.integers(0, len(dmat), len(dmat))
                hi_ = two_stage_idx(rng, hds) if multi else rng.integers(0, len(hmat), len(hmat))
                draws[b] = smd(hmat[hi_, fi], dmat[di, fi])
            ok = draws[~np.isnan(draws)]
            lo, hi2 = (np.percentile(ok, [2.5, 97.5]) if len(ok) > 100 else (np.nan, np.nan))
            rows_out.append({"aetiology": aet, "language": lang, "feature": SHORT[f],
                             "n_dys": int(np.sum(~np.isnan(dmat[:, fi]))),
                             "n_hc": int(np.sum(~np.isnan(hmat[:, fi]))),
                             "smd": round(obs, 4) if obs == obs else "",
                             "ci_lo": round(float(lo), 4) if lo == lo else "",
                             "ci_hi": round(float(hi2), 4) if hi2 == hi2 else "",
                             "hc_ratio": round(ratio, 4) if ratio == ratio else "",
                             "bootstrap": "two-stage" if multi else "speaker",
                             "note": ""})

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # report: forest table + sign consistency
    lines = ["# Test 3 — Per-feature HC-normalised effects by aetiology x language",
             "",
             f"Run 2026-09-13, seed {SEED}, {N_BOOT} bootstrap draws (two-stage dataset->speaker "
             "where the dysarthric cell or HC pool spans multiple datasets). "
             "SMD = Cohen's d of HC minus dysarthric speaker-level d-prime — positive = contrast "
             "REDUCED under dysarthria. hc_ratio = dys_mean / hc_mean (Figure-3 convention, 1.0 = healthy). "
             "Full numbers: `test3_effects.csv`.",
             ""]
    by_aet = defaultdict(lambda: defaultdict(dict))
    for r in rows_out:
        if r["feature"] not in SHORT.values():
            continue
        by_aet[r["aetiology"]][r["feature"]][r["language"]] = r
    for aet in ("PD", "CP", "ALS"):
        feats = by_aet[aet]
        langs = sorted({l for f in feats.values() for l in f})
        lines.append(f"## {aet}  (languages with HC pool: {', '.join(langs)})\n")
        lines.append("| Feature | " + " | ".join(langs) + " | sign + | CI excl. 0 |")
        lines.append("|---" * (len(langs) + 3) + "|")
        for f in SHORT.values():
            cells = feats.get(f, {})
            row = [f]
            pos = ex0 = n = 0
            for l in langs:
                r = cells.get(l)
                if not r or r["smd"] == "":
                    row.append("—")
                    continue
                n += 1
                s = float(r["smd"]); lo = float(r["ci_lo"]); hi2 = float(r["ci_hi"])
                pos += s > 0
                exc = lo > 0 or hi2 < 0
                ex0 += exc
                row.append(f"{s:+.2f} [{lo:+.2f},{hi2:+.2f}]" + ("*" if exc else ""))
            row.append(f"{pos}/{n}")
            row.append(f"{ex0}/{n}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.append("`*` = 95% CI excludes 0. `sign +` = languages where the contrast is reduced "
                 "(SMD > 0). Cells without a valid language HC pool (>=3 speakers) are excluded "
                 "(sw: single HC speaker).")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))
    print("\nwritten:", OUT_CSV, OUT_MD)


if __name__ == "__main__":
    main()
