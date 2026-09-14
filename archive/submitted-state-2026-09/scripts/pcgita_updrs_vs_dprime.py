#!/usr/bin/env python3
"""Validation check: do UPDRS-III item 18 labels (existing PC-GITA, our reference
ground truth) correlate with the Track 4 d-prime composite?

Existing PC-GITA labels mild/moderate/severe come from UPDRS-III item 18
mapping in scripts/process_pcgita.py:
  UPDRS-18 = 0 or 1 -> mild
  UPDRS-18 = 2     -> moderate
  UPDRS-18 = 3     -> severe
+ HC speakers      -> control

This is a 4-class severity scale that, unlike H&Y, IS speech-specific. If
the d-prime composite is tracking a real speech-degradation signal, it
should correlate with this scale (Spearman rho meaningfully negative,
predicted classes overlapping with truth).

Compares:
  - Existing PC-GITA (UPDRS-18 derived labels, n=100)         <-- speech-specific
  - PC-GITA_OneVoice-MSD26 (H&Y derived labels, n=120 valid)  <-- whole-body motor

Same Paper 3 threshold method (composite = mean of 5 consonant d-primes,
midpoint thresholds from labelled speakers in master CSV).
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
MASTER_FULL = ROOT / "results" / "track4_master.csv"
MASTER_MSD = ROOT / "results" / "track4_master_msd.csv"

CONSONANT_DPRIMES = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime",
    "strident_dprime", "manner_dprime",
]
SEVERITY_ORDER = ["control", "mild", "moderate", "severe"]
SEV_NUM = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}


def get_composite(row):
    vals = []
    for f in CONSONANT_DPRIMES:
        v = row.get(f, "")
        if v in ("", "nan", None):
            continue
        try:
            fv = float(v)
            if not np.isnan(fv):
                vals.append(fv)
        except (ValueError, TypeError):
            pass
    return float(np.mean(vals)) if len(vals) >= 3 else None


def midpoint_thresholds(cent):
    t_severe, t_moderate, t_mild = 0.5, 1.0, 1.5
    if "severe" in cent and "moderate" in cent:
        t_severe = (cent["severe"] + cent["moderate"]) / 2
    elif "severe" in cent and "mild" in cent:
        t_severe = cent["severe"] + (cent["mild"] - cent["severe"]) * 0.33
    elif "severe" in cent:
        t_severe = cent["severe"] * 1.2
    if "moderate" in cent and "mild" in cent:
        t_moderate = (cent["moderate"] + cent["mild"]) / 2
    elif "moderate" in cent and "control" in cent:
        t_moderate = cent["moderate"] + (cent["control"] - cent["moderate"]) * 0.33
    elif "moderate" in cent:
        t_moderate = cent["moderate"] * 1.15
    if "mild" in cent and "control" in cent:
        t_mild = (cent["mild"] + cent["control"]) / 2
    elif "mild" in cent:
        t_mild = cent["mild"] * 1.1
    return (t_severe, t_moderate, t_mild)


def classify(score, thr):
    t_severe, t_moderate, t_mild = thr
    if score < t_severe:
        return "severe"
    if score < t_moderate:
        return "moderate"
    if score < t_mild:
        return "mild"
    return "control"


def confusion(y_true, y_pred, classes):
    n = len(classes)
    M = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


def cohen_kappa(M):
    if M.sum() == 0:
        return float("nan")
    po = np.trace(M) / M.sum()
    rt = M.sum(axis=1) / M.sum()
    ct = M.sum(axis=0) / M.sum()
    pe = float((rt * ct).sum())
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


def lin_wkappa(M):
    if M.sum() == 0:
        return float("nan")
    K = M.shape[0]
    W = np.abs(np.subtract.outer(np.arange(K), np.arange(K))).astype(float) / (K - 1)
    obs = M / M.sum()
    rt = M.sum(axis=1) / M.sum()
    ct = M.sum(axis=0) / M.sum()
    exp = np.outer(rt, ct)
    num = (W * obs).sum()
    den = (W * exp).sum()
    return float("nan") if den == 0 else 1 - num / den


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def calibrate_within(rows, exclude_speakers=None):
    if exclude_speakers is None:
        exclude_speakers = set()
    by_sev = defaultdict(list)
    for r in rows:
        if r["speaker_id"] in exclude_speakers:
            continue
        sev = (r.get("severity_label") or "").strip().lower()
        if sev not in SEVERITY_ORDER:
            continue
        s = get_composite(r)
        if s is None:
            continue
        by_sev[sev].append(s)
    cent = {s: float(np.mean(by_sev[s])) for s in SEVERITY_ORDER if by_sev[s]}
    return midpoint_thresholds(cent), cent


def analyse_corpus(rows, label):
    """Compute correlation, predictions, confusion, kappa for a corpus."""
    valid = []
    for r in rows:
        sev = (r.get("severity_label") or "").strip().lower()
        s = get_composite(r)
        if s is None:
            continue
        valid.append({
            "speaker_id": r["speaker_id"],
            "severity_label": sev,
            "severity_numeric": SEV_NUM.get(sev, -1),
            "composite": s,
            "is_pd": r.get("aetiology", "") == "parkinsons",
            "is_hc": r.get("aetiology", "") == "healthy",
        })

    print(f"\n{'='*80}\n  {label}\n{'='*80}")
    print(f"  total speakers with valid composite: {len(valid)}")
    sev_dist = Counter(v["severity_label"] for v in valid)
    print(f"  severity distribution: {dict(sev_dist)}")

    # 1. Spearman correlation: composite vs severity_numeric (excluding unknown)
    valid_sev = [v for v in valid if v["severity_numeric"] >= 0]
    if len(valid_sev) >= 5:
        rho, p = spearmanr([v["composite"] for v in valid_sev],
                           [v["severity_numeric"] for v in valid_sev])
        print(f"\n  Spearman (composite vs severity_numeric, n={len(valid_sev)}): "
              f"rho={rho:+.3f}, p={p:.2e}")

    # 2. HC vs PD Mann-Whitney
    hc = [v["composite"] for v in valid if v["is_hc"]]
    pd = [v["composite"] for v in valid if v["is_pd"]]
    if len(hc) >= 3 and len(pd) >= 3:
        u, p_u = mannwhitneyu(hc, pd, alternative="two-sided")
        print(f"  HC vs PD (n_hc={len(hc)}, n_pd={len(pd)}): MWU p={p_u:.2e}, "
              f"median HC={np.median(hc):.2f}, median PD={np.median(pd):.2f}")

    # 3. Per-class composite means
    print(f"\n  Composite mean per severity class:")
    by_class = defaultdict(list)
    for v in valid_sev:
        by_class[v["severity_label"]].append(v["composite"])
    for s in SEVERITY_ORDER:
        if s in by_class:
            xs = by_class[s]
            print(f"    {s:>10}  n={len(xs):3d}  mean={np.mean(xs):.3f}  "
                  f"sd={np.std(xs):.3f}  median={np.median(xs):.3f}")

    # 4. Within-corpus calibration + LOSO predictions
    rows_for_cal = [r for r in rows
                    if (r.get("severity_label") or "").strip().lower() in SEVERITY_ORDER]
    thr_in, cent = calibrate_within(rows_for_cal)
    print(f"\n  In-sample centroids: " +
          " ".join(f"{s}={cent[s]:.2f}" for s in SEVERITY_ORDER if s in cent))
    print(f"  In-sample thresholds: severe<{thr_in[0]:.2f}  "
          f"moderate<{thr_in[1]:.2f}  mild<{thr_in[2]:.2f}")

    # LOSO
    loso_pairs = []
    for v in valid_sev:
        thr_i, cent_i = calibrate_within(rows_for_cal,
                                          exclude_speakers={v["speaker_id"]})
        pred = classify(v["composite"], thr_i)
        loso_pairs.append((v["severity_label"], pred))

    M = confusion([t for t, _ in loso_pairs], [p for _, p in loso_pairs],
                  SEVERITY_ORDER)
    print(f"\n  LOSO confusion (rows=truth, cols=predicted):")
    print(f"    {'':>10}  " + "  ".join(f"{c:>8}" for c in SEVERITY_ORDER) + "  total")
    for i, c in enumerate(SEVERITY_ORDER):
        print(f"    {c:>10}  " + "  ".join(f"{v:>8d}" for v in M[i]) +
              f"  {M[i].sum():>5d}")
    n = M.sum()
    if n:
        acc = float(np.trace(M)) / n
        off1 = sum(int(M[i, j]) for i in range(len(SEVERITY_ORDER))
                   for j in range(len(SEVERITY_ORDER))
                   if abs(i - j) <= 1) / n
        print(f"    n={n}  exact={acc:.3f}  off-by-1={off1:.3f}  "
              f"kappa={cohen_kappa(M):.3f}  weighted_kappa={lin_wkappa(M):.3f}")

    return valid_sev


def main():
    print("=" * 80)
    print("UPDRS-18 (existing PC-GITA) vs H&Y (PC-GITA-MSD): which label is")
    print("the d-prime composite tracking?")
    print("=" * 80)

    # Existing PC-GITA from full master
    full = load(MASTER_FULL)
    pcgita_existing = [r for r in full if r["dataset"] == "PC-GITA"]
    analyse_corpus(pcgita_existing, "PC-GITA (existing) — UPDRS-18-derived labels")

    # PC-GITA_OneVoice-MSD26 from MSD master
    msd = load(MASTER_MSD)
    pcgita_msd = [r for r in msd if r["dataset"] == "PC-GITA_OneVoice-MSD26"]
    analyse_corpus(pcgita_msd, "PC-GITA_OneVoice-MSD26 — H&Y-derived labels")

    # Czech and German for completeness
    czech = [r for r in msd if r["dataset"] == "Czech_OneVoice-MSD26"]
    german = [r for r in msd if r["dataset"] == "German_OneVoice-MSD26"]
    analyse_corpus(czech, "Czech_OneVoice-MSD26 — H&Y-derived labels")
    analyse_corpus(german, "German_OneVoice-MSD26 — H&Y-derived labels")


if __name__ == "__main__":
    main()
