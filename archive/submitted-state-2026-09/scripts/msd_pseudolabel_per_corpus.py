#!/usr/bin/env python3
"""Per-corpus within-bundle calibration of the Paper 3 pseudo-label method.

Same composite (mean of 5 consonant d-primes) and same midpoint-threshold
classification as msd_pseudolabel_paper3.py, but thresholds are derived
ONLY from each corpus's own labelled speakers.

This is intentionally an in-sample fit-and-evaluate (we use the H&Y-derived
labels we are testing against to calibrate the thresholds, then predict
the same speakers). That biases accuracy upward by exactly the amount
that thresholds can be tuned to recording protocol; it isolates "recording
protocol drift" from "label noise" relative to the cross-corpus run.

Reading guide:
  - Per-corpus kappa MUCH > cross-corpus kappa  -> recording-protocol drift
    is dominant; Paper 3 method is fine, just needs per-corpus anchoring.
  - Per-corpus kappa still low  -> the H&Y labels themselves are too noisy
    for the d-prime composite to track them, regardless of thresholds.

Also reports leave-one-speaker-out (LOSO) versions of the per-corpus
accuracy so the in-sample bias is bounded — for each speaker, recompute
thresholds from the OTHER labelled speakers in that corpus, then predict.

Outputs:
  - track4/results/msd_pseudolabels_per_corpus.csv
  - track4/results/msd_pseudolabel_per_corpus_vs_truth.csv
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MASTER_MSD = ROOT / "results" / "track4_master_msd.csv"
OUT_PREDS = ROOT / "results" / "msd_pseudolabels_per_corpus.csv"
OUT_CMP = ROOT / "results" / "msd_pseudolabel_per_corpus_vs_truth.csv"

CONSONANT_DPRIMES = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime",
    "strident_dprime", "manner_dprime",
]
SEVERITY_ORDER = ["control", "mild", "moderate", "severe"]

MSD_DATASETS = ["Czech_OneVoice-MSD26", "German_OneVoice-MSD26",
                "PC-GITA_OneVoice-MSD26"]


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
    row_t = M.sum(axis=1) / M.sum()
    col_t = M.sum(axis=0) / M.sum()
    pe = float((row_t * col_t).sum())
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


def linear_weighted_kappa(M):
    if M.sum() == 0:
        return float("nan")
    K = M.shape[0]
    W = np.abs(np.subtract.outer(np.arange(K), np.arange(K))).astype(float) / (K - 1)
    obs = M / M.sum()
    row_t = M.sum(axis=1) / M.sum()
    col_t = M.sum(axis=0) / M.sum()
    exp = np.outer(row_t, col_t)
    num = (W * obs).sum()
    den = (W * exp).sum()
    return float("nan") if den == 0 else 1 - num / den


def calibrate_corpus(spk_data, exclude_speakers=None):
    """Compute centroids and thresholds from one corpus's labelled speakers.
    spk_data: list of (speaker_id, score, severity) tuples (severity is real label).
    Returns (thresholds, centroids dict, n_used)."""
    if exclude_speakers is None:
        exclude_speakers = set()
    by_sev = defaultdict(list)
    for spk, score, sev in spk_data:
        if spk in exclude_speakers:
            continue
        if sev not in SEVERITY_ORDER:
            continue
        if score is None:
            continue
        by_sev[sev].append(score)
    cent = {s: float(np.mean(by_sev[s])) for s in SEVERITY_ORDER if by_sev[s]}
    thr = midpoint_thresholds(cent)
    n = sum(len(v) for v in by_sev.values())
    return thr, cent, n


def print_confusion(M, classes, label):
    print(f"\n  {label}  (rows = ground truth, cols = predicted)")
    print(f"    {'':>10}  " + "  ".join(f"{c:>8}" for c in classes) + "  total")
    for i, c in enumerate(classes):
        row = M[i]
        print(f"    {c:>10}  " + "  ".join(f"{v:>8d}" for v in row) +
              f"  {row.sum():>5d}")
    n = M.sum()
    if n:
        acc = float(np.trace(M)) / n
        off1 = sum(int(M[i, j]) for i in range(M.shape[0])
                   for j in range(M.shape[1]) if abs(i - j) <= 1) / n
        kappa = cohen_kappa(M)
        wkappa = linear_weighted_kappa(M)
        print(f"    n={n}  exact={acc:.3f}  off-by-1={off1:.3f}  "
              f"kappa={kappa:.3f}  weighted_kappa={wkappa:.3f}")


def main():
    print("=" * 80)
    print("Per-corpus within-bundle calibration of Paper 3 pseudo-labels")
    print("=" * 80)

    rows = []
    with open(MASTER_MSD, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    pred_rows = []
    cmp_rows = []

    for ds in MSD_DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        spk_scores = [(r["speaker_id"], get_composite(r),
                       (r.get("severity_label") or "").strip().lower())
                      for r in ds_rows]

        # In-sample calibration (uses all labelled speakers in the corpus)
        thr, cent, n_used = calibrate_corpus(spk_scores)
        print(f"\n--- {ds}  (n_labelled_used={n_used}) ---")
        print(f"  centroids: " +
              " ".join(f"{s}={cent[s]:.3f}" for s in SEVERITY_ORDER if s in cent))
        print(f"  thresholds: severe<{thr[0]:.3f}  "
              f"moderate<{thr[1]:.3f}  mild<{thr[2]:.3f}  control>=")

        # In-sample predictions
        in_preds = []
        for spk, score, sev_truth in spk_scores:
            if score is None:
                pred = "unknown"
            else:
                pred = classify(score, thr)
            in_preds.append((spk, score, sev_truth, pred))

        # Leave-one-out predictions
        loso_preds = []
        for i, (spk, score, sev_truth) in enumerate(spk_scores):
            if score is None:
                loso_preds.append((spk, score, sev_truth, "unknown", None))
                continue
            thr_i, cent_i, n_i = calibrate_corpus(spk_scores,
                                                   exclude_speakers={spk})
            # If degenerate (e.g. removing the only severe speaker), fall back
            # to in-sample thr
            thr_use = thr_i if n_i >= 4 else thr
            pred = classify(score, thr_use)
            loso_preds.append((spk, score, sev_truth, pred, thr_use))

        # In-sample confusion
        valid_in = [(t, p) for _, _, t, p in in_preds
                    if t in SEVERITY_ORDER and p in SEVERITY_ORDER]
        if valid_in:
            yt = [t for t, _ in valid_in]
            yp = [p for _, p in valid_in]
            M_in = confusion(yt, yp, SEVERITY_ORDER)
            print_confusion(M_in, SEVERITY_ORDER, "[IN-SAMPLE] confusion")

        # LOSO confusion
        valid_loso = [(t, p) for _, _, t, p, _ in loso_preds
                      if t in SEVERITY_ORDER and p in SEVERITY_ORDER]
        if valid_loso:
            yt = [t for t, _ in valid_loso]
            yp = [p for _, p in valid_loso]
            M_loso = confusion(yt, yp, SEVERITY_ORDER)
            print_confusion(M_loso, SEVERITY_ORDER, "[LOSO]      confusion")

        # Save per-speaker predictions (LOSO version is the honest one)
        for (spk, score, sev_truth, pred_in), (_, _, _, pred_loso, thr_loso) \
                in zip(in_preds, loso_preds):
            pred_rows.append({
                "dataset": ds,
                "speaker_id": spk,
                "ground_truth": sev_truth,
                "composite_dprime": f"{score:.3f}" if score is not None else "",
                "pred_in_sample": pred_in,
                "pred_loso": pred_loso,
                "thr_severe_corpus": f"{thr[0]:.3f}",
                "thr_moderate_corpus": f"{thr[1]:.3f}",
                "thr_mild_corpus": f"{thr[2]:.3f}",
            })

        # Comparison row
        for label, M, valid in [("in_sample", M_in if valid_in else None, valid_in),
                                  ("loso", M_loso if valid_loso else None, valid_loso)]:
            if M is None or not valid:
                continue
            n = M.sum()
            acc = float(np.trace(M)) / n
            off1 = sum(int(M[i, j]) for i in range(len(SEVERITY_ORDER))
                       for j in range(len(SEVERITY_ORDER))
                       if abs(i - j) <= 1) / n
            cmp_rows.append({
                "dataset": ds,
                "mode": label,
                "n_compared": n,
                "exact_accuracy": f"{acc:.3f}",
                "off_by_1_accuracy": f"{off1:.3f}",
                "cohen_kappa": f"{cohen_kappa(M):.3f}",
                "weighted_kappa_linear": f"{linear_weighted_kappa(M):.3f}",
                **{f"M_{a}_to_{b}": int(M[i, j])
                   for i, a in enumerate(SEVERITY_ORDER)
                   for j, b in enumerate(SEVERITY_ORDER)},
            })

    # Pooled (use LOSO predictions)
    pooled_valid = [(p["ground_truth"], p["pred_loso"]) for p in pred_rows
                    if p["ground_truth"] in SEVERITY_ORDER
                    and p["pred_loso"] in SEVERITY_ORDER]
    if pooled_valid:
        yt = [t for t, _ in pooled_valid]
        yp = [p for _, p in pooled_valid]
        M = confusion(yt, yp, SEVERITY_ORDER)
        print(f"\n--- POOLED (LOSO across 3 corpora, n={len(pooled_valid)}) ---")
        print_confusion(M, SEVERITY_ORDER, "[LOSO POOLED] confusion")
        n = M.sum()
        cmp_rows.append({
            "dataset": "POOLED",
            "mode": "loso",
            "n_compared": n,
            "exact_accuracy": f"{float(np.trace(M)) / n:.3f}",
            "off_by_1_accuracy": f"{sum(int(M[i, j]) for i in range(len(SEVERITY_ORDER)) for j in range(len(SEVERITY_ORDER)) if abs(i - j) <= 1) / n:.3f}",
            "cohen_kappa": f"{cohen_kappa(M):.3f}",
            "weighted_kappa_linear": f"{linear_weighted_kappa(M):.3f}",
            **{f"M_{a}_to_{b}": int(M[i, j])
               for i, a in enumerate(SEVERITY_ORDER)
               for j, b in enumerate(SEVERITY_ORDER)},
        })

    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PREDS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        w.writerows(pred_rows)
    print(f"\n  wrote {len(pred_rows)} per-speaker rows -> {OUT_PREDS}")

    if cmp_rows:
        with open(OUT_CMP, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
            w.writeheader()
            w.writerows(cmp_rows)
        print(f"  wrote comparison -> {OUT_CMP}")

    # Headline summary
    print(f"\n{'='*80}\nSUMMARY (in-sample is biased upward; LOSO is the honest signal)\n{'='*80}")
    print(f"{'dataset':<28} {'mode':<10} {'n':>4} {'exact':>6} {'off1':>6} "
          f"{'kappa':>7} {'wkappa':>7}")
    print("-" * 75)
    for r in cmp_rows:
        print(f"{r['dataset']:<28} {r['mode']:<10} {r['n_compared']:>4d} "
              f"{r['exact_accuracy']:>6} {r['off_by_1_accuracy']:>6} "
              f"{r['cohen_kappa']:>7} {r['weighted_kappa_linear']:>7}")


if __name__ == "__main__":
    main()
