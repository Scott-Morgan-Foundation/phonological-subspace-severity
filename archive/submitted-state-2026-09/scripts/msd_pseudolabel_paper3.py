#!/usr/bin/env python3
"""Apply the Paper 3 pseudo-label threshold method to the MSD bundle and
compare to the H&Y-derived ground-truth severity labels.

Method (mirrors track4/scripts/apply_pseudolabels_to_master.py):

  1. composite_dprime = mean of 5 consonant d-primes (nasal, voicing,
     sonorant, strident, manner); requires >= 3 valid features.
  2. From the labelled speakers in the FULL Track 4 master CSV
     (track4_master.csv), compute per-language severity centroids
     (mean composite per language x severity class) and global centroids.
  3. Thresholds = midpoints between adjacent severity centroids
     (severe<-->moderate, moderate<-->mild, mild<-->control).
  4. Per-language thresholds are used when n>=20 labelled speakers in that
     language; otherwise fall back to global.
  5. Apply to the MSD bundle (415 speakers), output predicted severity,
     compare to H&Y-derived ground truth.

The MSD speakers are EXCLUDED from threshold calibration (we want to
evaluate the SAP-trained thresholds, not fit-on-test). For Spanish, the
existing PC-GITA labelled speakers (UPDRS-18-derived) DO contribute to
the es-threshold — that is consistent with Paper 3's deployment.

Outputs:
  - track4/results/msd_pseudolabels_paper3.csv  (per-speaker predictions)
  - track4/results/msd_pseudolabel_vs_truth.csv (confusion matrix per ds)
  - console summary: confusion matrices + accuracy / kappa / off-by-one
"""
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MASTER_FULL = ROOT / "results" / "track4_master.csv"
MASTER_MSD = ROOT / "results" / "track4_master_msd.csv"
OUT_PREDS = ROOT / "results" / "msd_pseudolabels_paper3.csv"
OUT_CMP = ROOT / "results" / "msd_pseudolabel_vs_truth.csv"

CONSONANT_DPRIMES = [
    "nasal_dprime", "voicing_dprime", "sonorant_dprime",
    "strident_dprime", "manner_dprime",
]
SEVERITY_ORDER = ["control", "mild", "moderate", "severe"]
SEV_NUM = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}

DATASET_LANGUAGE = {
    "SAP": "en", "COPAS": "nl", "TORGO": "en", "Neurovoz": "es",
    "YouTube_French": "fr", "MDSC": "zh", "IPVS": "it",
    "VOC-ALS": "it", "PC-GITA": "es",
    "UASPEECH": "en", "UASPEECH_control": "en",
    "LibriSpeech_English": "en", "EWA-DB": "sk",
    "Hungarian_Dysarthria": "hu", "Hungarian_HC": "hu",
    "CDSD": "zh", "YouTube_German": "de", "SVD": "de",
    "EasyCall": "it", "Domotica": "nl",
    "SSNCE_Tamil": "ta", "SLR65_Tamil": "ta", "AVFAD": "pt",
    "CDLI_Kenyan_Swahili": "sw", "CHASING": "nl", "TreasureHunters1": "nl",
    # MSD bundle
    "Czech_OneVoice-MSD26": "cs",
    "German_OneVoice-MSD26": "de",
    "PC-GITA_OneVoice-MSD26": "es",
}

MSD_DATASETS = {"Czech_OneVoice-MSD26", "German_OneVoice-MSD26",
                "PC-GITA_OneVoice-MSD26"}


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
    """Compute (t_severe, t_moderate, t_mild) midpoint thresholds from centroids.
    Composite decreases with severity: control > mild > moderate > severe."""
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


def cohen_kappa(y_true, y_pred, classes):
    n = len(y_true)
    if n == 0:
        return float("nan")
    label2idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    M = np.zeros((K, K), dtype=float)
    for t, p in zip(y_true, y_pred):
        if t in label2idx and p in label2idx:
            M[label2idx[t], label2idx[p]] += 1
    if M.sum() == 0:
        return float("nan")
    po = np.trace(M) / M.sum()
    row_t = M.sum(axis=1) / M.sum()
    col_t = M.sum(axis=0) / M.sum()
    pe = float((row_t * col_t).sum())
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def linear_weighted_kappa(y_true, y_pred, classes):
    """Linear-weighted kappa: penalises larger ordinal errors more."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    label2idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    M = np.zeros((K, K), dtype=float)
    for t, p in zip(y_true, y_pred):
        if t in label2idx and p in label2idx:
            M[label2idx[t], label2idx[p]] += 1
    if M.sum() == 0:
        return float("nan")
    W = np.abs(np.subtract.outer(np.arange(K), np.arange(K))).astype(float) / (K - 1)
    obs = M / M.sum()
    row_t = M.sum(axis=1) / M.sum()
    col_t = M.sum(axis=0) / M.sum()
    exp = np.outer(row_t, col_t)
    num = (W * obs).sum()
    den = (W * exp).sum()
    if den == 0:
        return float("nan")
    return 1 - num / den


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def calibrate_thresholds(rows, exclude_datasets):
    """Compute global + per-language thresholds from labelled speakers.
    Excludes datasets we are evaluating on."""
    by_lang_sev = defaultdict(lambda: defaultdict(list))
    global_by_sev = defaultdict(list)

    for r in rows:
        if r["dataset"] in exclude_datasets:
            continue
        sev = (r.get("severity_label") or "").strip().lower()
        if sev not in SEVERITY_ORDER:
            continue
        score = get_composite(r)
        if score is None:
            continue
        lang = DATASET_LANGUAGE.get(r["dataset"], "en")
        by_lang_sev[lang][sev].append(score)
        global_by_sev[sev].append(score)

    thresholds = {}

    # Global
    g_cent = {s: float(np.mean(global_by_sev[s])) for s in SEVERITY_ORDER
              if global_by_sev[s]}
    g_thr = midpoint_thresholds(g_cent)
    thresholds["global"] = g_thr
    print(f"\n  GLOBAL  n={sum(len(v) for v in global_by_sev.values()):4d}  "
          f"centroids: " + " ".join(f"{s}={g_cent[s]:.2f}" for s in SEVERITY_ORDER if s in g_cent))
    print(f"          thresholds: severe<{g_thr[0]:.2f}  "
          f"moderate<{g_thr[1]:.2f}  mild<{g_thr[2]:.2f}  control>=")

    # Per-language (n>=20)
    for lang in sorted(by_lang_sev):
        cent = {s: float(np.mean(by_lang_sev[lang][s])) for s in SEVERITY_ORDER
                if by_lang_sev[lang][s]}
        n_total = sum(len(by_lang_sev[lang][s]) for s in SEVERITY_ORDER)
        if len(cent) >= 2 and n_total >= 20:
            thresholds[lang] = midpoint_thresholds(cent)
            print(f"  [{lang}]  n={n_total:4d}  centroids: " +
                  " ".join(f"{s}={cent[s]:.2f}" for s in SEVERITY_ORDER if s in cent))
            print(f"           thresholds: severe<{thresholds[lang][0]:.2f}  "
                  f"moderate<{thresholds[lang][1]:.2f}  mild<{thresholds[lang][2]:.2f}")
        else:
            print(f"  [{lang}]  n={n_total:4d}  too few labelled — using global")

    return thresholds


def confusion(y_true, y_pred, classes):
    n = len(classes)
    M = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


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
        # off-by-one accuracy
        off1 = 0
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if abs(i - j) <= 1:
                    off1 += M[i, j]
        off1_acc = off1 / n
        print(f"    n={n}  exact_acc={acc:.3f}  off-by-1_acc={off1_acc:.3f}")


def main():
    print("=" * 80)
    print("Paper 3 pseudo-label threshold method on MSD bundle vs ground truth")
    print("=" * 80)

    # 1. Calibrate thresholds from full master, excluding MSD bundle
    print("\nCalibrating thresholds from track4_master.csv "
          "(MSD bundle excluded from calibration):")
    full_rows = load_csv(MASTER_FULL)
    thresholds = calibrate_thresholds(full_rows, exclude_datasets=MSD_DATASETS)

    # 2. Apply to MSD bundle
    msd_rows = load_csv(MASTER_MSD)
    pred_rows = []
    for r in msd_rows:
        ds = r["dataset"]
        lang = DATASET_LANGUAGE.get(ds, "en")
        thr = thresholds.get(lang, thresholds["global"])
        score = get_composite(r)
        if score is None:
            pred = "unknown"
        else:
            pred = classify(score, thr)
        pred_rows.append({
            "dataset": ds,
            "speaker_id": r["speaker_id"],
            "language": lang,
            "aetiology": r.get("aetiology", ""),
            "ground_truth": r.get("severity_label", ""),
            "pseudo_severity": pred,
            "composite_dprime": f"{score:.3f}" if score is not None else "",
            "thr_severe": f"{thr[0]:.3f}",
            "thr_moderate": f"{thr[1]:.3f}",
            "thr_mild": f"{thr[2]:.3f}",
            "thr_source": lang if lang in thresholds else "global",
        })

    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PREDS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        w.writerows(pred_rows)
    print(f"\n  wrote {len(pred_rows)} pseudo-labels -> {OUT_PREDS}")

    # 3. Compare to ground truth per dataset (exclude unknown)
    cmp_rows = []
    print("\n" + "=" * 80)
    print("Pseudo-label vs ground-truth comparison")
    print("=" * 80)

    classes = SEVERITY_ORDER
    for ds in ["Czech_OneVoice-MSD26", "German_OneVoice-MSD26",
               "PC-GITA_OneVoice-MSD26"]:
        ds_preds = [p for p in pred_rows if p["dataset"] == ds]
        # Filter to rows where both ground-truth and prediction are valid
        valid = [(p["ground_truth"], p["pseudo_severity"]) for p in ds_preds
                 if p["ground_truth"] in classes and p["pseudo_severity"] in classes]
        if not valid:
            print(f"\n  {ds}: no valid pairs")
            continue
        y_true = [t for t, _ in valid]
        y_pred = [p for _, p in valid]
        M = confusion(y_true, y_pred, classes)
        print(f"\n--- {ds} (n={len(valid)}, {len(ds_preds) - len(valid)} excluded) ---")
        print_confusion(M, classes, "confusion")
        kappa = cohen_kappa(y_true, y_pred, classes)
        wkappa = linear_weighted_kappa(y_true, y_pred, classes)
        # Per-class precision / recall
        per_class = []
        for i, c in enumerate(classes):
            tp = int(M[i, i])
            fn = int(M[i, :].sum() - tp)
            fp = int(M[:, i].sum() - tp)
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            per_class.append((c, tp, fn, fp, prec, rec))

        n = M.sum()
        acc = float(np.trace(M)) / n
        off1 = sum(int(M[i, j]) for i in range(len(classes))
                   for j in range(len(classes)) if abs(i - j) <= 1) / n
        print(f"    cohen_kappa={kappa:.3f}  weighted_kappa(linear)={wkappa:.3f}")
        print(f"    {'class':>10}  {'TP':>4} {'FN':>4} {'FP':>4} {'prec':>6} {'rec':>6}")
        for c, tp, fn, fp, p, r in per_class:
            ps = f"{p:.3f}" if not np.isnan(p) else "  --"
            rs = f"{r:.3f}" if not np.isnan(r) else "  --"
            print(f"    {c:>10}  {tp:>4d} {fn:>4d} {fp:>4d} {ps:>6} {rs:>6}")

        cmp_rows.append({
            "dataset": ds,
            "n_compared": len(valid),
            "exact_accuracy": f"{acc:.3f}",
            "off_by_1_accuracy": f"{off1:.3f}",
            "cohen_kappa": f"{kappa:.3f}",
            "weighted_kappa_linear": f"{wkappa:.3f}",
            **{f"M_{a}_to_{b}": int(M[i, j])
               for i, a in enumerate(classes)
               for j, b in enumerate(classes)},
        })

    # Pooled
    all_valid = [(p["ground_truth"], p["pseudo_severity"]) for p in pred_rows
                 if p["ground_truth"] in classes and p["pseudo_severity"] in classes]
    if all_valid:
        y_true = [t for t, _ in all_valid]
        y_pred = [p for _, p in all_valid]
        M = confusion(y_true, y_pred, classes)
        print(f"\n--- POOLED (n={len(all_valid)}) ---")
        print_confusion(M, classes, "confusion")
        kappa = cohen_kappa(y_true, y_pred, classes)
        wkappa = linear_weighted_kappa(y_true, y_pred, classes)
        n = M.sum()
        acc = float(np.trace(M)) / n
        off1 = sum(int(M[i, j]) for i in range(len(classes))
                   for j in range(len(classes)) if abs(i - j) <= 1) / n
        print(f"    cohen_kappa={kappa:.3f}  weighted_kappa(linear)={wkappa:.3f}")

        cmp_rows.append({
            "dataset": "POOLED",
            "n_compared": len(all_valid),
            "exact_accuracy": f"{acc:.3f}",
            "off_by_1_accuracy": f"{off1:.3f}",
            "cohen_kappa": f"{kappa:.3f}",
            "weighted_kappa_linear": f"{wkappa:.3f}",
            **{f"M_{a}_to_{b}": int(M[i, j])
               for i, a in enumerate(classes)
               for j, b in enumerate(classes)},
        })

    if cmp_rows:
        with open(OUT_CMP, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
            w.writeheader()
            w.writerows(cmp_rows)
        print(f"\n  wrote comparison -> {OUT_CMP}")


if __name__ == "__main__":
    main()
