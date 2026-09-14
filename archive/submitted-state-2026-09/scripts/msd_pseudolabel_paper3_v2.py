#!/usr/bin/env python3
"""Paper 3 11-feature regression-based pseudo-label method, applied to the
MSD bundle and compared to ground-truth severity labels.

Mirrors track3/scripts/recalibrate_with_vta.py:
  features = 5 consonant + 4 vowel d-primes + boundary_sharpness + cross_position_cosim
  calibration = 188 labelled SAP (mild/mod/severe) + 150 LibriSpeech HC (control)
  pipeline   = StandardScaler -> Multinomial LogisticRegression
  composite  = Σ class_index × P(class)  (continuous expected-class score)
  thresholds = midpoints between class-mean composite scores
  apply -> 4-class severity prediction for each MSD speaker
  compare to severity_label (UPDRS-18 for existing PC-GITA, H&Y for MSD bundle)

This re-uses Paper 3's published thresholds calibrated on English/SAP, applied
"as-is" to Spanish, Czech, German PD speakers in the OneVoice protocol.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MASTER_FULL = ROOT / "results" / "track4_master.csv"
MASTER_MSD = ROOT / "results" / "track4_master_msd.csv"
OUT_PREDS = ROOT / "results" / "msd_pseudolabels_paper3_v2.csv"
OUT_CMP = ROOT / "results" / "msd_pseudolabel_paper3_v2_vs_truth.csv"

# 11 features as defined in Paper 3 §3.4
CONS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime",
        "strident_dprime", "manner_dprime"]
VOWEL = ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]
STRUCT = ["boundary_sharpness", "cross_position_cosim"]
FEATS_11 = CONS + VOWEL + STRUCT

SEV_ORD = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
SEV_NAMES = ["control", "mild", "moderate", "severe"]


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def extract(rows, feats):
    """Return X, y, meta for rows where ALL features are valid."""
    X, y, meta = [], [], []
    for r in rows:
        vals = [safe_float(r.get(f)) for f in feats]
        if any(v is None for v in vals):
            continue
        X.append(vals)
        sev = (r.get("severity_label") or "").strip().lower()
        y.append(SEV_ORD.get(sev, -1))
        meta.append(r)
    return np.array(X), np.array(y), meta


def fit_paper3_calibration(rows):
    """Fit the Paper 3 calibration: 188 SAP + 150 LibriSpeech HC, 11 features,
    multinomial LR. Returns scaler, classifier, training metadata."""
    labelled_sap = [r for r in rows
                    if r["dataset"] == "SAP"
                    and (r.get("severity_label") or "").strip().lower()
                        in ("mild", "moderate", "severe")]
    libri_hc = [r for r in rows
                if r["dataset"] == "LibriSpeech_English"
                and (r.get("is_control") or "").strip().lower() == "true"]

    cal = labelled_sap + libri_hc
    X_cal, y_cal_raw, _ = extract(cal, FEATS_11)
    # LibriSpeech extracted as -1 (no severity); remap -> 0 (control)
    y_cal = y_cal_raw.copy()
    y_cal[y_cal == -1] = 0

    scaler = StandardScaler().fit(X_cal)
    Xs = scaler.transform(X_cal)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                             random_state=42)
    clf.fit(Xs, y_cal)

    # Composite = expected class
    proba = clf.predict_proba(Xs)
    score_cal = proba @ clf.classes_

    # Thresholds = midpoints of class means
    means = []
    for c in [0, 1, 2, 3]:
        m = y_cal == c
        means.append(float(score_cal[m].mean()) if m.sum() else None)
    thresholds = []
    for a, b in zip(means[:-1], means[1:]):
        if a is None or b is None:
            thresholds.append(None)
        else:
            thresholds.append((a + b) / 2)

    print(f"\n  Paper 3 calibration:")
    print(f"    n labelled SAP (mild/mod/sev): {len(labelled_sap)}")
    print(f"    n LibriSpeech HC (control):    {len(libri_hc)}")
    print(f"    valid (all 11 features):       {len(X_cal)}")
    print(f"    class counts: {Counter(y_cal.tolist())}")
    print(f"    class-mean composite scores (ctrl, mild, mod, sev): "
          + " ".join(f"{m:.3f}" if m is not None else "N/A" for m in means))
    print(f"    thresholds (ctrl/mild, mild/mod, mod/sev):           "
          + " ".join(f"{t:.3f}" if t is not None else "N/A" for t in thresholds))

    return scaler, clf, thresholds, means


def threshold_to_class(score, thresholds):
    out = np.zeros(len(score), dtype=int)
    for i, s in enumerate(score):
        cls = 0
        for t in thresholds:
            if t is None:
                continue
            if s >= t:
                cls += 1
        out[i] = cls
    return out


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


def print_confusion(M, classes, label):
    print(f"\n  {label}")
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
        print(f"    n={n}  exact={acc:.3f}  off-by-1={off1:.3f}  "
              f"kappa={cohen_kappa(M):.3f}  weighted_kappa={lin_wkappa(M):.3f}")


def predict_for_corpus(rows, scaler, clf, thresholds, label):
    X, _, meta = extract(rows, FEATS_11)
    if len(X) == 0:
        print(f"\n  {label}: 0 speakers with all 11 features valid")
        return [], None, None
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)
    score = proba @ clf.classes_
    cls = threshold_to_class(score, thresholds)

    pred_rows = []
    pairs = []
    for s, c, m in zip(score, cls, meta):
        sev_truth = (m.get("severity_label") or "").strip().lower()
        pred_label = SEV_NAMES[c]
        pred_rows.append({
            "dataset": m["dataset"],
            "speaker_id": m["speaker_id"],
            "ground_truth": sev_truth,
            "composite_score": f"{s:.3f}",
            "pseudo_severity": pred_label,
        })
        if sev_truth in SEV_NAMES:
            pairs.append((sev_truth, pred_label))

    print(f"\n--- {label}  (n_with_all_11_feats={len(X)}, "
          f"n_dropped={len(rows) - len(X)}) ---")
    print(f"  pseudo-label distribution: " +
          ", ".join(f"{SEV_NAMES[k]}={v}" for k, v in
                    sorted(Counter(cls.tolist()).items())))

    M = None
    if pairs:
        yt = [t for t, _ in pairs]
        yp = [p for _, p in pairs]
        M = confusion(yt, yp, SEV_NAMES)
        print_confusion(M, SEV_NAMES, "confusion (rows=truth, cols=pred)")

    return pred_rows, M, len(pairs)


def main():
    print("=" * 80)
    print("Paper 3 11-feature regression composite -> MSD bundle")
    print("=" * 80)

    # 1. Fit on full master (Paper 3 calibration: SAP + LibriSpeech)
    full = list(csv.DictReader(open(MASTER_FULL, encoding="utf-8")))
    scaler, clf, thresholds, means = fit_paper3_calibration(full)

    # 2. Apply to existing PC-GITA (UPDRS-18 ground truth)
    pcgita_existing = [r for r in full if r["dataset"] == "PC-GITA"]
    preds_pcgita, M_pcgita, n_pcgita = predict_for_corpus(
        pcgita_existing, scaler, clf, thresholds,
        "PC-GITA (existing) — UPDRS-18 ground truth")

    # 3. Apply to MSD bundle (H&Y ground truth except for 'unknown' rows)
    msd = list(csv.DictReader(open(MASTER_MSD, encoding="utf-8")))
    all_pred_rows = list(preds_pcgita)
    cmp_rows = []

    for ds in ["Czech_OneVoice-MSD26", "German_OneVoice-MSD26",
               "PC-GITA_OneVoice-MSD26"]:
        ds_rows = [r for r in msd if r["dataset"] == ds]
        preds, M, n = predict_for_corpus(
            ds_rows, scaler, clf, thresholds,
            f"{ds} — H&Y ground truth")
        all_pred_rows.extend(preds)
        if M is not None:
            cmp_rows.append({
                "dataset": ds,
                "n_compared": n,
                "exact_accuracy": f"{float(np.trace(M)) / M.sum():.3f}",
                "off_by_1_accuracy": f"{sum(int(M[i,j]) for i in range(4) for j in range(4) if abs(i-j)<=1) / M.sum():.3f}",
                "cohen_kappa": f"{cohen_kappa(M):.3f}",
                "weighted_kappa_linear": f"{lin_wkappa(M):.3f}",
                **{f"M_{a}_to_{b}": int(M[i, j])
                   for i, a in enumerate(SEV_NAMES)
                   for j, b in enumerate(SEV_NAMES)},
            })

    # PC-GITA-existing comparison row
    if M_pcgita is not None:
        cmp_rows.insert(0, {
            "dataset": "PC-GITA (UPDRS-18)",
            "n_compared": n_pcgita,
            "exact_accuracy": f"{float(np.trace(M_pcgita)) / M_pcgita.sum():.3f}",
            "off_by_1_accuracy": f"{sum(int(M_pcgita[i,j]) for i in range(4) for j in range(4) if abs(i-j)<=1) / M_pcgita.sum():.3f}",
            "cohen_kappa": f"{cohen_kappa(M_pcgita):.3f}",
            "weighted_kappa_linear": f"{lin_wkappa(M_pcgita):.3f}",
            **{f"M_{a}_to_{b}": int(M_pcgita[i, j])
               for i, a in enumerate(SEV_NAMES)
               for j, b in enumerate(SEV_NAMES)},
        })

    # Save predictions + comparison CSV
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PREDS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_pred_rows[0].keys()))
        w.writeheader()
        w.writerows(all_pred_rows)
    print(f"\n  wrote {len(all_pred_rows)} predictions -> {OUT_PREDS}")

    if cmp_rows:
        with open(OUT_CMP, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
            w.writeheader()
            w.writerows(cmp_rows)
        print(f"  wrote comparison -> {OUT_CMP}")

    print(f"\n{'='*80}\nSUMMARY (Paper 3 11-feature regression composite)\n{'='*80}")
    print(f"{'corpus':<32} {'n':>4} {'exact':>6} {'off1':>6} "
          f"{'kappa':>7} {'wkappa':>7}")
    print("-" * 75)
    for r in cmp_rows:
        print(f"{r['dataset']:<32} {r['n_compared']:>4d} "
              f"{r['exact_accuracy']:>6} {r['off_by_1_accuracy']:>6} "
              f"{r['cohen_kappa']:>7} {r['weighted_kappa_linear']:>7}")


if __name__ == "__main__":
    main()
