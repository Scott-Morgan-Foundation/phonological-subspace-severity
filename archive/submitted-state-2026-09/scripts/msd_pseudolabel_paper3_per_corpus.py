#!/usr/bin/env python3
"""Paper 3 11-feature regression method, calibrated WITHIN each corpus (LOSO).

This is the proper application of Paper 3 to the MSD bundle: each corpus uses
its own labelled speakers as calibration, with leave-one-speaker-out
cross-validation.

Pipeline (per corpus):
  features = 5 cons + 4 vowel + boundary_sharpness + cross_position_cosim (11)
  for each labelled speaker:
    fit StandardScaler + multinomial LR on the OTHER labelled speakers in
      this corpus
    composite_score = Σ class_index × P(class) for the held-out speaker
  derive thresholds in score-space from the leave-one-out class means
  classify the held-out speaker by thresholding its composite

Comparison: pseudo-labels vs ground truth (UPDRS-18 for existing PC-GITA,
H&Y for MSD bundle).
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
OUT_PREDS = ROOT / "results" / "msd_pseudolabels_paper3_per_corpus.csv"
OUT_CMP = ROOT / "results" / "msd_pseudolabel_paper3_per_corpus_vs_truth.csv"

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


def fit_lr_score(X_train, y_train, X_query):
    """Fit StandardScaler + multinomial LR, return composite expected-class
    score for query rows."""
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    Xq = scaler.transform(X_query)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                             random_state=42)
    clf.fit(Xs, y_train)
    proba_train = clf.predict_proba(Xs)
    proba_query = clf.predict_proba(Xq)
    score_train = proba_train @ clf.classes_
    score_query = proba_query @ clf.classes_
    return score_train, score_query, clf


def score_to_class(score, thresholds):
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


def print_confusion(M, label):
    print(f"\n  {label}")
    print(f"    {'':>10}  " + "  ".join(f"{c:>8}" for c in SEV_NAMES) + "  total")
    for i, c in enumerate(SEV_NAMES):
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


def per_corpus_loso(rows, label):
    """For one corpus: LOSO calibration of the Paper 3 11-feature LR composite."""
    X, y, meta = extract(rows, FEATS_11)
    if len(X) == 0:
        print(f"\n--- {label}: 0 speakers with all 11 features valid ---")
        return [], None

    print(f"\n--- {label} (n_with_all_11_feats={len(X)}, "
          f"n_dropped={len(rows) - len(X)}) ---")
    print(f"  class distribution (in corpus): "
          + str({SEV_NAMES[c] if c >= 0 else 'unknown': int((y == c).sum())
                 for c in sorted(set(y.tolist()))}))

    # For LOSO we need at least 2 speakers per class (so LR can fit all classes
    # when one is held out). If a class has just 1 speaker, that speaker gets
    # predicted with thresholds derived from the in-sample fit on the rest.
    labelled_mask = y >= 0  # exclude 'unknown'
    Xl = X[labelled_mask]
    yl = y[labelled_mask]
    meta_l = [m for m, lab in zip(meta, labelled_mask) if lab]

    if len(np.unique(yl)) < 2:
        print(f"  too few classes ({np.unique(yl)}) — skipping LOSO")
        return [], None

    # First do an in-sample fit to derive thresholds (class means in score space)
    score_train, _, clf = fit_lr_score(Xl, yl, Xl)
    means = []
    for c in [0, 1, 2, 3]:
        m = yl == c
        means.append(float(score_train[m].mean()) if m.sum() else None)
    thresholds = []
    for a, b in zip(means[:-1], means[1:]):
        if a is None or b is None:
            thresholds.append(None)
        else:
            thresholds.append((a + b) / 2)
    print(f"  in-sample class means: " +
          " ".join(f"{m:.3f}" if m is not None else "N/A" for m in means))
    print(f"  thresholds:            " +
          " ".join(f"{t:.3f}" if t is not None else "N/A" for t in thresholds))

    # LOSO predictions over labelled speakers
    pred_rows = []
    pairs = []
    for i in range(len(Xl)):
        mask = np.ones(len(Xl), dtype=bool)
        mask[i] = False
        try:
            _, score_q, _ = fit_lr_score(Xl[mask], yl[mask], Xl[i:i+1])
            cls = score_to_class(score_q, thresholds)[0]
            pred_label = SEV_NAMES[cls]
        except Exception as e:
            pred_label = "unknown"
            score_q = [float("nan")]
        truth_label = SEV_NAMES[yl[i]]
        pred_rows.append({
            "dataset": meta_l[i]["dataset"],
            "speaker_id": meta_l[i]["speaker_id"],
            "ground_truth": truth_label,
            "composite_loso": f"{score_q[0]:.3f}" if not np.isnan(score_q[0]) else "",
            "pseudo_severity_loso": pred_label,
        })
        pairs.append((truth_label, pred_label))

    yt = [t for t, _ in pairs]
    yp = [p for _, p in pairs if p in SEV_NAMES]
    if len(yp) != len(yt):
        # filter to valid pairs
        valid = [(t, p) for t, p in pairs if p in SEV_NAMES]
        yt = [t for t, _ in valid]
        yp = [p for _, p in valid]
    M = confusion(yt, yp, SEV_NAMES)
    print_confusion(M, "LOSO confusion (rows=truth, cols=pred)")
    return pred_rows, M


def main():
    print("=" * 80)
    print("Paper 3 11-feature regression composite — per-corpus LOSO calibration")
    print("=" * 80)

    full = list(csv.DictReader(open(MASTER_FULL, encoding="utf-8")))
    msd = list(csv.DictReader(open(MASTER_MSD, encoding="utf-8")))

    corpora = [
        ("PC-GITA (UPDRS-18)", [r for r in full if r["dataset"] == "PC-GITA"]),
        ("Czech_OneVoice-MSD26", [r for r in msd if r["dataset"] == "Czech_OneVoice-MSD26"]),
        ("German_OneVoice-MSD26", [r for r in msd if r["dataset"] == "German_OneVoice-MSD26"]),
        ("PC-GITA_OneVoice-MSD26", [r for r in msd if r["dataset"] == "PC-GITA_OneVoice-MSD26"]),
    ]

    all_pred_rows = []
    cmp_rows = []

    for label, rows in corpora:
        preds, M = per_corpus_loso(rows, label)
        all_pred_rows.extend(preds)
        if M is not None and M.sum():
            cmp_rows.append({
                "dataset": label,
                "n_compared": int(M.sum()),
                "exact_accuracy": f"{float(np.trace(M)) / M.sum():.3f}",
                "off_by_1_accuracy": f"{sum(int(M[i,j]) for i in range(4) for j in range(4) if abs(i-j)<=1) / M.sum():.3f}",
                "cohen_kappa": f"{cohen_kappa(M):.3f}",
                "weighted_kappa_linear": f"{lin_wkappa(M):.3f}",
                **{f"M_{a}_to_{b}": int(M[i, j])
                   for i, a in enumerate(SEV_NAMES)
                   for j, b in enumerate(SEV_NAMES)},
            })

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

    print(f"\n{'='*80}\nSUMMARY (Paper 3 11-feature LR, per-corpus LOSO)\n{'='*80}")
    print(f"{'corpus':<32} {'n':>4} {'exact':>6} {'off1':>6} "
          f"{'kappa':>7} {'wkappa':>7}")
    print("-" * 75)
    for r in cmp_rows:
        print(f"{r['dataset']:<32} {r['n_compared']:>4d} "
              f"{r['exact_accuracy']:>6} {r['off_by_1_accuracy']:>6} "
              f"{r['cohen_kappa']:>7} {r['weighted_kappa_linear']:>7}")


if __name__ == "__main__":
    main()
