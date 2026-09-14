#!/usr/bin/env python3
"""Re-fit the SAP pseudo-label calibration with vowel_triangle_area included.

Compares the existing 11-feature composite (5 consonant d-primes + 4 vowel
d-primes + boundary_sharpness + cross_position_cosim) against a 12-feature
version that also includes vowel_triangle_area, and reports how many SAP
unlabelled speakers change pseudo-label class.

If the two label sets agree to within a few percent, VTA can be safely added
to the paper as a documentation fix without re-training Track 3 models. If a
substantial fraction of speakers flip class, we have to redo the SAP CDSD
training run.

Usage: python track3/scripts/recalibrate_with_vta.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[2]
MASTER = BASE / "track4" / "results" / "track4_master.csv"
EXISTING_PSEUDOLABELS = BASE / "track4" / "results" / "sap_pseudolabels_threshold.csv"

CONS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime",
        "strident_dprime", "manner_dprime"]
VOWEL = ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]
STRUCT_BASE = ["boundary_sharpness", "cross_position_cosim"]
VTA = ["vowel_triangle_area"]

FEATS_11 = CONS + VOWEL + STRUCT_BASE
FEATS_12 = FEATS_11 + VTA

SEV_ORD = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
SEV_NAMES = ["control", "mild", "moderate", "severe"]


def safe_float(v):
    if v in ("", "nan", None):
        return None
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except Exception:
        return None


def extract(rows, feats):
    X, y, meta = [], [], []
    for r in rows:
        vals = [safe_float(r.get(f)) for f in feats]
        if any(v is None for v in vals):
            continue
        X.append(vals)
        sev = r.get("severity_label", "unknown")
        y.append(SEV_ORD.get(sev, -1))
        meta.append(r)
    return np.array(X), np.array(y), meta


def composite_score(X, y, X_unlab):
    """Fit logistic regression on labelled data, return continuous severity
    score (expected ordinal class) for both labelled and unlabelled speakers."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Xu = scaler.transform(X_unlab)

    clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                             random_state=42)
    clf.fit(Xs, y)

    # Expected class = sum of class_index * prob -- continuous severity score
    classes = clf.classes_
    proba_lab = clf.predict_proba(Xs)
    proba_unl = clf.predict_proba(Xu)
    score_lab = proba_lab @ classes
    score_unl = proba_unl @ classes
    return score_lab, score_unl, clf


def derive_thresholds(score_lab, y_lab):
    """Stipancic-style ordinal thresholds: midpoints between class-mean
    composite scores (ordered ctrl < mild < moderate < severe)."""
    means = []
    for c in [0, 1, 2, 3]:
        m = y_lab == c
        if m.sum():
            means.append(float(score_lab[m].mean()))
        else:
            means.append(None)
    # midpoints between adjacent class means (only adjacent pairs that exist)
    thresholds = []
    for a, b in zip(means[:-1], means[1:]):
        if a is None or b is None:
            thresholds.append(None)
        else:
            thresholds.append((a + b) / 2)
    return means, thresholds


def threshold_to_class(score, thresholds):
    """Map composite score to integer severity class via thresholds."""
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


def main():
    # --- Load master CSV
    rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))

    # --- Labelled SAP (mild/moderate/severe only) + LibriSpeech English HC
    labelled_sap = [r for r in rows
                    if r["dataset"] == "SAP"
                    and r.get("severity_label") in ("mild", "moderate", "severe")]
    libri_hc = [r for r in rows
                if r["dataset"] == "LibriSpeech_English"
                and r.get("is_control", "False") == "True"]
    print(f"Labelled SAP (mild/mod/severe): {len(labelled_sap)}")
    print(f"LibriSpeech HC (control):       {len(libri_hc)}")

    # --- Unlabelled SAP (severity_label == "unknown")
    unlab_sap = [r for r in rows
                 if r["dataset"] == "SAP"
                 and r.get("severity_label", "unknown") == "unknown"]
    print(f"Unlabelled SAP target pool:     {len(unlab_sap)}")

    # Combine labelled + LibriSpeech HC = calibration set
    cal_rows = labelled_sap + libri_hc

    for n_feats, feats, label in [(11, FEATS_11, "11 features (current)"),
                                  (12, FEATS_12, "12 features (+VTA)")]:
        print()
        print("=" * 70)
        print(f"  {label}: {feats}")
        print("=" * 70)

        X_cal, y_cal_raw, _ = extract(cal_rows, feats)
        # Map LibriSpeech rows (currently y=-1) to "control" class 0
        y_cal = y_cal_raw.copy()
        y_cal[y_cal == -1] = 0  # control
        # Bump dys classes by 1 so they line up with ordinal
        # mild=1, moderate=2, severe=3
        # Actually y_cal already had mild=0/mod=1/sev=2 from SEV_ORD if not present
        # SEV_ORD has control=0, mild=1, moderate=2, severe=3 — that's what we want
        # but extract assigned mild=1 etc, and -1 for unknown/control

        # Pull unlabelled feature matrix
        X_unl, _, meta_unl = extract(unlab_sap, feats)

        print(f"  Calibration N: {len(X_cal)}, Unlabelled N: {len(X_unl)}")
        print(f"  Calibration class counts: {Counter(y_cal.tolist())}")

        score_cal, score_unl, clf = composite_score(X_cal, y_cal, X_unl)

        means, thresholds = derive_thresholds(score_cal, y_cal)
        print(f"  Class-mean composite scores (ctrl, mild, mod, sev): {[f'{m:.3f}' if m is not None else 'N/A' for m in means]}")
        print(f"  Thresholds (ctrl/mild, mild/mod, mod/sev):          {[f'{t:.3f}' if t is not None else 'N/A' for t in thresholds]}")

        cls_unl = threshold_to_class(score_unl, thresholds)
        dist = Counter(cls_unl.tolist())
        print(f"  Unlabelled pseudo-label distribution: " +
              ", ".join(f"{SEV_NAMES[k]}={v}" for k, v in sorted(dist.items())))

        if n_feats == 11:
            saved_unl_meta = meta_unl
            saved_cls_11 = cls_unl
        else:
            # Compare to 11-feature labels
            # NOTE: the unlabelled speaker set may differ slightly between the two
            # runs because the 12-feature run drops anyone missing VTA. Build a
            # mapping by speaker_id and intersect.
            id_to_11 = {m["speaker_id"]: c for m, c in zip(saved_unl_meta, saved_cls_11)}
            id_to_12 = {m["speaker_id"]: c for m, c in zip(meta_unl, cls_unl)}
            shared = set(id_to_11) & set(id_to_12)
            only_11 = set(id_to_11) - set(id_to_12)
            only_12 = set(id_to_12) - set(id_to_11)
            print()
            print(f"  --- 11 vs 12 comparison ---")
            print(f"  Speakers with valid features in both runs: {len(shared)}")
            print(f"  Lost from 12-feature run (missing VTA):    {len(only_11)}")
            print(f"  Added in 12-feature run (gained VTA):      {len(only_12)}")
            agree = sum(1 for s in shared if id_to_11[s] == id_to_12[s])
            disagree = len(shared) - agree
            print(f"  Agree:    {agree:>5} ({100*agree/len(shared):.1f} %)")
            print(f"  Disagree: {disagree:>5} ({100*disagree/len(shared):.1f} %)")
            # break disagreements down by direction
            shifts = Counter()
            for s in shared:
                a, b = id_to_11[s], id_to_12[s]
                if a != b:
                    shifts[(SEV_NAMES[a], SEV_NAMES[b])] += 1
            if shifts:
                print(f"  Class shifts (11 -> 12):")
                for (a, b), n in sorted(shifts.items(), key=lambda x: -x[1]):
                    print(f"    {a:>9} -> {b:<9}  {n}")


if __name__ == "__main__":
    main()
