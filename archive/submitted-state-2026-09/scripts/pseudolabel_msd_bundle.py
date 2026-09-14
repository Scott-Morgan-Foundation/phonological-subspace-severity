#!/usr/bin/env python3
"""
Task #40 — Per-corpus Stipancic-calibrated pseudo-labels for the MSD bundle.

Approach: HC-anchored calibration per corpus.
  1. For each corpus (Czech, German, PC-GITA-MSD), z-score each bedrock feature
     against the corpus's own HC speakers.
  2. Compute composite severity score = mean of the 5 z-scores.
     (More negative = more deviant from HC = more severe.)
  3. Train a 4-class LR (control/mild/moderate/severe) on the labeled speakers
     across all 3 corpora combined (HC-anchored z-scores).
  4. Apply to unknowns; report agreement with H&Y/UPDRS labels on labeled speakers.
  5. Validate: train on Czech+German, predict PC-GITA-MSD PD (held-out UPDRS-18 labels).

Output:
  - track4/results/msd_pseudolabels.csv   — per-speaker pseudo-labels
  - Printed DGX manifest update commands  — paste onto DGX to update manifest_v2_dgx.jsonl

Usage:
    python papers/Published papers/Paper 2/scripts/pseudolabel_msd_bundle.py
"""
import sys, io, codecs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout

import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

BASE = Path(__file__).resolve().parent.parent.parent.parent  # dysarthria/
MSD_CSV   = BASE / "track4" / "results" / "track4_master_msd.csv"
OUT_CSV   = BASE / "track4" / "results" / "msd_pseudolabels.csv"

BEDROCK = [
    "manner_dprime", "strident_dprime", "voicing_dprime",
    "f0_std_semitones", "f0_range_semitones",
]

SEV_ORDER  = ["control", "mild", "moderate", "severe"]
SEV_NUM    = {s: i for i, s in enumerate(SEV_ORDER)}


def load_msd():
    df = pd.read_csv(MSD_CSV)
    # Fill unknown severity
    df["severity_label"] = df["severity_label"].fillna("unknown")
    return df


def zscored_per_corpus(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    For each corpus, z-score bedrock features against that corpus's HC speakers.
    Returns df with new columns '<feat>_z' and 'composite_z'.
    """
    df = df.copy()
    z_cols = [f + "_z" for f in features]
    df[z_cols] = np.nan

    for corpus, grp in df.groupby("dataset"):
        hc_mask = (df["dataset"] == corpus) & (df["severity_label"] == "control")
        hc = df.loc[hc_mask, features]
        if hc.empty:
            print(f"  WARNING: no HC in {corpus}, skipping z-score")
            continue
        mu = hc.mean()
        sd = hc.std().clip(lower=1e-6)
        for feat, zcol in zip(features, z_cols):
            df.loc[df["dataset"] == corpus, zcol] = (
                df.loc[df["dataset"] == corpus, feat] - mu[feat]
            ) / sd[feat]

    # Composite: mean of available z-scores (note: lower = more deviant)
    df["composite_z"] = df[z_cols].mean(axis=1, skipna=False)
    return df


def build_xy(df, label_col="severity_label", classes=None):
    """Return X (z-cols), y (int), mask of valid rows."""
    if classes is None:
        classes = SEV_ORDER
    z_cols = [f + "_z" for f in BEDROCK]
    valid = (
        df[label_col].isin(classes) &
        df[z_cols].notna().all(axis=1)
    )
    sub = df[valid].copy()
    X = sub[z_cols].values
    y = sub[label_col].map(SEV_NUM).values
    return X, y, sub


def main():
    print("=" * 70)
    print("Task #40 — Per-corpus HC-anchored pseudo-labels for MSD bundle")
    print("=" * 70)

    df = load_msd()
    print(f"\nLoaded {len(df)} speakers from {df['dataset'].nunique()} corpora")
    for ds, g in df.groupby("dataset"):
        print(f"  {ds}: {len(g)} total  |  "
              f"HC={len(g[g.severity_label=='control'])}  "
              f"PD={len(g[g.aetiology=='parkinsons'])}  "
              f"unknown={(g.severity_label=='unknown').sum()}")

    # Step 1: z-score per corpus
    df = zscored_per_corpus(df, BEDROCK)
    z_cols = [f + "_z" for f in BEDROCK]

    # Report composite stats per class per corpus
    print("\n--- Composite z-score by severity class (per corpus) ---")
    for ds, g in df.groupby("dataset"):
        print(f"\n  {ds}:")
        for sev in SEV_ORDER:
            sub = g[g.severity_label == sev]["composite_z"].dropna()
            if len(sub):
                print(f"    {sev:10s}: n={len(sub):3d}  mean={sub.mean():+.3f}  "
                      f"std={sub.std():.3f}")

    # Step 2: Cross-corpus validation
    # Train on Czech + German (H&Y labels), validate on PC-GITA-MSD (UPDRS-18 labels)
    print("\n--- Cross-corpus validation: train Czech+German → predict PC-GITA-MSD ---")
    train_mask = df["dataset"].isin(["Czech_OneVoice-MSD26", "German_OneVoice-MSD26"])
    val_mask   = df["dataset"] == "PC-GITA_OneVoice-MSD26"

    X_tr, y_tr, sub_tr = build_xy(df[train_mask])
    X_val, y_val, sub_val = build_xy(df[val_mask])

    if len(X_tr) >= 10 and len(X_val) >= 5:
        clf_xval = LogisticRegression(max_iter=500, class_weight="balanced",
                                      random_state=42)
        clf_xval.fit(X_tr, y_tr)
        y_pred_val = clf_xval.predict(X_val)
        present = sorted(set(y_val) | set(y_pred_val))
        names = [SEV_ORDER[i] for i in present]
        print(classification_report(y_val, y_pred_val,
                                    labels=present, target_names=names,
                                    zero_division=0))
        kappa = cohen_kappa_score(y_val, y_pred_val, weights="linear")
        print(f"  Weighted kappa vs UPDRS-18 labels: {kappa:.3f}")
    else:
        print("  Insufficient data for cross-corpus validation")

    # Step 3: Train on ALL labeled MSD speakers (4-class)
    print("\n--- Final model: all labeled MSD speakers (5-fold CV) ---")
    X_all, y_all, sub_all = build_xy(df)
    print(f"  Training on {len(X_all)} speakers with all 5 bedrock features")
    print(f"  Class dist: {dict(pd.Series(y_all).map({v:k for k,v in SEV_NUM.items()}).value_counts())}")

    clf_final = LogisticRegression(max_iter=500, class_weight="balanced",
                                   random_state=42)
    if len(np.unique(y_all)) >= 2 and len(X_all) >= 10:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_cv = cross_val_predict(clf_final, X_all, y_all, cv=cv)
        present = sorted(set(y_all))
        names = [SEV_ORDER[i] for i in present]
        print(classification_report(y_all, y_cv, labels=present,
                                    target_names=names, zero_division=0))
        print(f"  CV weighted kappa: {cohen_kappa_score(y_all, y_cv, weights='linear'):.3f}")
    clf_final.fit(X_all, y_all)

    # Step 4: Predict ALL PD speakers (labeled + unknown) for pseudo_severity
    print("\n--- Applying pseudo-labels to all PD speakers ---")
    pd_mask = df["aetiology"] == "parkinsons"
    has_feat = df[z_cols].notna().all(axis=1)
    apply_mask = pd_mask & has_feat

    df_apply = df[apply_mask].copy()
    X_apply = df_apply[z_cols].values
    y_pred   = clf_final.predict(X_apply)
    y_proba  = clf_final.predict_proba(X_apply)

    df_apply["pseudo_severity"]    = [SEV_ORDER[y] for y in y_pred]
    df_apply["pseudo_confidence"]  = y_proba.max(axis=1).round(3)
    df_apply["pseudo_source"]      = "track4_per_corpus_calibrated"

    # Agreement stats: pseudo vs H&Y/UPDRS labeled
    labeled_mask = df_apply["severity_label"].isin(["mild","moderate","severe"])
    if labeled_mask.sum() >= 5:
        print(f"\n  Agreement on {labeled_mask.sum()} labeled PD speakers:")
        agree = (df_apply.loc[labeled_mask, "pseudo_severity"] ==
                 df_apply.loc[labeled_mask, "severity_label"])
        print(f"  Exact match: {agree.sum()}/{labeled_mask.sum()} "
              f"({100*agree.mean():.1f}%)")
        y_true_lab = df_apply.loc[labeled_mask, "severity_label"].map(SEV_NUM)
        y_pred_lab = df_apply.loc[labeled_mask, "pseudo_severity"].map(SEV_NUM)
        kappa = cohen_kappa_score(y_true_lab, y_pred_lab, weights="linear")
        print(f"  Weighted kappa vs H&Y/UPDRS labels: {kappa:.3f}")

        print("\n  Per-corpus agreement:")
        for ds, g in df_apply[labeled_mask].groupby("dataset"):
            match = (g["pseudo_severity"] == g["severity_label"]).mean()
            print(f"    {ds}: {match:.1%} exact match (n={len(g)})")

    # Pseudo-label distribution
    print("\n  Pseudo-label distribution (PD speakers with features):")
    for ds, g in df_apply.groupby("dataset"):
        dist = g["pseudo_severity"].value_counts().to_dict()
        unk_orig = (df.loc[apply_mask & (df["dataset"]==ds), "severity_label"]
                    == "unknown").sum()
        print(f"    {ds}: {dist}  (was unknown: {unk_orig})")

    # Step 5: Output CSV
    out_cols = ["dataset","speaker_id","language","aetiology",
                "severity_label","pseudo_severity","pseudo_confidence",
                "pseudo_source","composite_z"] + z_cols
    df_apply[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\nPseudo-labels saved to:\n  {OUT_CSV}")
    print(f"  ({len(df_apply)} PD speakers with features)")

    # Step 6: Per-speaker table for unknown-only cases
    unknowns = df_apply[df_apply["severity_label"] == "unknown"]
    if len(unknowns):
        print(f"\n--- Unknown speakers now pseudo-labelled ({len(unknowns)}) ---")
        for _, row in unknowns.iterrows():
            print(f"  {row['dataset']:30s}  {row['speaker_id']:20s}  "
                  f"→ {row['pseudo_severity']:10s}  conf={row['pseudo_confidence']:.2f}")

    # Step 7: Print DGX update instructions
    print("\n--- DGX manifest update ---")
    print("Run on DGX to apply pseudo_severity to manifest_v2_dgx.jsonl:")
    print("""
  python3 - <<'PYEOF'
import json, csv
from pathlib import Path

base = Path(os.environ.get('DYSARTHRIA_BASE', '~/dysarthria')).expanduser() / 'data' / 'processed'
pseudo = {}
with open('/tmp/msd_pseudolabels.csv') as f:
    for row in csv.DictReader(f):
        pseudo[(row['dataset'], row['speaker_id'])] = {
            'pseudo_severity': row['pseudo_severity'],
            'pseudo_confidence': float(row['pseudo_confidence']),
            'pseudo_source': row['pseudo_source'],
        }

for ds in ('Czech_OneVoice-MSD26','German_OneVoice-MSD26','PC-GITA_OneVoice-MSD26'):
    src = base / ds / 'manifest_v2_dgx.jsonl'
    if not src.exists():
        print(f'SKIP: {src}')
        continue
    records = [json.loads(l) for l in open(src)]
    updated = 0
    for r in records:
        key = (ds, r.get('speaker_id',''))
        if key in pseudo:
            r.update(pseudo[key])
            updated += 1
    bak = src.with_suffix('.jsonl.bak_pseudolabel')
    import shutil; shutil.copy(src, bak)
    with open(src, 'w') as f:
        for r in records: f.write(json.dumps(r) + '\\n')
    print(f'{ds}: updated {updated}/{len(records)} records')
PYEOF
""")


if __name__ == "__main__":
    main()
