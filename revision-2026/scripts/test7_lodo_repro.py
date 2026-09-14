#!/usr/bin/env python
"""Test 7 — reproduce the LODO nearest-centroid classifier (ANALYSIS_PLAN §8).

Submitted §4.1: leave-one-dataset-out nearest-centroid on 5-dim consonant d'
profiles, 6 groups, 2,928 speakers -> 40.3% acc / 28.6% balanced acc / 22.6%
macro F1; per-class F1 HC .639, CP .399, ALS .270, PD/DS/Stroke < .03.

The original script is NOT in the freeze (gap; logged), so scaling and metric
are unknown — we grid over {raw, zscore(train)} x {euclidean, cosine} and report
which cell reproduces the submitted numbers.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MASTER = BASE / "frozen_package" / "results" / "track4_master.csv"
OUT = BASE / "results" / "test7_lodo_repro.json"

FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime",
         "manner_dprime"]
GROUPS = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
          "down_syndrome": "DS", "stroke": "Stroke"}


def load():
    X, y, ds = [], [], []
    with open(MASTER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = GROUPS.get(r["aetiology"])
            if not g:
                continue
            try:
                v = [float(r[c]) for c in FEATS]
            except ValueError:
                continue
            if any(x != x for x in v):
                continue
            X.append(v)
            y.append(g)
            ds.append(r["dataset"])
    return np.array(X), np.array(y), np.array(ds)


def run(X, y, ds, scale, metric):
    preds = np.empty(len(y), dtype=object)
    for d in np.unique(ds):
        tr, te = ds != d, ds == d
        Xtr, Xte = X[tr], X[te]
        if scale == "zscore":
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd < 1e-12] = 1
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        cents, labels = [], []
        for g in sorted(set(y[tr])):
            cents.append(Xtr[y[tr] == g].mean(0))
            labels.append(g)
        C = np.array(cents)
        if metric == "euclidean":
            dist = ((Xte[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        else:
            a = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
            b = C / np.linalg.norm(C, axis=1, keepdims=True)
            dist = 1 - a @ b.T
        preds[te] = [labels[i] for i in dist.argmin(1)]
    classes = sorted(set(y))
    acc = float((preds == y).mean())
    recalls, f1s = [], {}
    for g in classes:
        tp = ((preds == g) & (y == g)).sum()
        fp = ((preds == g) & (y != g)).sum()
        fn = ((preds != g) & (y == g)).sum()
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(rec)
        f1s[g] = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
    return {"acc": round(acc, 3), "balanced_acc": round(float(np.mean(recalls)), 3),
            "macro_f1": round(float(np.mean(list(f1s.values()))), 3), "per_class_f1": f1s}


def main():
    X, y, ds = load()
    print(f"speakers with complete 5-feat profiles in 6 groups: {len(y)} "
          f"(submitted: 2,928)")
    out = {"n_speakers": int(len(y)),
           "submitted": {"n": 2928, "acc": 0.403, "balanced_acc": 0.286,
                         "macro_f1": 0.226,
                         "per_class_f1": {"HC": 0.639, "CP": 0.399, "ALS": 0.270}}}
    for scale in ("raw", "zscore"):
        for metric in ("euclidean", "cosine"):
            r = run(X, y, ds, scale, metric)
            out[f"{scale}_{metric}"] = r
            print(f"{scale}/{metric}: {r}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
