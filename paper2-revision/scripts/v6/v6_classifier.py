"""v6: leave-one-dataset-out nearest-centroid aetiology classifier (§4.5).

Six groups, speakers with >= 3 of 5 consonant d-primes valid; class centroids are
nan-means over the training folds. Missing-value handling: "train_mean" (primary) fills
missing features with the training-fold feature mean; "mask" compares a test speaker with
each centroid on that speaker's available features only. Variants: {raw, zscore(train)}
x {euclidean, cosine} x {train_mean, mask}. Primary protocol: raw profiles, cosine distance,
train-mean fill (the fill whose per-class pattern is closest to the submitted result). The submitted-input script for this analysis is not in the package and its
missing-value handling is unknown; `frozen_check` reruns this code on the submitted master
and does not reproduce the submitted 23.7% macro F1, so the corrected-input values below
come from this documented protocol, not from the submitted one.
"""
import numpy as np
import pandas as pd

from common import MASTER, FROZEN_MASTER, C5, write

G = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
     "down_syndrome": "DS", "stroke": "Stroke"}


def load(p):
    m = pd.read_csv(p)
    m = m[m.aetiology.isin(G) & (m[C5].notna().sum(1) >= 3)]
    return m[C5].to_numpy(float), m.aetiology.map(G).to_numpy(), m.dataset.to_numpy()


def run(X, y, ds, scale, metric, fill):
    pred = np.empty(len(y), dtype=object)
    for d in np.unique(ds):
        tr, te = ds != d, ds == d
        Xtr, Xte = X[tr].copy(), X[te].copy()
        mu = np.nanmean(Xtr, 0)
        if fill == "train_mean":
            Xtr = np.where(np.isnan(Xtr), mu, Xtr)
            Xte = np.where(np.isnan(Xte), mu, Xte)
        if scale == "zscore":
            m_, s_ = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
            s_[s_ < 1e-12] = 1
            Xtr, Xte = (Xtr - m_) / s_, (Xte - m_) / s_
        labs = sorted(set(y[tr]))
        C = np.array([np.nanmean(Xtr[y[tr] == g], 0) for g in labs])
        out = []
        for x in Xte:
            mk = ~np.isnan(x)
            a, b = x[mk], C[:, mk]
            if metric == "euclidean":
                dd = ((b - a) ** 2).sum(1)
            else:
                dd = 1 - (b @ a) / np.linalg.norm(b, axis=1) / np.linalg.norm(a)
            out.append(labs[int(np.argmin(dd))])
        pred[te] = out
    f1, rec = {}, []
    for g in sorted(set(y)):
        tp = ((pred == g) & (y == g)).sum(); fp = ((pred == g) & (y != g)).sum(); fn = ((pred != g) & (y == g)).sum()
        r = tp / (tp + fn) if tp + fn else 0.0
        p = tp / (tp + fp) if tp + fp else 0.0
        rec.append(r)
        f1[g] = float(2 * p * r / (p + r)) if p + r else 0.0
    return {"n": int(len(y)), "acc": float((pred == y).mean()), "balanced_acc": float(np.mean(rec)),
            "macro_f1": float(np.mean(list(f1.values()))), "per_class_f1": f1}


out = {"definition": __doc__}
for tag, path in (("frozen_check", FROZEN_MASTER), ("corrected", MASTER)):
    X, y, ds = load(path)
    out[tag] = {f"{s}_{mt}_{fl}": run(X, y, ds, s, mt, fl) for s in ("raw", "zscore")
                for mt in ("euclidean", "cosine") for fl in ("mask", "train_mean")}
    out[tag]["primary"] = "raw_cosine_train_mean"
    best = max((k for k in out[tag] if isinstance(out[tag][k], dict)), key=lambda k: out[tag][k]["macro_f1"])
    out[tag]["best_variant"] = best
write("v6_classifier.json", out, [MASTER, FROZEN_MASTER])
for tag in ("frozen_check", "corrected"):
    for k, v in out[tag].items():
        if isinstance(v, dict):
            print(tag, k, v["n"], round(v["acc"], 3), round(v["balanced_acc"], 3), round(v["macro_f1"], 3),
                  {g: round(x, 3) for g, x in v["per_class_f1"].items()})
    print(tag, "best", out[tag]["best_variant"])
