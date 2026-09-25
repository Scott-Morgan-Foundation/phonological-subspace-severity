#!/usr/bin/env python3
"""v8: interval and p-value for the Test 2 sensitivity analyses whose point estimates only were saved
(v7 check NV-2): the 5-speaker cell threshold (bootstrap CI + block-permutation p) and every
leave-one-dataset-out fold (bootstrap CI). Uses the functions of scripts/corrected/
test2_aetiology_vs_language.py unchanged (same loader, profiles, Delta, resampling); the only
addition is a permutation routine that passes min_cell through to build_profiles, which the
original permute() does not expose. Seeds fixed; draws per sensitivity as in the original
(N_BOOT_SENS = 2,000; permutation 2,000 for the min-cell-5 variant).
Output: results/v8/v8_test2_sensitivity.json
"""
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[2]
SRC = REV / "scripts" / "corrected" / "test2_aetiology_vs_language.py"
spec = importlib.util.spec_from_file_location("t2c", SRC)
t2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t2)  # type: ignore
OUT = REV / "results" / "v8" / "v8_test2_sensitivity.json"
N_SENS = 2_000
N_PERM = 2_000


def permute_min_cell(rng, dys, hc, normalise, n_draws, min_cell):
    recs = []
    for (lang, aet), (mat, ds) in dys.items():
        for i in range(len(mat)):
            recs.append((lang, ds[i], aet, mat[i]))
    by_block = defaultdict(list)
    for j, r in enumerate(recs):
        by_block[(r[0], r[1])].append(j)
    labels0 = np.array([r[2] for r in recs])
    draws = np.empty(n_draws)
    for b in range(n_draws):
        labels = labels0.copy()
        for js in by_block.values():
            js = np.array(js)
            labels[js] = labels0[js[rng.permutation(len(js))]]
        d2, d2ds = defaultdict(list), defaultdict(list)
        for j, r in enumerate(recs):
            d2[(r[0], labels[j])].append(r[3])
            d2ds[(r[0], labels[j])].append(r[1])
        d2 = {k: (np.array(v), d2ds[k]) for k, v in d2.items()}
        draws[b] = t2.delta(t2.build_profiles(d2, hc, normalise, min_cell=min_cell))[0]
    return draws


def main():
    dys5, hc5 = t2.load(t2.FEATS5)
    out = {"definition": __doc__, "source_script": str(SRC.relative_to(REV)).replace("\\", "/"),
           "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "master": str(t2.MASTER.relative_to(REV)).replace("\\", "/"),
           "master_sha256": hashlib.sha256(t2.MASTER.read_bytes()).hexdigest(),
           "seed": t2.SEED, "n_boot": N_SENS, "n_perm": N_PERM}
    pr5 = t2.build_profiles(dys5, hc5, True, min_cell=5)
    d5 = t2.delta(pr5)[0]
    lo, hi, _ = t2.boot_ci(np.random.default_rng(t2.SEED + 55), dys5, hc5, True, N_SENS, min_cell=5)
    pd = permute_min_cell(np.random.default_rng(t2.SEED + 56), dys5, hc5, True, N_PERM, 5)
    ok = pd[~np.isnan(pd)]
    out["min_cell_5"] = {"delta": d5, "ci95": [lo, hi], "perm_lang_dataset_p": float(np.mean(ok >= d5)),
                         "perm_null_mean": float(np.mean(ok)), "n_perm_valid": int(len(ok)),
                         "cells_used": sorted(f"{l}|{a}" for (l, a) in pr5)}
    lodo = {}
    datasets = sorted({d for (mat, ds) in dys5.values() for d in ds})
    for i, D in enumerate(datasets):
        dd = {}
        for k, (mat, ds) in dys5.items():
            keep = [j for j, x in enumerate(ds) if x != D]
            if keep:
                dd[k] = (mat[keep], [ds[j] for j in keep])
        d = t2.delta(t2.build_profiles(dd, hc5, True))[0]
        lo, hi, _ = t2.boot_ci(np.random.default_rng(t2.SEED + 300 + i), dd, hc5, True, N_SENS)
        lodo[D] = {"delta": d, "ci95": [lo, hi]}
    out["LODO"] = lodo
    out["LODO_negative"] = sorted(k for k, v in lodo.items() if v["delta"] < 0)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["min_cell_5"], indent=1))
    print("LODO negative:", out["LODO_negative"])


if __name__ == "__main__":
    main()
