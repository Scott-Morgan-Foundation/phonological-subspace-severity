#!/usr/bin/env python3
"""v17 (CSL R2): same-language healthy-control reference for the centred profile-shape test.

v12_matched_severity_features.py z-scores each speaker's five consonant d' against ALL healthy
controls pooled across languages. Here each speaker is z-scored against the healthy controls of
their OWN language (>= MIN_HC with all five values); speakers whose language has fewer are
dropped and counted. Everything else (strata, groups >= 5 per stratum, centred statistic,
10,000 within-stratum permutations, seed 0, p = (1 + #null >= obs) / (N + 1)) is the v12 code,
imported unchanged. The per-feature Kruskal-Wallis tests do not use the HC reference and are
not repeated. This is a sensitivity check; it cannot separate aetiology from language/corpus.
"""
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REV = HERE.parents[2]
spec = importlib.util.spec_from_file_location("v12", REV / "scripts" / "v12" / "v12_matched_severity_features.py")
v12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v12)

OUT = REV / "results" / "v17" / "v17_matched_severity_samelang.json"
MIN_HC = 5


def main():
    rows = list(csv.DictReader(open(v12.MASTER, encoding="utf-8")))
    hc = defaultdict(list)
    for r in rows:
        if r["aetiology"] == "healthy":
            v = [v12.f(r[c]) for c in v12.CONS5]
            if all(x is not None for x in v):
                hc[r["language"]].append(v)
    ref = {l: (np.mean(v, 0), np.std(v, 0, ddof=1)) for l, v in hc.items() if len(v) >= MIN_HC}

    lab = [r for r in rows if r["aetiology"] in v12.AET and r["severity_label"] in v12.STRATA]
    sap = [r for r in lab if r["dataset"] == "SAP"]
    out = {"hc_per_language": {l: len(v) for l, v in hc.items()}, "min_hc": MIN_HC}
    rng = np.random.RandomState(0)
    for name, subset in (("A_all_corpora", lab), ("B_SAP_english", sap)):
        by_lang = defaultdict(list)
        dropped = defaultdict(int)
        for r in subset:
            if r["language"] in ref:
                by_lang[r["language"]].append(r)
            else:
                dropped[r["language"]] += 1
        # profile_test z-scores with one (mu, sd); run it per language block by pre-standardising.
        std_rows = []
        for l, rs in by_lang.items():
            mu, sd = ref[l]
            for r in rs:
                v = [v12.f(r[c]) for c in v12.CONS5]
                if any(x is None for x in v):
                    continue
                z = (np.array(v) - mu) / sd
                rr = dict(r)
                for c, zz in zip(v12.CONS5, z):
                    rr[c] = str(zz)
                std_rows.append(rr)
        res = v12.profile_test(std_rows, np.zeros(5), np.ones(5), rng)
        res["dropped_no_same_language_hc"] = dict(dropped)
        out[name] = res
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for name in ("A_all_corpora", "B_SAP_english"):
        ps = out[name]
        print(f"=== {name} === cells {ps['cells']} dropped {ps['dropped_no_same_language_hc']}")
        for m in ("centred", "raw", "level"):
            print(f"  {m}: p={ps[m]['p']:.4f} stat={ps[m]['stat']:.2f} null={ps[m]['null_mean']:.2f}")
        print("  centred profiles", ps["profile_features"], json.dumps(ps["centred_profiles"]))


if __name__ == "__main__":
    main()
