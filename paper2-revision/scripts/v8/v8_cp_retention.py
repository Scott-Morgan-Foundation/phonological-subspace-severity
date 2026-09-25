#!/usr/bin/env python3
"""v8: CP d-prime retention (group mean / same-language healthy-control mean, per consonant feature)
by severity label and language (v7 check C-5: §5.1 Swahili vs English comparison). Same definition as
scripts/v6/v6_crossling.py severe_cp_retention, extended to mild and moderate. Output:
results/v8/v8_cp_retention.json (ratio per feature and min-max range per language x severity).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v6"))
from common import MASTER, C5, sha256  # noqa: E402

REV = Path(__file__).resolve().parents[2]
m = pd.read_csv(MASTER)
out = {"definition": __doc__, "inputs": {"corrected/track4_master.csv": sha256(MASTER)}, "retention": {}}
cp = m[m.aetiology == "cerebral_palsy"]
for lang in sorted(cp.language.unique()):
    hc = m[(m.aetiology == "healthy") & (m.language == lang)]
    if hc.empty:
        continue
    out["retention"][lang] = {"n_hc": int(len(hc))}
    for sev in ("mild", "moderate", "severe"):
        g = cp[(cp.language == lang) & (cp.severity_label == sev)]
        if len(g) < 2:
            continue
        r = {c: float(g[c].mean() / hc[c].mean()) for c in C5 if g[c].notna().any() and hc[c].notna().any()}
        out["retention"][lang][sev] = {"n": int(len(g)), "ratio_5c": r, "range_5c": [min(r.values()), max(r.values())]}
p = REV / "results" / "v8" / "v8_cp_retention.json"
p.write_text(json.dumps(out, indent=1), encoding="utf-8")
for l, v in out["retention"].items():
    print(l, {k: (x["n"], [round(y, 3) for y in x["range_5c"]]) for k, x in v.items() if k != "n_hc"})
