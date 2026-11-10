"""Hungarian provenance: per-corpus agreement between the 2026-09-13 re-extraction (released hu config) and the retained master values."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REV = Path(__file__).resolve().parents[2]
m = pd.read_csv(REV / "corrected/track4_master.csv", low_memory=False)
m = m[m.language == "hu"].copy()
m["speaker_id"] = m.speaker_id.astype(str)
r = pd.read_csv(REV / "results/test9_hu_dprime.csv", dtype={"speaker_id": str})
mp = {"manner": "manner_dprime", "nasal": "nasal_dprime", "sonorant": "sonorant_dprime", "strident": "strident_dprime",
      "voicing": "voicing_dprime", "vowel_back": "back_dprime", "vowel_high": "high_dprime", "vowel_low": "low_dprime",
      "vowel_round": "round_dprime"}
j = r.merge(m, on=["dataset", "speaker_id"])
out = {}
for ds, g in list(j.groupby("dataset")) + [("ALL", j)]:
    x, y = [], []
    for a, b in mp.items():
        t = g[[a, b]].dropna()
        x += list(t[a]); y += list(t[b])
    x, y = np.array(x), np.array(y)
    out[ds] = {"speakers": int(len(g)), "values": int(len(x)), "spearman": round(float(spearmanr(x, y)[0]), 4),
               "median_abs_diff": round(float(np.median(abs(x - y))), 4), "p90_abs_diff": round(float(np.percentile(abs(x - y), 90)), 4)}
master_identical = pd.read_csv(REV / "frozen_package/results/track4_master.csv", low_memory=False)
f = list(mp.values())
fb = master_identical[master_identical.language == "hu"].set_index("speaker_id")[f]
fa = m.set_index("speaker_id")[f]
fb.index = fb.index.astype(str)
out["retained_equals_submitted"] = bool(np.allclose(fa.sort_index().values, fb.sort_index().values, equal_nan=True))
hd = m[m.dataset == "Hungarian_Dysarthria"]
six = {"parkinsons", "stroke", "cerebral_palsy", "als", "down_syndrome"}
out["hungarian_dysarthria_in_six_groups"] = hd[hd.aetiology.isin(six)].aetiology.value_counts().to_dict()
(REV / "results/v18").mkdir(exist_ok=True)
(REV / "results/v18/v18_hu_provenance.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
