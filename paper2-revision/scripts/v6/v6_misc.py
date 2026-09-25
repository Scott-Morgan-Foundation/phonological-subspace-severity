"""v6: remaining count checks on the corrected inputs.

Dutch re-scoring counts (results/nl_repair_dprime.csv = the DGX re-scoring output);
control-vs-mild Cohen's d for the 5 consonant d-primes and CTC-Conf (convention coding);
Figure 2 per-group valid n for vowel triangle area; companion-paper numbers are checked
against the Paper 1 arXiv source (papers/Published papers/Paper 1/arxiv/manuscript.tex).
"""
import re

import numpy as np
import pandas as pd

from common import MASTER, REV, C5, V4, SEV, write

NL = REV / "results" / "nl_repair_dprime.csv"
P1 = REV.parent.parent / "Paper 1" / "arxiv" / "manuscript.tex"
AM = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS", "healthy": "HC", "down_syndrome": "DS", "stroke": "Stroke"}
m = pd.read_csv(MASTER)
m["g"] = m.aetiology.map(AM)
m["s"] = m.severity_label.map(SEV)
m.loc[m.is_control.astype(str) == "True", "s"] = 0
m.loc[m.g.isna(), "s"] = np.nan
D9 = C5 + V4
nl = m[m.language == "nl"]
rep = pd.read_csv(NL)
out = {"definition": __doc__,
       "dutch": {"master_rows": int(len(nl)), "rescored_speakers": int(len(rep)),
                 "not_rescored": int(len(nl) - len(rep)),
                 "with_any_dprime_after": int((nl[D9].notna().sum(1) > 0).sum()),
                 "rescored_without_any_dprime": int(len(rep) - (nl[D9].notna().sum(1) > 0).sum())}}


def cd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s)


cm = {}
for c in C5 + ["artp_score"]:
    a = m[(m.s == 0)][c].dropna(); b = m[(m.s == 1)][c].dropna()
    cm[c] = {"d_control_minus_mild": cd(a, b), "mean_gap": float(a.mean() - b.mean()), "n": [int(len(a)), int(len(b))]}
out["control_vs_mild"] = cm
out["fig2_vta_valid_n"] = {g: int(m[m.g == g].vowel_triangle_area.notna().sum()) for g in AM.values()}
tex = P1.read_text(encoding="utf-8")
out["paper1_checks"] = {k: bool(re.search(p, tex)) for k, p in {
    "890 speakers across 10 corpora, 5 languages": r"890 speakers across 10 corpora, 5 languages",
    "pooled Spearman rho = -0.47 to -0.55": r"pooled Spearman \$\\rho = -0\.47\$ to \$-0\.55\$",
    "12-dimensional phonological profile": r"12-dimensional phonological profile",
    "leave-one-corpus-out": r"leave-one-corpus-out",
    "random-effects meta-analysis": r"random-effects meta-analysis",
    "FDR correction": r"FDR correction"}.items()}
write("v6_misc.json", out, [MASTER, NL, P1])
print(out["dutch"]); print({k: round(v["d_control_minus_mild"], 3) for k, v in cm.items()}, cm["artp_score"]["mean_gap"])
print(out["fig2_vta_valid_n"]); print(out["paper1_checks"])
