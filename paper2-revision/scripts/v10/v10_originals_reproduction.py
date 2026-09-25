#!/usr/bin/env python3
"""v10: does each 'original' script, run on the SUBMITTED inputs, reproduce the numbers printed in the SUBMITTED manuscript?

For every original the revision re-runs on the corrected inputs, this compares the original's own output on the
submitted inputs (the frozen logs, or the v8 submitted-input re-run where no frozen log exists) with the value
printed in the submitted manuscript (frozen_package/paper2_csl_submission.zip:paper2-csl.tex; line numbers refer
to that file). A value is 'reproduced' when the log value lies within half a unit of the manuscript's last printed digit.
Output: results/v10/v10_originals_reproduction.json
"""
import hashlib
import json
import re
import zipfile
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
SUB = zipfile.ZipFile(REV / "frozen_package" / "paper2_csl_submission.zip").read("paper2-csl.tex").decode("utf-8")
LOGS = {"robustness_analyses.py": REV / "frozen_package/logs/robustness_analyses.log",
        "fixed_token_dprime.py": REV / "frozen_package/logs/fixed_token_dprime.log",
        "robustness_pass5.py": REV / "frozen_package/logs/robustness_pass5.log",
        "reviewer_experiments.py": REV / "results/v8/originals/submitted/reviewer_experiments.log",
        "sap_excluded_sensitivity.py": REV / "results/v8/originals/submitted/sap_excluded_sensitivity.log",
        "cross_model_comparison.py": REV / "results/v8/originals/submitted/cross_model_comparison.log"}
TXT = {k: p.read_text(encoding="utf-8") for k, p in LOGS.items()}


def rnd(x, printed):
    d = len(printed.split(".")[1]) if "." in printed else 0
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), ROUND_HALF_EVEN)


def grab(script, pattern, group=1):
    m = re.search(pattern, TXT[script], re.M)
    return m.group(group) if m else None


_NORM, _MAP = [], []
for _i, _c in enumerate(SUB):
    if _c.isspace():
        if _NORM and _NORM[-1] == " ":
            continue
        _NORM.append(" ")
    else:
        _NORM.append(_c)
    _MAP.append(_i)
_NORM = "".join(_NORM)


def sub_line(snippet):
    """Line of the snippet in the submitted tex (whitespace-insensitive: table rows span lines)."""
    i = _NORM.find(re.sub(r"\s+", " ", snippet))
    return None if i < 0 else SUB.count("\n", 0, _MAP[i]) + 1


rows = []


def row(script, what, printed, snippet, log_value, note=""):
    ln = sub_line(snippet)
    assert ln is not None, f"submitted snippet not found: {snippet!r}"
    if log_value is None:
        ok = None
    else:
        d = len(printed.split(".")[1]) if "." in printed else 0
        # reproduced when the log value lies within half a unit of the printed last digit (either rounding
        # convention of a value the log prints at higher precision, e.g. 0.945 printed as 0.95)
        ok = abs(float(log_value) - float(printed)) <= 0.5 * 10 ** -d + 1e-12
    rows.append({"script": script, "quantity": what, "submitted_printed": printed, "submitted_line": ln,
                 "original_on_submitted_inputs": log_value, "reproduced": ok, "note": note})


RA, FT, P5, RE, SX, CM = LOGS.keys()
# robustness_analyses.py (frozen log)
row(RA, "4.1 stratified-bootstrap rho CI lower", "-0.574", "95\\% CI: -0.574", grab(RA, r"95% CI: \[([+-][\d.]+),"))
row(RA, "4.1 stratified-bootstrap rho CI upper", "-0.514", "-0.574, -0.514", grab(RA, r"95% CI: \[[+-][\d.]+, ([+-][\d.]+)\]"))
row(RA, "contribution (b) PD mean cosine", "0.979", "PD mean cosine 0.979", grab(RA, r"PD \(6 languages\):\n\s+Mean cosine: ([\d.]+)"))
row(RA, "contribution (b) PD lower", "0.957", "0.979 {[}95\\% CI: 0.957", grab(RA, r"PD \(6 languages\):\n\s+Mean cosine: [\d.]+ \[([\d.]+)"))
row(RA, "contribution (b) tightest pair lower bound", "0.899", "lower bound of 0.899", grab(RA, r"Tightest pair: nl-pt: [\d.]+ \[([\d.]+)"))
row(RA, "LODO rho range lower", "-0.575", "-0.575 to -0.410", grab(RA, r"LODO severity rho: range \[([+-][\d.]+)"))
row(RA, "LODO eps2 mean", "0.289", "mean 0.289", grab(RA, r"LODO eta2: range \[[\d.]+, [\d.]+\], mean ([\d.]+)"))
# fixed_token_dprime.py (frozen log)
row(FT, "Table 6 rho at 200 tokens", "-0.733", "200 & 467 & -0.733", grab(FT, r"Severity rho: ([+-][\d.]+) \(p=[\d.e+-]+, n=467\)"))
row(FT, "Table 6 eps2 at 20 tokens", "0.355", "-0.586 & < $10^{-78}$ & 0.355", grab(FT, r"eps2=([\d.]+)"))
row(FT, "5.1 permutation: PD observed cosine", "0.979", "observed 0.979, null 0.982", grab(FT, r"Observed mean cosine: ([\d.]+)"))
row(FT, "5.1 permutation: PD null mean", "0.982", "null 0.982, p = 0.84", grab(FT, r"Null distribution: mean=([\d.]+)"),
    "the original's Experiment B on the submitted inputs gives null 0.9975 and p = 1.00, not the printed 0.982 / 0.84")
row(FT, "5.1 permutation: PD p", "0.84", "p = 0.84)", grab(FT, r"p\(perm >= observed\): ([\d.]+)"))
# robustness_pass5.py (frozen log)
row(P5, "common-set rho at 20 tokens", "-0.747", "rho = -0.747 to -0.733", grab(P5, r"Budget\s+20: n=467, rho=([+-][\d.]+)"))
row(P5, "min-HC PD cosine", "0.979", "PD cosine remains 0.979", grab(P5, r"Minimum HC >= 5 per language ===\n(?:.*\n)?\s+PD: \d+ langs, cos=([\d.]+)"))
row(P5, "min-HC CP cosine (3 languages)", "0.984", "CP 0.984 (3 languages)", grab(P5, r"CP: 3 langs, cos=([\d.]+)"))
# reviewer_experiments.py (v8 submitted-input re-run; no frozen log)
row(RE, "Holm HC vs PD d", "1.00", "PD vs. HC d = +1.00", grab(RE, r"HC vs PD\s+\S+\s+\S+\s+\S+\s+([+-][\d.]+)"))
row(RE, "Holm CP vs PD d", "-1.01", "PD vs. CP d = -1.01", grab(RE, r"CP vs PD\s+\S+\s+\S+\s+\S+\s+([+-][\d.]+)"))
row(RE, "token-matched control vs mild pairs", "185", "n = 185 matched pairs", grab(RE, r"Control vs Mild: n_matched=(\d+)"))
row(RE, "token-matched control vs mild d", "0.60", "Cohen's d = 0.60", grab(RE, r"Control vs Mild: n_matched=\d+, d=([+-][\d.]+)"))
row(RE, "token-matched moderate vs severe d", "0.95", "d = 0.95, p", grab(RE, r"Moderate vs Severe: n_matched=\d+, d=([+-][\d.]+)"))
row(RE, "min-n (n>=10) PD cosine", "0.976", "(PD cosine 0.976)", grab(RE, r"Minimum n >= 10 per language-aetiology cell ===\n\s+PD: \d+ langs, \d+ pairs, mean cos=([\d.]+)"))
# sap_excluded_sensitivity.py (v8 submitted-input re-run; repo script)
row(SX, "SAP-excluded n", "2,161".replace(",", ""), "(n = 2,161)", grab(SX, r"SAP EXCLUDED \(n=(\d+)\)"))
row(SX, "SAP-excluded severity rho", "-0.410", "rho = -0.410", grab(SX, r"n=1573, rho=([+-][\d.]+)"))
row(SX, "SAP-excluded composite eps2", "0.106", "eps2 = 0.106", grab(SX, r"SAP EXCLUDED[\s\S]*?H=[\d.]+, p=[\d.e+-]+, eps2=([\d.]+)"))
row(SX, "SAP-excluded composite H", "182", "H = 182", grab(SX, r"SAP EXCLUDED[\s\S]*?H=([\d.]+), p=[\d.e+-]+, eps2"))
row(SX, "nasal eps2 with SAP", "0.152", "nasal 0.152", grab(SX, r"nasal_dprime\s+([\d.]+)\s+[\d.]+\s+[+-]"))
row(SX, "nasal eps2 without SAP", "0.010", "0.152→0.010", grab(SX, r"nasal_dprime\s+[\d.]+\s+([\d.]+)\s+[+-]"))
row(SX, "high eps2 without SAP", "0.204", "high 0.190→0.204", grab(SX, r"high_dprime\s+[\d.]+\s+([\d.]+)\s+[+-]"))
row(SX, "round eps2 without SAP", "0.221", "round 0.165→0.221", grab(SX, r"round_dprime\s+[\d.]+\s+([\d.]+)\s+[+-]"))
# cross_model_comparison.py (v8 submitted-input re-run, local frozen backbone files)
row(CM, "per-backbone rho MMS", "-0.495", "MMS-300M rho = -0.495", grab(CM, r"^\s+mms\s+(?:[+-][\d.]+\s+){5}([+-][\d.]+)"))
row(CM, "per-backbone rho HuBERT-base", "-0.561", "HuBERT-base rho = -0.561", grab(CM, r"^\s+hubert-base\s+(?:[+-][\d.]+\s+){5}([+-][\d.]+)"))
tbl = re.search(r"TABLE 5: Feature Importance Ranking[\s\S]*?\n\s+nasal\s+(.+)\n\s+voicing\s+(.+)\n\s+sonorant\s+(.+)\n\s+strident\s+(.+)\n\s+manner\s+(.+)\n",
                TXT[CM])
ranks = {f: [int(x) for x in re.findall(r"(\d)\(", tbl.group(i + 1))]
         for i, f in enumerate(["nasal", "voicing", "sonorant", "strident", "manner"])}
nasal_first = sum(1 for r in ranks["nasal"] if r == 1)
top3_all = [f for f, r in ranks.items() if all(x <= 3 for x in r)]
rows.append({"script": CM, "quantity": "nasality ranks first in 3 of 6 models", "submitted_printed": "3",
             "submitted_line": sub_line("Nasality ranks first in 3 of 6 models"),
             "original_on_submitted_inputs": nasal_first, "reproduced": nasal_first == 3, "note": ""})
rows.append({"script": CM, "quantity": "top 3 (nasality, voicing, stridency) stable across all 6 backbones",
             "submitted_printed": "nasality, voicing, stridency in the top 3 of all 6",
             "submitted_line": sub_line("stable across all 6 backbones"),
             "original_on_submitted_inputs": {"ranks_per_backbone": ranks, "features_top3_in_all_six": top3_all},
             "reproduced": set(top3_all) == {"nasal", "voicing", "strident"},
             "note": "on the submitted inputs no contrast is in the top 3 of all six backbones"})

summary = {}
for r in rows:
    s = summary.setdefault(r["script"], {"reproduced": 0, "not_reproduced": 0, "not_reproduced_items": []})
    if r["reproduced"]:
        s["reproduced"] += 1
    else:
        s["not_reproduced"] += 1
        s["not_reproduced_items"].append(r["quantity"])
out = {"definition": __doc__, "submitted_tex_sha256": hashlib.sha256(SUB.encode("utf-8")).hexdigest(),
       "logs": {k: {"path": str(p.relative_to(REV)).replace("\\", "/"), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for k, p in LOGS.items()},
       "rows": rows, "summary": summary,
       "by_quantity": {r["quantity"]: {"submitted_printed": r["submitted_printed"],
                                        "original_on_submitted_inputs": r["original_on_submitted_inputs"],
                                        "reproduced": r["reproduced"]} for r in rows}}
(REV / "results" / "v10" / "v10_originals_reproduction.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
