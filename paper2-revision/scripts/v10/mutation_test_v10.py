#!/usr/bin/env python3
"""Planted-error test for scripts/v10/check_consistency.py. Plants ONE error at a time in a copy of the manuscript (or
Highlights) inside the given package root, runs the check there, and records whether it exits non-zero; restores the
file after every run and re-runs the clean check at the end.
Part A: the 19 errors the v9 checker planted (adapted where the v10 text changed).
Part B: further errors in the high-risk regions (abstract, Highlights, contributions, tables, figure captions,
Conclusion), where every error must be caught.
Usage: python scripts/v10/mutation_test_v10.py [<package root>]   (default: the directory two levels up)
Output: <root>/results/v10/check_consistency_mutation_test.json and .md"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
TEX, HL = ROOT / "paper2-csl_v10.tex", ROOT / "submission_highlights_v2.txt"

A = [
    ("T3 nasality eps2 0.426->0.427", "tex", "& 0.426\n", "& 0.427\n", True),
    ("abstract 21.9% -> 22.9% macro F1", "tex", "limited (21.9\\% macro F1)", "limited (22.9\\% macro F1)", True),
    ("4.1 HC n 1,212 -> 1,221", "tex", "(HC, n = 1,212)", "(HC, n = 1,221)", False),
    ("token-matched d 0.94 -> 0.95", "tex", "(n = 37, d = 0.94,", "(n = 37, d = 0.95,", False),
    ("4.3 PD CI lower 0.977 -> 0.975", "tex", "PD 0.980 [0.977, 0.984]", "PD 0.980 [0.975, 0.984]", False),
    ("Kendall W PD 0.44 -> 0.45", "tex", "Kendall's W = 0.44 PD", "Kendall's W = 0.45 PD", False),
    ("Holm 12 of 15 -> 13 of 15", "tex", "12 of 15 pairs remain", "13 of 15 pairs remain", False),
    ("Table 5 HL-WavLM 0.968 -> 0.963", "tex", "HuBERT-large vs WavLM-base\n & 0.968", "HuBERT-large vs WavLM-base\n & 0.963", True),
    ("swap XLS-R/HuBERT-large severity rho", "tex", "XLS-R-300M rho = -0.590, HuBERT-base rho = -0.543, MMS-300M rho = -0.541, wav2vec2-base rho = -0.528, WavLM-base rho = -0.527, and HuBERT-large rho = -0.521",
     "XLS-R-300M rho = -0.521, HuBERT-base rho = -0.543, MMS-300M rho = -0.541, wav2vec2-base rho = -0.528, WavLM-base rho = -0.527, and HuBERT-large rho = -0.590", False),
    ("SAP-excluded p<10^-63 -> 10^-60", "tex", "p < $10^{-63}$) versus", "p < $10^{-60}$) versus", False),
    ("nine -> eight robustness analyses (word)", "tex", "We conducted nine robustness", "We conducted eight robustness", False),
    ("Q3 rho -0.39 -> -0.40", "tex", "rho = -0.39, p < 10$^{-16}$", "rho = -0.40, p < 10$^{-16}$", False),
    ("classifier HC F1 0.627 -> 0.672", "tex", "(F1 = 0.627)", "(F1 = 0.672)", False),
    ("CTC mean moderate 0.858 -> 0.885", "tex", "moderate 0.858", "moderate 0.885", False),
    ("VTA HC 31.2 -> 32.1", "tex", "(HC 31.2, PD 23.7", "(HC 32.1, PD 23.7", False),
    ("Swahili PD 0.991 -> 0.919", "tex", "cosine similarity 0.991 to the multi-language PD mean", "cosine similarity 0.919 to the multi-language PD mean", False),
    ("min-HC CP 3 languages -> 2 languages", "tex", "across 3 languages (against", "across 2 languages (against", False),
    ("LODO 24 deletions -> 25 (v10 wording)", "tex", "all 24 yielded significant", "all 25 yielded significant", False),
    ("Table 6 eps2 0.540 -> 0.504", "tex", "& 0.540\n", "& 0.504\n", True),
]
B = [
    ("abstract 3,374 -> 3,347 speakers", "tex", "analysis to 3,374 speakers", "analysis to 3,347 speakers", True),
    ("abstract block permutation p 0.055 -> 0.050", "tex", "(block permutation p = 0.055; Delta = +0.0004, 95\\% CI", "(block permutation p = 0.050; Delta = +0.0004, 95\\% CI", True),
    ("abstract 6 -> 7 SSL backbones (single digit)", "tex", "using 6 SSL backbones", "using 7 SSL backbones", True),
    ("contribution (b) PD lower 0.956 -> 0.958", "tex", "gives PD 0.978 {[}0.956,", "gives PD 0.978 {[}0.958,", True),
    ("contribution (a) moderate-only n 42 -> 44", "tex", "(n = 42 PD, 45 pooled)", "(n = 44 PD, 45 pooled)", True),
    ("Table 1 SAP 1,233 -> 1,223", "tex", "SAP [20] (1,233)", "SAP [20] (1,223)", True),
    ("Table 2 XLS-R 128 -> 126 languages", "tex", "128 languages, 436K h", "126 languages, 436K h", True),
    ("Table 4 PD languages 6 -> 5", "tex", "PD\n & 6 (en, es, it, nl, pt, sk)", "PD\n & 5 (en, es, it, nl, pt, sk)", True),
    ("Figure 1 13 -> 12 features", "tex", "Rows show 13 phonological", "Rows show 12 phonological", True),
    ("Figure 2 HC n 1,445 -> 1,454", "tex", "(HC n = 1,445,", "(HC n = 1,454,", True),
    ("Figure 4 XLS-R CI upper 0.12 -> 0.13", "tex", "95\\% CI [-0.02, 0.12]", "95\\% CI [-0.02, 0.13]", True),
    ("Table 6 caption fewer than 5 -> 6", "tex", "have fewer than 5 qualifying", "have fewer than 6 qualifying", True),
    ("Table 6 p bound 10^-94 -> 10^-95", "tex", "< $10^{-94}$", "< $10^{-95}$", True),
    ("Conclusion ablation rho -0.453 -> -0.435", "tex", "(rho = -0.453, epsilon-squared", "(rho = -0.435, epsilon-squared", True),
    ("Highlights d 0.01 -> 0.02", "hl", "pooled difference d = 0.01", "pooled difference d = 0.02", True),
]


def run():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "scripts/v10/check_consistency.py"], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return r.returncode, [l.strip() for l in r.stdout.splitlines() if l.startswith("  ")][:2]


res = []
for part, M in (("A", A), ("B", B)):
    for label, doc, old, new, high_risk in M:
        path = TEX if doc == "tex" else HL
        orig = path.read_text(encoding="utf-8")
        n = orig.count(old)
        if n == 0:
            res.append({"part": part, "error": label, "high_risk": high_risk, "status": "PATTERN NOT FOUND"})
            continue
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        try:
            path.write_text(orig.replace(old, new, 1), encoding="utf-8")
            rc, first = run()
        finally:
            shutil.copy2(bak, path)
            bak.unlink()
        res.append({"part": part, "error": label, "high_risk": high_risk, "exit": rc, "caught": rc != 0, "first": first})
# Part C: an estimate for the rest of the body. 20 distinct numbers with at least two decimals drawn at random (seed 42)
# among the manuscript numbers whose status in the clean report is LOCATED (matched to a saved output only by value,
# not to a registry entry); the first occurrence outside the high-risk regions has its last digit changed by one.
# Some of these sit inside a sentence bound by a registry context and are caught by check A; the rest depend on
# check B, which only locates values. The result is the catch rate for such body numbers, reported as found.
import random
import re
sys.path.insert(0, str(ROOT / "scripts" / "v10"))
import hr_regions as H  # noqa: E402
run()  # refreshes the clean report
rep = json.loads((ROOT / "results" / "v10" / "check_consistency_report.json").read_text(encoding="utf-8"))
tex0 = TEX.read_text(encoding="utf-8")
R = H.regions(tex0)
a0 = tex0.find("\\begin{abstract}")
cands = []
for t in rep["tokens"]:
    if t["doc"] != "tex" or t["status"] != "LOCATED" or not re.search(r"\d\.\d{2,}", t["tok"]):
        continue
    positions = [m.start() for m in re.finditer(re.escape(t["tok"]), tex0) if m.start() > a0]
    positions = [p for p in positions if not any(a <= p < b for a, b in R.values())]
    if positions:
        cands.append((t["tok"], positions[0]))
seen, pool = set(), []
for tok, pos in cands:
    if tok not in seen:
        seen.add(tok)
        pool.append((tok, pos))
random.Random(42).shuffle(pool)
C = []
for tok, pos in pool[:20]:
    last = tok[-1]
    mut = tok[:-1] + str((int(last) + 1) % 10)
    C.append((f"body {tok} -> {mut} at char {pos}", pos, tok, mut))
for label, pos, tok, mut in C:
    orig = TEX.read_text(encoding="utf-8")
    bak = TEX.with_suffix(".tex.bak")
    shutil.copy2(TEX, bak)
    try:
        TEX.write_text(orig[:pos] + mut + orig[pos + len(tok):], encoding="utf-8")
        rc, first = run()
    finally:
        shutil.copy2(bak, TEX)
        bak.unlink()
    res.append({"part": "C", "error": label, "high_risk": False, "exit": rc, "caught": rc != 0, "first": first})
rc_clean, _ = run()
summ = {p: {"planted": sum(1 for r in res if r["part"] == p and "exit" in r),
            "caught": sum(1 for r in res if r["part"] == p and r.get("caught"))} for p in ("A", "B", "C")}
hr = [r for r in res if r["high_risk"] and "exit" in r]
summ["high_risk"] = {"planted": len(hr), "caught": sum(1 for r in hr if r["caught"])}
body = [r for r in res if not r["high_risk"] and "exit" in r]
summ["body"] = {"planted": len(body), "caught": sum(1 for r in body if r["caught"])}
summ["clean_rerun_exit"] = rc_clean
summ["patterns_not_found"] = [r["error"] for r in res if r.get("status") == "PATTERN NOT FOUND"]
out = ROOT / "results" / "v10"
(out / "check_consistency_mutation_test.json").write_text(json.dumps({"summary": summ, "results": res}, indent=1), encoding="utf-8")
lines = [f"# Planted-error test (root: package copy)", "", f"Summary: {json.dumps(summ)}", "",
         "| part | planted error | high-risk | exit | caught | first failure |", "|---|---|---|---|---|---|"]
for r in res:
    lines.append(f"| {r['part']} | {r['error']} | {'yes' if r['high_risk'] else 'no'} | {r.get('exit', r.get('status'))} | "
                 f"{r.get('caught', '')} | {' / '.join(x[:110] for x in r.get('first', [])).replace('|', '/')} |")
(out / "check_consistency_mutation_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summ, indent=1))
