#!/usr/bin/env python3
"""v8: run the ORIGINAL submitted analysis scripts, unchanged, on the corrected inputs.

The three scripts are byte-identical copies of the originals (SHA256 recorded below):
  - reviewer_experiments.py     (frozen_package/scripts; token-matched comparisons, severity-source
                                 ablation, min-n cosines, Holm post hoc)
  - cross_model_comparison.py   (frozen_package/scripts; per-backbone feature ranking, TABLE 5)
  - sap_excluded_sensitivity.py (public repo Scott-Morgan-Foundation/phonological-subspace-severity,
                                 commit 12e7950, 2026-04-22)
Each reads its inputs from $DYSARTHRIA_BASE/results/track4/. The only change is that path: this
wrapper builds a staging directory holding copies of the corrected inputs (or, for the
reproduction check, the submitted inputs from frozen_package/results) and sets DYSARTHRIA_BASE to it.
Console output of every run is saved verbatim to results/v8/originals/<input_state>/<script>.log;
the parsed values used in the manuscript are in results/v8/v8_originals.json.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
ORIG = Path(__file__).resolve().parent / "originals"
OUT = REV / "results" / "v8" / "originals"
STATES = {"corrected": REV / "corrected", "submitted": REV / "frozen_package" / "results"}
INPUT_FILES = ["track4_master.csv", "track4_results_hubert-large.csv", "track4_results_wavlm.csv",
               "track4_results_wav2vec2.csv", "track4_results_xlsr.csv", "track4_results_mms.csv"]
SCRIPTS = ["reviewer_experiments.py", "cross_model_comparison.py", "sap_excluded_sensitivity.py"]
EXPECTED = {"reviewer_experiments.py": "b5c4d1f71de2114b25521a6743c2787f0c1fbaf9c9afc53f5197725629f80432",
            "cross_model_comparison.py": "17ddf602f7ff9167b79a6a49c11f78dd8f519333124477fc1028b9e19fba192d",
            "sap_excluded_sensitivity.py": "e3f31b03759aea8cabed8e1800c6882890bd9ac1d32f016c72787a39ad36bd07"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_all():
    rec = {"scripts": {}, "states": {}}
    for s in SCRIPTS:
        h = sha(ORIG / s)
        assert h == EXPECTED[s], f"{s} is not the original ({h})"
        rec["scripts"][s] = h
    for state, src in STATES.items():
        stage = OUT / state / "stage"
        (stage / "results" / "track4").mkdir(parents=True, exist_ok=True)
        inputs = {}
        for fn in INPUT_FILES:
            p = src / fn
            if p.exists():
                shutil.copy2(p, stage / "results" / "track4" / fn)
                inputs[fn] = sha(p)
        env = {**os.environ, "DYSARTHRIA_BASE": str(stage), "PYTHONIOENCODING": "utf-8"}
        logs = {}
        for s in SCRIPTS:
            r = subprocess.run([sys.executable, str(ORIG / s)], env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            log = OUT / state / (s.replace(".py", ".log"))
            log.write_text(r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else ""), encoding="utf-8")
            logs[s] = {"returncode": r.returncode, "log": str(log.relative_to(REV)).replace("\\", "/"),
                       "log_sha256": sha(log)}
        rec["states"][state] = {"inputs": inputs, "runs": logs}
    return rec


def grab(text, pat, cast=float, flags=0):
    m = re.search(pat, text, flags)
    return None if m is None else [cast(g) for g in m.groups()]


def parse(rec):
    res = {}
    for state in STATES:
        L = {s: (REV / rec["states"][state]["runs"][s]["log"]).read_text(encoding="utf-8") for s in SCRIPTS}
        rv = L["reviewer_experiments.py"]
        tm = {}
        for name in ("Control vs Mild", "Mild vs Moderate", "Moderate vs Severe"):
            g = grab(rv, name + r": n_matched=(\d+), d=([+-][\d.]+), p=([\d.e+-]+)")
            tm[name] = None if g is None else {"n_matched": int(g[0]), "d": g[1], "p": g[2]}
        cm = L["cross_model_comparison.py"]
        t5 = cm[cm.index("TABLE 5"):cm.index("KEY FINDINGS")] if "TABLE 5" in cm else ""
        head = re.search(r"^\s+Feature\s+(.+)$", t5, re.M)
        models = head.group(1).split() if head else []
        ranks = {}
        for line in t5.splitlines():
            m = re.match(r"\s+(nasal|voicing|sonorant|strident|manner)\s+(.*)", line)
            if m:
                cells = re.findall(r"(\d+)\(([\d.]+)\)", m.group(2))
                ranks[m.group(1)] = {mod: {"rank": int(r), "abs_rho": float(v)} for mod, (r, v) in zip(models, cells)}
        by_model = {mod: sorted(ranks, key=lambda f: ranks[f][mod]["rank"]) for mod in models if all(mod in ranks[f] for f in ranks)}
        first = {}
        for order in by_model.values():
            first[order[0]] = first.get(order[0], 0) + 1
        top3 = [set(o[:3]) for o in by_model.values()]
        sx = L["sap_excluded_sensitivity.py"]

        def section(title):
            i = sx.find(title)
            return sx[i:] if i >= 0 else ""
        full, noSAP = section("FULL DATASET"), section("SAP EXCLUDED")
        noSAP = noSAP[:noSAP.find("COMPARISON")] if "COMPARISON" in noSAP else noSAP
        full = full[:full.find("SAP EXCLUDED")]

        def kw(block):
            d = {}
            for m in re.finditer(r"^\s+(\w+_dprime|vowel_triangle_area|speech_rate|pause_rate|vowel_duration_cv)\s+"
                                 r"([\d.]+)\s+([\d.e+-]+)\s+([\d.]+)", block, re.M):
                d[m.group(1)] = {"H": float(m.group(2)), "p": float(m.group(3)), "eps2": float(m.group(4))}
            comp = grab(block, r"H=([\d.]+), p=([\d.e+-]+), eps2=([\d.]+), k=(\d+), N=(\d+)")
            sev = grab(block, r"n=(\d+), rho=([+-]?[\d.]+), p=([\d.e+-]+)")
            n = grab(block, r"\(n=(\d+)\)", int)
            comp_d = None if comp is None else dict(zip(["H", "p", "eps2", "k", "N"], comp))
            sev_d = None if sev is None else dict(zip(["n", "rho", "p"], sev))
            return {"per_feature_kw_dysarthric_groups": d, "composite_kw": comp_d, "severity": sev_d,
                    "n_speakers": None if n is None else n[0]}
        top3n = {f: sum(1 for o in by_model.values() if f in o[:3]) for f in ranks}
        res[state] = {"token_matched": tm,
                      "feature_ranking_by_abs_severity_rho": {"models": models, "per_feature": ranks,
                                                              "order_by_model": by_model,
                                                              "first_counts": first,
                                                              "top3_common_to_all": sorted(set.intersection(*top3)) if top3 else [],
                                                              "top3_by_model": {k: v[:3] for k, v in by_model.items()},
                                                              "top3_count_per_feature": top3n},
                      "sap_excluded": {"full": kw(full), "sap_excluded": kw(noSAP),
                                       "composition_excl": {m.group(1): int(m.group(2)) for m in re.finditer(
                                           r"^\s+(HC|PD|CP|ALS|DS|Stroke): (\d+)$", noSAP, re.M)}}}
    return res


if __name__ == "__main__":
    rec = run_all() if "--parse-only" not in sys.argv else json.loads(
        (REV / "results" / "v8" / "v8_originals.json").read_text(encoding="utf-8"))
    rec["definition"] = __doc__
    rec["parsed"] = parse(rec)
    p = REV / "results" / "v8" / "v8_originals.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec["parsed"], indent=1)[:6000])
