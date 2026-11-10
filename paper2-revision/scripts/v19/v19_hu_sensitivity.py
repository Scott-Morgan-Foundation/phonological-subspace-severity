#!/usr/bin/env python3
"""v19 (CSL R2): does the Hungarian_Dysarthria reproduction discrepancy change any reported result?

The cause of the discrepancy is not investigated (no audio is re-extracted). Every script that turns the
speaker-level tables into a reported number is re-run, unchanged, on four versions of those tables:

  V0_retained     the corrected tables as used for the paper (Hungarian values = submitted values)
  V1_reextracted  in the HuBERT-base master, the nine phonological d' values of all 59 Hungarian speakers
                  replaced by the 2026-09-13 re-extraction with the released Hungarian configuration
                  (results/test9_hu_dprime.csv); a value missing in the re-extraction is set missing.
                  The five other backbone tables are left as they are (they were not re-extracted).
  V2_excl_HD      the 31 Hungarian_Dysarthria speakers removed from all six backbone tables
  V3_excl_all_hu  all 59 Hungarian speakers removed (also removes the 28 Hungarian controls from every
                  healthy-control reference)

In V0 every script must exit 0 (v10_facts.py exits 1 in V2 only: it reports facts about the removed corpus). In V1-V3 the consistency check is expected to exit 1: its
failures list the printed numbers that would change. Each variant gets a mirror of the revision folder (scripts/, corrected/, results/, frozen_package/,
figures/, external/); only corrected/ differs. The producer scripts below run there in dependency order,
as subprocesses, with their own relative paths. V0 must reproduce the saved canonical outputs.
The fixed-token analysis is not re-run: it holds no Hungarian speaker (asserted).

Usage:  v19_hu_sensitivity.py run <variant>   (the four can run in parallel)
        v19_hu_sensitivity.py compare
Output: results/v19/hu_sensitivity/{<variant>/..., diff.json}
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[2]
OUTD = REV / "results" / "v19" / "hu_sensitivity"
WORK = Path(os.environ.get("V19_WORK", str(REV.parent / "_v19_hu_mirrors")))
HU_RE = REV / "results" / "test9_hu_dprime.csv"
BACKBONE_FILES = ["track4_master.csv", "track4_results_hubert-large.csv", "track4_results_wavlm.csv",
                  "track4_results_wav2vec2.csv", "track4_results_xlsr.csv", "track4_results_mms.csv"]
MAP = {"manner": "manner_dprime", "nasal": "nasal_dprime", "sonorant": "sonorant_dprime",
       "strident": "strident_dprime", "voicing": "voicing_dprime", "vowel_back": "back_dprime",
       "vowel_high": "high_dprime", "vowel_low": "low_dprime", "vowel_round": "round_dprime"}
VARIANTS = ["V0_retained", "V1_reextracted", "V2_excl_HD", "V3_excl_all_hu"]
# (cwd relative to scripts/, script) in dependency order
PRODUCERS = [
    ("corrected", "test2_aetiology_vs_language.py"), ("corrected", "test3_per_feature_effects.py"),
    ("corrected", "test6_backbone_stability.py"), ("corrected", "test7_lodo_repro.py"),
    ("corrected", "test9_recompute.py"),
    (".", "rebuild_phase1.py"), (".", "aetiology_undetermined.py"), (".", "regen_fig2_v5.py"),
    ("v6", "v6_master_analyses.py"), ("v6", "v6_classifier.py"), ("v6", "v6_crossling.py"),
    ("v6", "v6_test45_analysis.py"), ("v6", "v6_table1_check.py"),
    ("v8", "v8_test6.py"), ("v8", "v8_run_originals.py"), ("v8", "v8_master_facts.py"),
    ("v8", "v8_test2_sensitivity.py"), ("v8", "v8_token_quartiles.py"), ("v8", "v8_cp_retention.py"),
    ("v9", "v9_originals_rerun.py local"), ("v9", "v9_originals_rerun.py parse"),
    ("v10", "v10_facts.py"), ("v10", "regen_fig4_v10.py"), ("v10", "v10_hr_values.py"),
    ("v12", "v12_matched_severity_features.py"), ("v17", "v17_matched_severity_samelang.py"),
    ("v15", "check_consistency.py"),  # re-derives every bound printed number of the R1 text; exits 1 on any change
]
MIRROR_DIRS = ["scripts", "corrected", "results", "frozen_package", "figures", "external"]
ROOT_GLOBS = ["*.tex", "*.md", "*.txt"]  # manuscripts, letters and numbers maps read by the consistency check


def rows_of(p):
    with open(p, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def write_rows(p, fields, rows):
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_inputs(name, dst):
    log = {}
    re_tab = {(r["dataset"], r["speaker_id"]): r for r in rows_of(HU_RE)[1]}
    for fn in BACKBONE_FILES:
        fields, rows = rows_of(REV / "corrected" / fn)
        hu = [r for r in rows if r["language"] == "hu"]
        if name == "V0_retained" or (name == "V1_reextracted" and fn != "track4_master.csv"):
            continue  # mirror copy is already the canonical file
        if name == "V1_reextracted":
            nrep = nmiss = 0
            for r in hu:
                e = re_tab[(r["dataset"], r["speaker_id"])]
                for a, b in MAP.items():
                    if e[a] in ("", "nan", "NaN"):
                        nmiss += r[b] not in ("", "nan", "NaN")
                        r[b] = ""
                    else:
                        r[b] = e[a]
                        nrep += 1
            log[fn] = {"hu_rows": len(hu), "values_replaced": nrep, "values_newly_missing": nmiss}
        else:
            keep = (lambda r: r["dataset"] != "Hungarian_Dysarthria") if name == "V2_excl_HD" \
                else (lambda r: r["language"] != "hu")
            n0 = len(rows)
            rows = [r for r in rows if keep(r)]
            log[fn] = {"rows_removed": n0 - len(rows)}
        write_rows(dst / "corrected" / fn, fields, rows)
    return log


def run_variant(name):
    m = WORK / name
    if m.exists():
        shutil.rmtree(m)
    for d in MIRROR_DIRS:
        if (REV / d).exists():
            shutil.copytree(REV / d, m / d, ignore=shutil.ignore_patterns("v19", "__pycache__"))
    for g in ROOT_GLOBS:
        for f in REV.glob(g):
            shutil.copy2(f, m / f.name)
    log = {"inputs": make_inputs(name, m), "scripts": []}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    t_start = time.time()
    for cwd, script in PRODUCERS:
        t0 = time.time()
        p = subprocess.run([sys.executable] + script.split(), cwd=m / "scripts" / cwd, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        (m / f"_console_{script.replace(' ', '_')}.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
        log["scripts"].append({"script": f"{cwd}/{script}", "rc": p.returncode, "sec": round(time.time() - t0, 1)})
        print(name, script, p.returncode, round(time.time() - t0, 1), file=sys.stderr, flush=True)
    written = []
    for f in (m / "results").rglob("*"):
        if f.is_file() and f.stat().st_mtime >= t_start and f.suffix in (".json", ".csv"):
            written.append(str(f.relative_to(m)).replace("\\", "/"))
    log["written"] = sorted(written)
    (OUTD / name).mkdir(parents=True, exist_ok=True)
    (OUTD / name / "run_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


def flatten(obj, pre=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) and "item" in x for x in obj):
            for x in obj:
                out.update(flatten({k: v for k, v in x.items() if k != "item"}, f"{pre}[{x['item']}]"))
        else:
            for i, v in enumerate(obj):
                out.update(flatten(v, f"{pre}[{i}]"))
    elif isinstance(obj, bool):
        out[pre] = obj
    elif isinstance(obj, (int, float)):
        out[pre] = float(obj)
    elif isinstance(obj, str):
        out[pre] = obj
    return out


def load(p):
    if p.suffix == ".json":
        return flatten(json.loads(p.read_text(encoding="utf-8")))
    df = pd.read_csv(p, low_memory=False)
    keyc = [c for c in df.columns if df[c].dtype == object]
    out = {}
    for i, r in df.iterrows():
        k = "|".join(str(r[c]) for c in keyc) or str(i)
        for c in df.columns:
            if c not in keyc:
                out[f"{k}.{c}"] = None if pd.isna(r[c]) else float(r[c])
    return out


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))
    return a == b


def is_pathlike(k, v):
    return isinstance(v, str) and (":\\" in v or "/" in v or "\\" in v or k.endswith(("sha256", "sha", "date", "time")))


def compare():
    logs = {v: json.loads((OUTD / v / "run_log.json").read_text(encoding="utf-8")) for v in VARIANTS}
    files = sorted(set().union(*[set(l["written"]) for l in logs.values()]))
    files = [f for f in files if not f.startswith("results/v8/originals/") and not f.startswith("results/v9/originals/")
             or f.endswith(".json")]
    res = {"inputs": {v: logs[v]["inputs"] for v in VARIANTS},
           "script_rc": {v: [s for s in logs[v]["scripts"] if s["rc"] != 0] for v in VARIANTS},
           "V0_vs_canonical": {}, "changes": {v: {} for v in VARIANTS[1:]}}
    for rel in files:
        canon_p = REV / rel
        mir = {v: WORK / v / rel for v in VARIANTS}
        if not mir["V0_retained"].exists():
            continue
        v0 = load(mir["V0_retained"])
        if canon_p.exists():
            c = load(canon_p)
            bad = [k for k in c if k in v0 and not same(c[k], v0[k]) and not is_pathlike(k, c[k])]
            res["V0_vs_canonical"][rel] = {"n_values": len(c), "n_mismatch": len(bad), "examples": bad[:5]}
        for v in VARIANTS[1:]:
            if not mir[v].exists():
                res["changes"][v][rel] = "MISSING"
                continue
            x = load(mir[v])
            ch = {}
            for k in sorted(set(v0) | set(x)):
                a, b = v0.get(k), x.get(k)
                if not same(a, b) and not is_pathlike(k, a if a is not None else b):
                    ch[k] = [a, b]
            if ch:
                res["changes"][v][rel] = ch
    (OUTD / "diff.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    summary = {"script_failures": {v: res["script_rc"][v] for v in VARIANTS},
               "V0_mismatch_files": {k: v for k, v in res["V0_vs_canonical"].items() if v["n_mismatch"]},
               "V0_files_compared": len(res["V0_vs_canonical"]),
               "changed_values": {v: {f: (len(c) if isinstance(c, dict) else c) for f, c in res["changes"][v].items()}
                                  for v in VARIANTS[1:]}}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    spk = pd.read_csv(REV / "results/v17/v17_fixed_token_hcdir_speakers.csv")
    assert (spk.language != "hu").all() and not spk.speaker.str.contains("Hungar").any()
    OUTD.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "run":
        run_variant(sys.argv[2])
    else:
        compare()
