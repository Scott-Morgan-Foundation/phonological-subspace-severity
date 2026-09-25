#!/usr/bin/env python3
"""v8: provenance of the HuBERT-base values for the 18 SLR65_Tamil speakers (Methods §3.1, §4.2).

Runs on the DGX (stdlib only):  python3 v8_tamil_hubert_provenance.py > v8_tamil_hubert_provenance.json
Reads the DGX Track 4 result files and extraction logs, and records:
  - SHA256 and mtime of each file used;
  - per file: number of SLR65_Tamil and SSNCE_Tamil rows, how many carry nasal d', and the
    SLR65 mean nasal d' (the value that identifies which set of 18 values a file holds);
  - every log line that states the Tamil direction pool, and the SLR65 summary line of the
    2026-04-13 HuBERT-base run.
It asserts nothing about runs whose logs are absent.
"""
import csv
import hashlib
import json
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

BASE = Path.home() / "dysarthria"
FILES = ["results/track4/track4_master_backup_20260409.csv",
         "results/track4/track4_results_hubert_base.csv",
         "results/track4/track4_results_hu_ta_update.csv",
         "results/track4/track4_master.csv"]
LOGS = ["logs/extract_ssnce_tamil.log", "logs/hu_ta_backbones.log", "logs/multi_backbone.log"]


def meta(p):
    b = p.read_bytes()
    return {"sha256": hashlib.sha256(b).hexdigest(),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}


def num(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


out = {"definition": __doc__, "files": {}, "logs": {}}
for f in FILES:
    p = BASE / f
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    slr = [r for r in rows if r["dataset"] == "SLR65_Tamil"]
    ss = [r for r in rows if r["dataset"] == "SSNCE_Tamil"]
    nas = [num(r.get("nasal_dprime")) for r in slr]
    nas = [x for x in nas if x is not None]
    blocks = []
    seen = {}
    for r in slr:
        seen.setdefault(r["speaker_id"], []).append(num(r.get("nasal_dprime")))
    first = [v[0] for v in seen.values() if v[0] is not None]
    second = [v[1] for v in seen.values() if len(v) > 1 and v[1] is not None]
    out["files"][f] = {**meta(p), "n_rows": len(rows), "slr65_rows": len(slr),
                       "slr65_unique_speakers": len(seen),
                       "slr65_rows_with_nasal": len(nas),
                       "slr65_mean_nasal_first_occurrence": round(statistics.mean(first), 4) if first else None,
                       "slr65_mean_nasal_second_occurrence": round(statistics.mean(second), 4) if second else None,
                       "ssnce_rows": len(ss),
                       "ssnce_rows_with_nasal": sum(1 for r in ss if num(r.get("nasal_dprime")) is not None),
                       "ssnce_id_examples": [r["speaker_id"] for r in ss[:3]]}
for f in LOGS:
    p = BASE / f
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    keep = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if re.search(r"HC phone tokens from \['SLR65_Tamil'", s) or s.startswith("Loading ") \
                or s.startswith("Merged with existing") or s.startswith("Wrote ") \
                or s.startswith("Auto-adding HC dataset 'SLR65_Tamil'") or s.startswith("Extracted for"):
            keep.append([i, s[:200]])
    summ = []
    for i, ln in enumerate(lines):
        if ln.strip() == "SLR65_Tamil: 18 speakers":
            summ.append([i + 1] + [x.strip() for x in lines[i + 1:i + 13] if "nasal_dprime" in x])
    out["logs"][f] = {**meta(p), "lines": keep, "slr65_summary_nasal": summ}
print(json.dumps(out, indent=1))
