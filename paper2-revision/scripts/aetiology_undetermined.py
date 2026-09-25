#!/usr/bin/env python3
"""Item 3d: count aetiology-undetermined speakers in corrected master.
Expected: 68 unknown + 1 MS = 69.
"""
import csv, json, pathlib
from collections import Counter

MASTER = pathlib.Path(__file__).resolve().parent.parent / "corrected" / "track4_master.csv"
OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "rebuild_2026-09-23" / "aetiology_undetermined.json"

with MASTER.open(encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    rows = list(rdr)

# Try common column names
aet_cols = [c for c in rows[0].keys() if 'aetiology' in c.lower() or c.lower() in ('aet', 'diagnosis')]
print("candidate aetiology columns:", aet_cols)

# Try aetiology column
main_six = {"healthy", "hc", "control", "pd", "parkinson", "cp", "als", "ds", "downsyndrome", "stroke",
            "parkinsons", "cerebral_palsy", "down_syndrome", "downs"}

col = aet_cols[0] if aet_cols else None
if col is None:
    raise SystemExit("no aetiology column found")

counts = Counter()
undetermined_by_label = Counter()
for r in rows:
    v = (r.get(col) or "").strip().lower()
    counts[v] += 1
    # collapse into main-six or other
    if not any(k in v for k in main_six):
        undetermined_by_label[v] += 1

undet_total = sum(undetermined_by_label.values())

# Try to isolate MS
ms = sum(v for k, v in undetermined_by_label.items() if 'ms' in k or 'sclerosis' in k or 'multiple' in k)
unknown = sum(v for k, v in undetermined_by_label.items() if k in {'unknown', 'undetermined', 'n/a', '', 'none', 'mixed'})
result = {
    "aetiology_column": col,
    "total_rows": len(rows),
    "aetiology_value_counts": dict(counts),
    "undetermined_by_label": dict(undetermined_by_label),
    "undetermined_total": undet_total,
    "multiple_sclerosis_count": ms,
    "unknown_count": unknown,
    "expected_prose": "68 with unknown aetiology plus 1 with multiple sclerosis, an aetiology outside the main six analysed",
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
