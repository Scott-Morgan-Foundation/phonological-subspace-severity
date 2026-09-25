#!/usr/bin/env python3
"""Item 6b: recompute COPAS control values from corrected master."""
import csv, json, pathlib, statistics

MASTER = pathlib.Path(__file__).resolve().parent.parent / "corrected" / "track4_master.csv"
OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "rebuild_2026-09-23" / "copas_recompute_v4.json"

def flt(v):
    try:
        x = float(v)
        return x if x == x else None  # NaN
    except (TypeError, ValueError):
        return None

with MASTER.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def mean(vs):
    vs = [v for v in vs if v is not None]
    return statistics.mean(vs) if vs else None, len(vs)

# COPAS controls
copas_ctrl = [r for r in rows if r['dataset'] == 'COPAS' and r['is_control'].lower() in ('true','1','yes')]
copas_all  = [r for r in rows if r['dataset'] == 'COPAS']

# Global healthy controls (all corpora)
all_ctrl = [r for r in rows if r['is_control'].lower() in ('true','1','yes')]
all_speakers = rows

def stat(name, field, group):
    vals = [flt(r[field]) for r in group]
    m, n = mean(vals)
    return {"name": name, "field": field, "n": n, "mean": round(m, 4) if m is not None else None}

result = {
    "COPAS_controls_n_records": len(copas_ctrl),
    "COPAS_total_n_records": len(copas_all),
    "global_controls_n_records": len(all_ctrl),
    "all_speakers_n_records": len(all_speakers),
    "COPAS_controls": {
        "lowness": stat("COPAS controls lowness", "low_dprime", copas_ctrl),
        "nasality": stat("COPAS controls nasality", "nasal_dprime", copas_ctrl),
    },
    "global_controls": {
        "lowness": stat("Global controls lowness", "low_dprime", all_ctrl),
        "nasality": stat("Global controls nasality", "nasal_dprime", all_ctrl),
    },
    "all_speakers": {
        "lowness": stat("All-speaker lowness", "low_dprime", all_speakers),
        "nasality": stat("All-speaker nasality", "nasal_dprime", all_speakers),
    },
    "COPAS_all_rows": {
        "lowness": stat("COPAS all rows lowness", "low_dprime", copas_all),
        "nasality": stat("COPAS all rows nasality", "nasal_dprime", copas_all),
    },
    "lavonne_targets": {
        "COPAS_control_lowness": 8.54,
        "COPAS_control_lowness_n": 7,
        "global_control_lowness_mean": 3.03,
        "COPAS_nasality": 3.55,
        "all_speaker_nasality_mean_approx": 3.0,
        "global_control_nasality_mean_target": 4.07,
    },
}

OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
