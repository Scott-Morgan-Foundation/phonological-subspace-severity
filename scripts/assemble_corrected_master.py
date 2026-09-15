# Repair step 3a: merge nl_repair d-prime values into the corrected master, freeze + checksum.
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CORR = BASE / "corrected"
NL = BASE / "results" / "nl_repair_dprime.csv"
DCOLS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime",
         "manner_dprime", "high_dprime", "low_dprime", "back_dprime", "round_dprime"]


def safe(s):
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")


nl = {}
for r in csv.DictReader(open(NL, encoding="utf-8")):
    nl[(r["dataset"], r["speaker_id"])] = r

with open(CORR / "track4_master.csv", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    rows = list(rdr)
    fields = rdr.fieldnames

replaced = missing = 0
changed_vals = 0
for r in rows:
    if r["language"] != "nl":
        continue
    k = (r["dataset"], safe(r["speaker_id"]))
    if k not in nl:
        missing += 1
        for c in DCOLS:
            r[c] = ""
        continue
    src = nl[k]
    for c in DCOLS:
        old, new = r[c], src.get(c, "")
        if old != new:
            changed_vals += 1
        r[c] = new
    replaced += 1

with open(CORR / "track4_master.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

sums = {}
for p in sorted(CORR.glob("*.csv")):
    sums[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(CORR / "SHA256SUMS_corrected.json").write_text(json.dumps(sums, indent=1), encoding="utf-8")
print(f"nl rows re-scored: {replaced}; nl rows WITHOUT repair values: {missing}; "
      f"d-prime cells changed: {changed_vals}")
print(json.dumps(sums, indent=1))
