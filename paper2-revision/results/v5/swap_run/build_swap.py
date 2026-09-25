"""Build original/ and swapped/ sandboxes for the HuBERT-base Tamil swap test.

swapped master = corrected/track4_master.csv with the 9 d-prime columns of the
18 SLR65_Tamil rows replaced by the SECOND SLR65 block of
track4_results_hu_ta_update.csv (pooled SLR65+SSNCE directions).
"""
import csv, hashlib, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parents[2]
SRC = REV / "corrected"
UPD = HERE / "track4_results_hu_ta_update.csv"
DP = ["back_dprime", "high_dprime", "low_dprime", "manner_dprime", "nasal_dprime",
      "round_dprime", "sonorant_dprime", "strident_dprime", "voicing_dprime"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


upd = list(csv.DictReader(open(UPD, encoding="utf-8")))
slr = [(i, r) for i, r in enumerate(upd) if r["dataset"] == "SLR65_Tamil"]
first, second = {}, {}
for i, r in slr:
    (second if r["speaker_id"] in first else first)[r["speaker_id"]] = r
assert len(first) == 18 and len(second) == 18, (len(first), len(second))

with open(SRC / "track4_master.csv", encoding="utf-8", newline="") as f:
    rd = csv.DictReader(f)
    fields = rd.fieldnames
    master = list(rd)

check = []
swapped = []
n_rep = 0
for r in master:
    r2 = dict(r)
    if r["dataset"] == "SLR65_Tamil" and r["speaker_id"] in second:
        a, b = first[r["speaker_id"]], second[r["speaker_id"]]
        match_first = all(r[c] == a[c] for c in DP)
        for c in DP:
            r2[c] = b[c]
        n_rep += 1
        check.append({"speaker_id": r["speaker_id"], "master_equals_first_block": match_first,
                      "nasal_before": r["nasal_dprime"], "nasal_after": b["nasal_dprime"]})
    swapped.append(r2)
assert n_rep == 18, n_rep

for name, rows in (("original", master), ("swapped", swapped)):
    base = HERE / name
    (base / "corrected").mkdir(parents=True, exist_ok=True)
    (base / "results").mkdir(parents=True, exist_ok=True)
    (base / "scripts" / "corrected").mkdir(parents=True, exist_ok=True)
    for p in SRC.glob("track4_results_*.csv"):
        shutil.copy2(p, base / "corrected" / p.name)
    shutil.copy2(SRC / "speaker_inventory.csv", base / "corrected" / "speaker_inventory.csv")
    with open(base / "corrected" / "track4_master.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for p in (REV / "scripts" / "corrected").glob("*.py"):
        shutil.copy2(p, base / "scripts" / "corrected" / p.name)

meta = {
    "update_file": str(UPD.name), "update_sha256": sha(UPD),
    "corrected_master_sha256": sha(SRC / "track4_master.csv"),
    "original_sandbox_master_sha256": sha(HERE / "original/corrected/track4_master.csv"),
    "swapped_sandbox_master_sha256": sha(HERE / "swapped/corrected/track4_master.csv"),
    "rows_replaced": n_rep, "columns_replaced": DP,
    "all_master_rows_equal_first_block": all(c["master_equals_first_block"] for c in check),
    "per_speaker": check,
}
(HERE / "swap_build_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in meta.items() if k != "per_speaker"}, indent=1))
