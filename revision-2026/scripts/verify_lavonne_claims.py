# Verify LaVonne's v1.3 audit claims against the frozen master.
import csv
from collections import Counter, defaultdict

rows = list(csv.DictReader(open("frozen_package/results/track4_master.csv", encoding="utf-8")))

# 1. MDSC: CP-labelled rows with is_control=True
c = Counter()
for r in rows:
    if r["dataset"] == "MDSC":
        c[(r["aetiology"], r["is_control"])] += 1
print("MDSC aetiology x is_control:", dict(c))

# 2. COPAS: non-healthy speakers marked control
c2 = Counter()
for r in rows:
    if r["dataset"] == "COPAS" and r["is_control"].lower() == "true":
        c2[r["aetiology"]] += 1
print("COPAS is_control=True by aetiology:", dict(c2))
n_nonhealthy_ctrl = sum(v for k, v in c2.items() if k != "healthy")
print("COPAS non-healthy controls:", n_nonhealthy_ctrl, "(LaVonne: 115)")

# also global: is_control=True vs aetiology==healthy mismatch
mm = Counter()
for r in rows:
    ctrl = r["is_control"].lower() == "true"
    healthy = r["aetiology"] == "healthy"
    if ctrl != healthy:
        mm[(r["dataset"], r["aetiology"], r["is_control"])] += 1
print("\nALL control/healthy mismatches by (dataset, aetiology, is_control):")
for k, v in sorted(mm.items(), key=lambda kv: -kv[1]):
    print("  ", k, v)

# 3. SAP join vs backbone CSVs
mast_sap = {(r["dataset"], r["speaker_id"]) for r in rows if r["dataset"].startswith("SAP")}
mast_ds = Counter(r["dataset"] for r in rows if r["dataset"].startswith("SAP"))
bb = list(csv.DictReader(open("frozen_package/results/track4_results_wavlm.csv", encoding="utf-8")))
bb_sap = {(r["dataset"], r["speaker_id"]) for r in bb if r["dataset"].startswith("SAP")}
bb_ds = Counter(r["dataset"] for r in bb if r["dataset"].startswith("SAP"))
print("\nmaster SAP datasets:", dict(mast_ds), "| wavlm SAP datasets:", dict(bb_ds))
exact = mast_sap & bb_sap
print("exact-join SAP overlap:", len(exact), "of master", len(mast_sap), "/ wavlm", len(bb_sap))
mast_ids = {k[1] for k in mast_sap}
bb_ids = {k[1] for k in bb_sap}
print("speaker-id-only overlap:", len(mast_ids & bb_ids), "(dropped by exact join:", len(mast_ids & bb_ids) - len(exact), ")")

# 4. VOC-ALS in master
print("\nVOC-ALS rows in master:", sum(1 for r in rows if "VOC" in r["dataset"].upper()))
inv = list(csv.DictReader(open("frozen_package/results/speaker_inventory.csv", encoding="utf-8")))
print("VOC-ALS rows in speaker_inventory:", sum(1 for r in inv if "VOC" in r["dataset"].upper()))
