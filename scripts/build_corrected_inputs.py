# Repair step 1 (LaVonne v1.3 item 2): build corrected inputs into revision_2026-10/corrected/.
# Frozen package is NOT touched. Every change is counted and reported.
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FP = BASE / "frozen_package" / "results"
OUT = BASE / "corrected"
OUT.mkdir(exist_ok=True)

changes = {}


def fix_rows(rows, tag):
    n_mdsc = n_copas = n_hu = n_sap = 0
    for r in rows:
        if r.get("dataset") == "MDSC" and r.get("aetiology") == "cerebral_palsy" \
                and r.get("is_control", "").lower() == "true":
            r["aetiology"] = "healthy"
            if "severity_label" in r and r["severity_label"] not in ("control", ""):
                r["severity_label"] = "control"
            n_mdsc += 1
        if r.get("dataset") == "COPAS" and r.get("aetiology") != "healthy" \
                and r.get("is_control", "").lower() == "true":
            r["is_control"] = "False"
            if r.get("severity_label") == "control":
                r["severity_label"] = "unknown"
            n_copas += 1
        if r.get("dataset") == "Hungarian_Dysarthria" and r.get("aetiology") != "healthy" \
                and r.get("is_control", "").lower() == "true":
            r["is_control"] = "False"
            n_hu += 1
        if r.get("dataset") == "SAP_unlabeled":
            r["dataset"] = "SAP"
            n_sap += 1
    changes[tag] = {"mdsc_cp_controls_to_healthy": n_mdsc,
                    "copas_nonhealthy_control_unset": n_copas,
                    "hu_mixed_control_unset": n_hu,
                    "sap_unlabeled_canonicalised": n_sap}
    return rows


def process(fname):
    with open(FP / fname, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        fields = rdr.fieldnames
    rows = fix_rows(rows, fname)
    with open(OUT / fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


for fname in ["track4_master.csv", "speaker_inventory.csv",
              "track4_results_hubert-large.csv", "track4_results_wavlm.csv",
              "track4_results_wav2vec2.csv", "track4_results_xlsr.csv",
              "track4_results_mms.csv"]:
    process(fname)

(OUT / "CHANGES_step1.json").write_text(json.dumps(changes, indent=1), encoding="utf-8")
print(json.dumps(changes, indent=1))
print("NOTE: nl d-prime values in track4_master.csv are still the contaminated-direction values;"
      " they are replaced by the DGX nl_repair output in step 2.")
