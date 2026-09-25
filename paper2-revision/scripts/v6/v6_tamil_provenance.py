"""v6: provenance record for the 18 duplicated SLR65 Tamil keys in the five non-HuBERT-base
backbone files (Methods §3.1). Computes, from the package inputs, which columns differ between
the two copies and the control-token totals that identify the two direction pools; records the
DGX extraction-log lines (quoted verbatim, with the log SHA256s) and the merge code lines of
frozen_package/scripts/extract_features.py that explain the duplication.
"""
import pandas as pd

from common import CORR, REV, write

FILES = ["track4_results_hubert-large.csv", "track4_results_wavlm.csv", "track4_results_wav2vec2.csv",
         "track4_results_xlsr.csv", "track4_results_mms.csv"]
EXTRACT = REV / "frozen_package" / "scripts" / "extract_features.py"
DPR = ["back_dprime", "high_dprime", "low_dprime", "manner_dprime", "nasal_dprime", "round_dprime",
       "sonorant_dprime", "strident_dprime", "voicing_dprime"]

out = {"definition": __doc__, "files": {}}
for fn in FILES:
    d = pd.read_csv(CORR / fn)
    d["row"] = d.index + 2
    dup = d[d.duplicated(["dataset", "speaker_id"], keep=False)]
    first = dup.drop_duplicates(["dataset", "speaker_id"], keep="first").set_index(["dataset", "speaker_id"])
    last = dup.drop_duplicates(["dataset", "speaker_id"], keep="last").set_index(["dataset", "speaker_id"])
    cols = [c for c in d.columns if c not in ("row",)]
    equal = [c for c in cols if c not in ("dataset", "speaker_id") and
             ((first[c] == last[c]) | (first[c].isna() & last[c].isna())).all()]
    differ = [c for c in cols if c not in ("dataset", "speaker_id") and c not in equal]
    ssnce_hc = d[(d.dataset == "SSNCE_Tamil") & (d.is_control.astype(str) == "True")]
    out["files"][fn] = {"n_duplicated_keys": int(len(first)), "datasets": sorted(first.index.get_level_values(0).unique()),
                        "first_copy_rows": [int(first.row.min()), int(first.row.max())],
                        "last_copy_rows": [int(last.row.min()), int(last.row.max())],
                        "columns_equal": equal, "columns_differ": differ,
                        "slr65_18_n_phones_sum": int(first.n_phones.sum()),
                        "ssnce_control_n": int(len(ssnce_hc)),
                        "ssnce_control_n_phones_sum": int(ssnce_hc.n_phones.sum())}
code = EXTRACT.read_text(encoding="utf-8").splitlines()
out["extract_features_py"] = {"auto_add_hc_686_692": code[685:692], "tamil_hc_map_68": code[67],
                              "merge_832_842": code[831:842]}
out["dgx_logs"] = {
    "logs/multi_backbone.log": {"sha256": "2a47c55f5ecdf6b4101f3d4c5cce79674df254e6bce6928b0ed0fb21a637fbbc",
                                "run": "2026-04-12, all datasets, wavlm/wav2vec2/xlsr/mms",
                                "lines": ["173:    21710 HC phone tokens from ['SLR65_Tamil', 'SSNCE_Tamil']"]},
    "logs/hu_ta_backbones.log": {"sha256": "b64e7c5c4a0263d4d6c723de5a341aaf337895db61eb33775050c941f99e2481",
                                 "run": "2026-04-14, --dataset SSNCE_Tamil (and Hungarian), all five backbones",
                                 "lines": ["126:  Auto-adding HC dataset 'SLR65_Tamil' for ta feature directions",
                                           "170:    164200 HC phone tokens from ['SLR65_Tamil', 'SSNCE_Tamil']",
                                           "186:  Merged with existing (2174 kept + 48 new)",
                                           "389:    164200 HC phone tokens from ['SLR65_Tamil', 'SSNCE_Tamil']",
                                           "405:  Merged with existing (3162 kept + 48 new)"]}}
f0 = out["files"][FILES[0]]
out["reading"] = (
    f"Only the direction-dependent d-prime columns differ between copies; alignment-derived columns are identical. "
    f"The 18 SLR65 speakers' phone counts sum to {f0['slr65_18_n_phones_sum']} = the 21,710 control tokens of the "
    f"2026-04-12 run (SLR65 only); adding the {f0['ssnce_control_n']} SSNCE controls "
    f"({f0['ssnce_control_n_phones_sum']} tokens) gives the 164,200 tokens of the 2026-04-14 run (SLR65 + SSNCE, as the "
    "Methods describe). The merge step removes existing rows only for the requested dataset, so the re-scored SLR65 rows "
    "were appended and the earlier ones kept. Keep-last therefore selects the Methods-consistent copy. Not established: "
    "why only 18 of the 50 SLR65 speakers were extracted.")
write("v6_tamil_provenance.json", out, [CORR / f for f in FILES] + [EXTRACT])
print(out["reading"])
for fn, v in out["files"].items():
    print(fn, v["n_duplicated_keys"], v["first_copy_rows"], v["last_copy_rows"], v["columns_differ"], v["slr65_18_n_phones_sum"], v["ssnce_control_n"], v["ssnce_control_n_phones_sum"])
print(out["extract_features_py"]["merge_832_842"])
