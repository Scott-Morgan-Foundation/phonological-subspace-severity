"""Tasks 2-4 of the v5 compute pass, on corrected/track4_master.csv (read-only).

2a clinical-only severity-source ablation (logic of frozen_package/scripts/reviewer_experiments.py exp. 1)
2b HC-DS composite gap before/after token-count adjustment (logic of adjust_token_count.py; linear and log variants)
2c CTC-Conf correlations (artp_score column)
3  per-feature reduction from HC (ratio and Cohen's d)
4  Tamil counts
Writes results/v5/unsourced_numbers.json
"""
import csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REV = HERE.parents[2]
MASTER = REV / "corrected" / "track4_master.csv"
INV = REV / "corrected" / "speaker_inventory.csv"
OUT = HERE.parent / "unsourced_numbers.json"

C5 = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
D9 = C5 + ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]
SEV = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
MAIN = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
CLIN = {"TORGO", "UASPEECH", "UASPEECH_control", "SAP", "COPAS", "SSNCE_Tamil", "MDSC", "IPVS",
        "PC-GITA", "Hungarian_Dysarthria", "Neurovoz", "CDLI_Kenyan_Swahili", "YouTube_French",
        "YouTube_German", "CHASING", "TreasureHunters1"}


def f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def comp(r, feats=C5, mn=3):
    v = [f(r[c]) for c in feats]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if len(v) >= mn else None


rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
res = {"master": str(MASTER.relative_to(REV))}

# ---- 2a
lab = [r for r in rows if r["severity_label"] in SEV and r["aetiology"] in MAIN]
clin = [r for r in lab if r["dataset"] in CLIN]
a = {}
for name, sub in (("all_labelled", lab), ("clinical_only", clin)):
    pairs = [(SEV[r["severity_label"]], comp(r)) for r in sub]
    pairs = [p for p in pairs if p[1] is not None]
    s, c = zip(*pairs)
    rho, p = stats.spearmanr(s, c)
    by = defaultdict(list)
    for si, ci in pairs:
        by[si].append(ci)
    g = defaultdict(list)
    for r in sub:
        cc = comp(r)
        if cc is not None:
            g[r["aetiology"]].append(cc)
    valid = {k: v for k, v in g.items() if len(v) >= 5}
    H, pk = stats.kruskal(*valid.values())
    N = sum(len(v) for v in valid.values()); k = len(valid)
    a[name] = {"n": len(pairs), "rho": float(rho), "p": float(p),
               "severity_means": {["control", "mild", "moderate", "severe"][x]: float(np.mean(by[x])) for x in sorted(by)},
               "severity_n": {["control", "mild", "moderate", "severe"][x]: len(by[x]) for x in sorted(by)},
               "kw_H": float(H), "kw_p": float(pk), "eps2": float(max((H - k + 1) / (N - k), 0)), "kw_N": N, "kw_k": k}
a["nonclinical_labelled_datasets"] = dict(Counter(r["dataset"] for r in lab if r["dataset"] not in CLIN))
a["manuscript"] = {"n": 705, "rho": -0.452, "eps2": 0.185, "means": [2.75, 2.08, 1.75, 1.15], "full_sample_eps2": 0.291}
res["2a_severity_source_ablation"] = a

# ---- 2b
def gap(rs, mn):
    hc = [comp(r, mn=mn) for r in rs if r["aetiology"] == "healthy"]
    ds = [comp(r, mn=mn) for r in rs if r["aetiology"] == "down_syndrome"]
    hc = [x for x in hc if x is not None]; ds = [x for x in ds if x is not None]
    return float(np.mean(hc) - np.mean(ds)), len(hc), len(ds)


def adjusted(transform):
    out = [dict(r) for r in rows]
    for feat in D9 + ["vowel_triangle_area"]:
        pairs = [(transform(f(r["n_phones"])), f(r[feat])) for r in rows
                 if f(r["n_phones"]) and f(r["n_phones"]) > 0 and f(r[feat]) is not None]
        x, y = map(np.asarray, zip(*pairs))
        slope = stats.linregress(x, y).slope
        gm = float(np.mean([f(r[feat]) for r in rows if f(r[feat]) is not None]))
        for r in out:
            n, v = f(r["n_phones"]), f(r[feat])
            if n and n > 0 and v is not None:
                r[feat] = str(v - slope * transform(n) + gm)
    return out


b = {}
for mn in (1, 3):
    before = gap(rows, mn)
    b[f"min{mn}"] = {"before": before,
                     "after_linear_nphones": gap(adjusted(lambda n: n), mn),
                     "after_log_nphones": gap(adjusted(lambda n: math.log(n)), mn)}
b["manuscript"] = {"before": 1.63, "after": 1.59, "text_says": "regressing out log(n_phones)",
                   "frozen_script_regresses": "linear n_phones (adjust_token_count.py)"}
res["2b_hc_ds_gap"] = b

# ---- 2c CTC-Conf = artp_score?
c = {"column": "artp_score"}
art = [r for r in rows if f(r.get("artp_score")) is not None]
c["n_speakers_with_value"] = len(art)
c["n_datasets"] = len({r["dataset"] for r in art})
c["n_languages"] = len({r["language"] for r in art})
corr = {}
for col in ("speech_rate", "nasal_dprime", "vowel_duration_cv", "pause_rate", "voicing_dprime",
            "sonorant_dprime", "strident_dprime", "manner_dprime"):
    pr = [(f(r["artp_score"]), f(r[col])) for r in art if f(r[col]) is not None]
    x, y = zip(*pr)
    corr[col] = {"rho": float(stats.spearmanr(x, y)[0]), "n": len(pr)}
c["spearman_with_features_all_speakers"] = corr
for scope, cond in (("main6_is_control0", lambda r: r["aetiology"] in MAIN),
                    ("all_labelled", lambda r: True)):
    pr = []
    for r in art:
        if not cond(r):
            continue
        s = 0 if r["is_control"] == "True" else SEV.get(r["severity_label"])
        if s is None:
            continue
        pr.append((s, f(r["artp_score"])))
    s, v = zip(*pr)
    by = defaultdict(list)
    for si, vi in pr:
        by[si].append(vi)
    c[f"severity_{scope}"] = {"rho": float(stats.spearmanr(s, v)[0]), "n": len(pr),
                              "means": {["control", "mild", "moderate", "severe"][k]: float(np.mean(by[k])) for k in sorted(by)}}
pr = [(SEV[r["severity_label"]], f(r["artp_score"])) for r in art if r["severity_label"] in SEV]
s, v = zip(*pr)
by = defaultdict(list)
for si, vi in pr:
    by[si].append(vi)
c["severity_label_only"] = {"rho": float(stats.spearmanr(s, v)[0]), "n": len(pr),
                            "means": {["control", "mild", "moderate", "severe"][k]: float(np.mean(by[k])) for k in sorted(by)}}
c["manuscript"] = {"speech_rate": 0.501, "nasal": 0.407, "vowel_duration_cv": -0.398, "pause_rate": -0.353,
                   "voicing": 0.298, "severity_rho": -0.126, "severity_n": 1875, "n_speakers": 3229, "n_datasets": 22,
                   "means": [0.920, 0.918, 0.890, 0.811]}
res["2c_ctc_conf"] = c

# ---- 3 reduction from HC
hc = {feat: [f(r[feat]) for r in rows if r["aetiology"] == "healthy" and f(r[feat]) is not None] for feat in C5}
red = {}
for aet in ["parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke", "ALL_DYSARTHRIC"]:
    sel = [r for r in rows if (r["aetiology"] == aet if aet != "ALL_DYSARTHRIC" else r["aetiology"] in MAIN[1:])]
    d = {}
    for feat in C5:
        x = np.asarray([f(r[feat]) for r in sel if f(r[feat]) is not None])
        h = np.asarray(hc[feat])
        sp = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(h) - 1) * h.var(ddof=1)) / (len(x) + len(h) - 2))
        d[feat] = {"ratio_to_HC": float(x.mean() / h.mean()), "cohen_d_HC_minus_aet": float((h.mean() - x.mean()) / sp), "n": len(x)}
    red[aet] = {"per_feature": d,
                "most_reduced_by_ratio": min(d, key=lambda k: d[k]["ratio_to_HC"]),
                "least_reduced_by_ratio": max(d, key=lambda k: d[k]["ratio_to_HC"]),
                "most_reduced_by_d": max(d, key=lambda k: d[k]["cohen_d_HC_minus_aet"]),
                "least_reduced_by_d": min(d, key=lambda k: d[k]["cohen_d_HC_minus_aet"])}
res["3_reduction_from_HC"] = red

# ---- 4 Tamil counts
t = {}
for ds in ("SLR65_Tamil", "SSNCE_Tamil"):
    rr = [r for r in rows if r["dataset"] == ds]
    t[ds] = {"master_rows": len(rr),
             "by_aetiology": dict(Counter(r["aetiology"] for r in rr)),
             "with_any_dprime": sum(1 for r in rr if any(f(r[c]) is not None for c in D9)),
             "with_nasal_dprime": sum(1 for r in rr if f(r["nasal_dprime"]) is not None),
             "composite_min3_valid": sum(1 for r in rr if comp(r) is not None),
             "inventory_rows": sum(1 for r in csv.DictReader(open(INV, encoding="utf-8")) if r["dataset"] == ds)}
t["language_ta_master_rows"] = sum(1 for r in rows if r["language"] == "ta")
t["language_ta_with_any_dprime"] = sum(1 for r in rows if r["language"] == "ta" and any(f(r[c]) is not None for c in D9))
t["master_total_rows"] = len(rows)
t["master_rows_with_any_dprime"] = sum(1 for r in rows if any(f(r[c]) is not None for c in D9))
res["4_tamil_counts"] = t

OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1)[:9000])
