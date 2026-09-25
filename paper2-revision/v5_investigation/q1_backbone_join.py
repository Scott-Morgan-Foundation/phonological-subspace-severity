"""Phase A Q1a-1d: cross-backbone join audit. Read-only on inputs."""
import csv, hashlib, json, os, datetime
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
BB = {"hubert-base": "track4_master.csv", "hubert-large": "track4_results_hubert-large.csv",
      "wavlm": "track4_results_wavlm.csv", "wav2vec2": "track4_results_wav2vec2.csv",
      "xlsr": "track4_results_xlsr.csv", "mms": "track4_results_mms.csv"}
DIRS = {"frozen": BASE / "frozen_package" / "results", "corrected": BASE / "corrected"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fnum(v):
    try:
        f = float(v)
        return f if f == f else np.nan
    except (TypeError, ValueError):
        return np.nan


def read_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def comp(vec, k=3):
    v = vec[~np.isnan(vec)]
    return float(v.mean()) if len(v) >= k else np.nan


def keyed(rows, rule):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["speaker_id"])].append(np.array([fnum(r[c]) for c in FEATS]))
    out = {}
    for k, vs in groups.items():
        if rule == "first":
            out[k] = vs[0]
        elif rule == "last":
            out[k] = vs[-1]
        elif rule == "drop":
            if len(vs) == 1:
                out[k] = vs[0]
        elif rule == "mean":
            out[k] = np.nanmean(np.vstack(vs), axis=0) if len(vs) > 1 else vs[0]
    return out


def pairs_table(data):
    res = {}
    comps = {b: {k: comp(v) for k, v in d.items()} for b, d in data.items()}
    for b1, b2 in combinations(BB, 2):
        common = set(comps[b1]) & set(comps[b2])
        pr = [(comps[b1][k], comps[b2][k]) for k in sorted(common)]
        pr = [(x, y) for x, y in pr if x == x and y == y]
        rho = spearmanr([p[0] for p in pr], [p[1] for p in pr]).statistic
        res[f"{b1}|{b2}"] = {"n": len(pr), "rho": round(float(rho), 4)}
    return res, comps


report = {}
# checksums + mtimes
inv = {}
for dn, d in DIRS.items():
    for b, fn in BB.items():
        p = d / fn
        if p.exists():
            st = os.stat(p)
            inv[f"{dn}/{fn}"] = {"sha256": sha(p), "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"), "bytes": st.st_size}
        else:
            inv[f"{dn}/{fn}"] = "MISSING"
report["inputs"] = inv

rows = {dn: {b: read_rows(d / fn) for b, fn in BB.items() if (d / fn).exists()} for dn, d in DIRS.items()}

# duplicates
dup = {}
for dn in rows:
    for b, rs in rows[dn].items():
        c = Counter((r["dataset"], r["speaker_id"]) for r in rs)
        dk = {k: n for k, n in c.items() if n > 1}
        ident = diff = 0
        by_ds = Counter()
        samples = []
        grp = defaultdict(list)
        for r in rs:
            k = (r["dataset"], r["speaker_id"])
            if k in dk:
                grp[k].append(r)
        for k, rr in grp.items():
            by_ds[k[0]] += 1
            vecs = [tuple(r[c] for c in FEATS) for r in rr]
            if len(set(vecs)) == 1:
                ident += 1
            else:
                diff += 1
            if len(samples) < 4:
                samples.append({"key": list(k), "rows": [{c: r.get(c) for c in ["aetiology", "severity_label", "is_control", "language", "n_phones"] + FEATS if c in r} for r in rr]})
        dup[f"{dn}/{b}"] = {"n_rows": len(rs), "n_unique_keys": len(c), "n_dup_keys": len(dk),
                           "dup_keys_identical_features": ident, "dup_keys_differing_features": diff,
                           "dup_keys_by_dataset": dict(by_ds), "samples": samples}
report["duplicates"] = dup

# pairs under each input set x rule
for dn in rows:
    for rule in ["first", "last", "drop", "mean"]:
        data = {b: keyed(rows[dn][b], rule) for b in BB if b in rows[dn]}
        if len(data) < 6:
            continue
        pt, comps = pairs_table(data)
        report[f"pairs_{dn}_{rule}"] = pt
        if rule == "last":
            report[f"valid_per_backbone_{dn}"] = {b: sum(1 for v in comps[b].values() if v == v) for b in BB}
            # per-dataset included/excluded
            acc = {}
            for b in BB:
                inc, exc = Counter(), Counter()
                for k, v in comps[b].items():
                    (inc if v == v else exc)[k[0]] += 1
                acc[b] = {"included": dict(inc), "excluded_lt3": dict(exc)}
            report[f"per_dataset_accounting_{dn}"] = acc
            if dn == "frozen":
                hb = {k for k, v in comps["hubert-base"].items() if v == v}
                hl = {k for k, v in comps["hubert-large"].items() if v == v}
                miss = hl - hb
                report["frozen_hubertlarge_valid_not_in_hubertbase_valid"] = {"count": len(miss), "by_dataset": dict(Counter(k[0] for k in miss))}
                miss2 = hb - hl
                report["frozen_hubertbase_valid_not_in_hubertlarge_valid"] = {"count": len(miss2), "by_dataset": dict(Counter(k[0] for k in miss2))}
                # key-format evidence for SAP
                ex_hb = sorted(k for k in comps["hubert-base"] if "SAP" in k[0].upper())[:5]
                ex_hl = sorted(k for k in comps["hubert-large"] if "SAP" in k[0].upper())[:5]
                report["frozen_SAP_key_examples"] = {"hubert-base": ex_hb, "hubert-large": ex_hl}
                report["frozen_datasets_hubert-base"] = sorted({k[0] for k in comps["hubert-base"]})
                report["frozen_datasets_hubert-large"] = sorted({k[0] for k in comps["hubert-large"]})

(OUT / "q1_backbone_join.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k.startswith(("pairs_", "valid_", "frozen_hubert", "frozen_SAP"))}, indent=1, default=str)[:9000])
for k, v in dup.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "samples"})
