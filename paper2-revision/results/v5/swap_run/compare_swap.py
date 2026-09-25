"""Compare manuscript quantities on original vs Tamil-swapped HuBERT-base master.

Reads original/ and swapped/ sandboxes built by build_swap.py (their results/
were produced by the unmodified scripts/corrected/*.py). Adds direct
computations for quantities without a saved generating script.
Writes results/v5/hubert_tamil_swap_impact.json and .md
"""
import csv, importlib.util, json, math, sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REV = HERE.parents[2]
OUTJ = HERE.parent / "hubert_tamil_swap_impact.json"
OUTM = HERE.parent / "hubert_tamil_swap_impact.md"

spec = importlib.util.spec_from_file_location("rp1", REV / "scripts" / "rebuild_phase1.py")
rp1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp1)

C5 = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
AETMAP = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
          "down_syndrome": "DS", "stroke": "Stroke"}
SEV = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
SEED = 42


def f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def comp(r, feats=C5, mn=3):
    v = [f(r[c]) for c in feats]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if len(v) >= mn else None


def cohen(x, y):
    x, y = np.asarray(x), np.asarray(y)
    s = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return (x.mean() - y.mean()) / s


def pairwise(rows):
    g = {a: [] for a in AETMAP.values()}
    for r in rows:
        a = AETMAP.get(r["aetiology"])
        c = comp(r)
        if a and c is not None:
            g[a].append(c)
    out = {}
    order = ["HC", "PD", "CP", "ALS", "DS", "Stroke"]
    for a, b in combinations(order, 2):
        out[f"{a}-{b}"] = abs(cohen(g[a], g[b]))
    out["mean_PD_exec"] = float(np.mean([out["PD-CP"], out["PD-DS"], out["PD-Stroke"]]))
    out["n"] = {a: len(v) for a, v in g.items()}
    return out


def moderate(rows):
    res = {}
    for label, mn in (("all5", 5), ("min3", 3)):
        pd, ex = [], []
        for r in rows:
            if r["severity_label"] != "moderate":
                continue
            a = AETMAP.get(r["aetiology"])
            c = comp(r, mn=mn)
            if c is None:
                continue
            if a == "PD":
                pd.append(c)
            elif a in ("CP", "DS", "Stroke"):
                ex.append(c)
        res[label] = {"d": float(cohen(pd, ex)), "n_pd": len(pd), "n_exec": len(ex)}
    return res


def severity(rows, n_boot=1000):
    keep = []
    for r in rows:
        if r["aetiology"] not in AETMAP:
            continue
        s = 0 if r["is_control"] == "True" else SEV.get(r["severity_label"])
        if s is None:
            continue
        keep.append((r, s))
    out = {}
    feats = {"composite": None, **{c: c for c in C5}}
    for name, col in feats.items():
        xs, ys, tok = [], [], []
        for r, s in keep:
            v = comp(r) if col is None else f(r[col])
            if v is None:
                continue
            if col is None:
                pass
            xs.append(v); ys.append(s); tok.append(float(r["n_phones"] or 0))
        xs, ys, tok = map(np.asarray, (xs, ys, tok))
        rho = spearmanr(xs, ys)[0]
        q = np.quantile(tok, [0.25, 0.5, 0.75])
        strata = np.digitize(tok, q)
        idx_by = [np.where(strata == k)[0] for k in range(4)]
        rng = np.random.default_rng(SEED)
        bs = []
        for _ in range(n_boot):
            ii = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by if len(ix)])
            bs.append(spearmanr(xs[ii], ys[ii])[0])
        out[name] = {"rho": float(rho), "n": int(len(xs)),
                     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def keylast(p, mn=3):
    d = {}
    for r in load(p):
        c = comp(r, mn=mn)
        d[(r["dataset"], r["speaker_id"])] = c
    return {k: v for k, v in d.items() if v is not None}


def crossbackbone(sandbox):
    files = {"hubert-base": "track4_master.csv", "hubert-large": "track4_results_hubert-large.csv",
             "wavlm": "track4_results_wavlm.csv", "wav2vec2": "track4_results_wav2vec2.csv",
             "xlsr": "track4_results_xlsr.csv", "mms": "track4_results_mms.csv"}
    comps = {b: keylast(sandbox / "corrected" / fn) for b, fn in files.items()}
    out = {}
    for a, b in combinations(files, 2):
        ks = sorted(set(comps[a]) & set(comps[b]))
        out[f"{a}|{b}"] = {"rho": float(spearmanr([comps[a][k] for k in ks], [comps[b][k] for k in ks])[0]),
                           "n": len(ks)}
    return out


def J(p):
    return json.load(open(p, encoding="utf-8"))


def collect(name):
    sb = HERE / name
    rows = load(sb / "corrected" / "track4_master.csv")
    res = {}
    t3 = rp1.job_table3(rows)
    res["table3"] = {r["feature"]: {"H": r["H"], "p": r["p"], "eps2": r["epsilon_squared"]}
                     for r in t3["rows"] if "H" in r}
    res["pairwise_d"] = pairwise(rows)
    res["moderate_only"] = moderate(rows)
    res["severity"] = severity(rows)
    t2 = J(sb / "results" / "test2_delta_corrected.json")
    res["test2"] = {"delta": t2["primary_hcnorm"]["delta"], "ci95": t2["primary_hcnorm"]["ci95"],
                    "p_block_perm": t2["perm_lang_dataset_hcnorm"]["p_null_ge_obs"],
                    "centred": t2["sensitivities"].get("centred", t2["sensitivities"].get("centered")),
                    "repro_feats5": t2["repro_submitted_permutation"]["feats5"]}
    t9 = J(sb / "results" / "test9_recompute_corrected.json")
    res["test9"] = {k: t9[k] for k in ("eps2_13", "holm_pairs", "pd_pairwise_d",
                                        "severity_matched_moderate_PD_vs_exec_d", "table4_5feat",
                                        "table4_9feat", "severe_cp_retention")}
    res["test6"] = J(sb / "results" / "test6_backbones_corrected.json")
    res["test7"] = J(sb / "results" / "test7_lodo_corrected.json")
    res["crossbackbone_keylast_min3"] = crossbackbone(sb)
    return res


def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flat(v, f"{p}/{k}" if p else str(k)))
    elif isinstance(d, (list, tuple)) and all(isinstance(x, (int, float)) for x in d):
        for i, x in enumerate(d):
            o[f"{p}[{i}]"] = x
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        o[p] = d
    return o


orig, swap = collect("original"), collect("swapped")
fo, fs = flat(orig), flat(swap)
diffs = []
for k in sorted(fo):
    a, b = fo[k], fs.get(k)
    if b is None:
        continue
    if a != b:
        diffs.append({"key": k, "original": a, "swapped": b,
                      "changes_at_3dp": round(a, 3) != round(b, 3) if abs(a) < 1e3 else round(a) != round(b)})
json.dump({"original": orig, "swapped": swap, "diffs": diffs,
           "n_numeric_compared": len(fo), "n_changed_any": len(diffs),
           "n_changed_at_3dp": sum(d["changes_at_3dp"] for d in diffs)},
          open(OUTJ, "w", encoding="utf-8"), indent=1, default=float)

lines = ["# HuBERT-base Tamil swap impact", "",
         f"Numeric quantities compared: {len(fo)}; changed at any precision: {len(diffs)}; "
         f"changed at 3 dp: {sum(d['changes_at_3dp'] for d in diffs)}", "",
         "| quantity | original | swapped | changes at 3 dp |", "|---|---|---|---|"]
for d in diffs:
    if d["changes_at_3dp"] or any(s in d["key"] for s in ("hubert-base", "severity/composite", "test2/", "pairwise_d", "moderate")):
        lines.append(f"| {d['key']} | {d['original']:.6g} | {d['swapped']:.6g} | {'YES' if d['changes_at_3dp'] else 'no'} |")
OUTM.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:6]))
print("changed at 3dp:")
for d in diffs:
    if d["changes_at_3dp"]:
        print(" ", d["key"], round(d["original"], 4), "->", round(d["swapped"], 4))
