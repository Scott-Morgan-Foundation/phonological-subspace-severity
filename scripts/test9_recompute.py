#!/usr/bin/env python
"""Test 9.1 — recompute every master-derivable displayed number (ANALYSIS_PLAN §10).

Each block prints: submitted value -> recomputed value -> PASS/FAIL (tolerance in
brackets). Numbers anchored only in run logs (fixed-token, layer ablation) are
checked separately against the frozen logs. Seed 42 where sampling is involved.
Output: results/test9_recompute.json (+ console report).
"""
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MASTER = BASE / "frozen_package" / "results" / "track4_master.csv"
OUT = BASE / "results" / "test9_recompute.json"

CONS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime",
        "manner_dprime"]
FEATS13 = CONS + ["high_dprime", "low_dprime", "back_dprime", "round_dprime",
                  "vowel_triangle_area", "speech_rate", "pause_rate",
                  "vowel_duration_cv"]
FEATS9 = CONS + ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]
G6 = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
      "down_syndrome": "DS", "stroke": "Stroke"}
EXEC = ("CP", "DS", "Stroke")

rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))


def f(v):
    try:
        x = float(v)
        return x if x == x else np.nan
    except (TypeError, ValueError):
        return np.nan


def comp(r, feats=CONS):
    v = np.array([f(r[c]) for c in feats])
    v = v[~np.isnan(v)]
    return v.mean() if len(v) else np.nan


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                 (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def check(name, submitted, recomputed, tol):
    ok = abs(recomputed - submitted) <= tol
    report.append({"item": name, "submitted": submitted,
                   "recomputed": round(float(recomputed), 4), "tol": tol,
                   "verdict": "PASS" if ok else "FAIL"})
    print(f"{'PASS' if ok else 'FAIL':4s} {name}: submitted {submitted} -> {recomputed:.4f}")


report = []
out = {}

# ---- counts -----------------------------------------------------------------
n_spk = len(rows)
n_ds = len({r["dataset"] for r in rows})
n_lang = len({r["language"] for r in rows})
check("n speakers", 3374, n_spk, 0)
check("n datasets", 25, n_ds, 0)
check("n languages", 12, n_lang, 0)

# ---- 4.1: KW eps2 per 13 features, count > 0.14 ------------------------------
grp = defaultdict(list)
for r in rows:
    g = G6.get(r["aetiology"])
    if g:
        grp[g].append(r)
eps2 = {}
for ft in FEATS13:
    gl = []
    for g in ("HC", "PD", "CP", "ALS", "DS", "Stroke"):
        v = np.array([f(r[ft]) for r in grp[g]])
        v = v[~np.isnan(v)]
        if len(v) >= 3:
            gl.append(v)
    H, p = kruskal(*gl)
    n = sum(len(g) for g in gl)
    eps2[ft] = (H - len(gl) + 1) / (n - len(gl))
n_large = sum(1 for v in eps2.values() if v > 0.14)
check("features with eps2 > 0.14 (of 13)", 10, n_large, 0)
out["eps2_13"] = {k: round(float(v), 4) for k, v in eps2.items()}

# ---- 4.1: Holm pairwise (15 pairs) on composite ------------------------------
cvals = {g: np.array([x for x in (comp(r) for r in grp[g]) if x == x])
         for g in ("HC", "PD", "CP", "ALS", "DS", "Stroke")}
pairs = list(combinations(["HC", "PD", "CP", "ALS", "DS", "Stroke"], 2))
pv = []
for a, b in pairs:
    _, p = mannwhitneyu(cvals[a], cvals[b])
    pv.append(p)
order = np.argsort(pv)
sig = [False] * 15
for rank, idx in enumerate(order):
    if pv[idx] * (15 - rank) < 0.05:
        sig[idx] = True
    else:
        break
check("Holm-significant pairs (of 15)", 12, sum(sig), 1)
out["holm_pairs"] = {f"{a}|{b}": {"p": float(p), "sig": bool(s)}
                     for (a, b), p, s in zip(pairs, pv, sig)}

# ---- 4.1: PD vs execution cluster d ------------------------------------------
ex = np.concatenate([cvals[g] for g in EXEC])
check("Cohen's d PD vs execution cluster", 0.83, abs(cohens_d(cvals["PD"], ex)), 0.03)
# PD pairwise d range 0.62-1.01
ds_ = [abs(cohens_d(cvals["PD"], cvals[g])) for g in ("HC", "CP", "ALS", "DS", "Stroke")]
out["pd_pairwise_d"] = [round(float(x), 3) for x in ds_]
print("     PD pairwise |d| vs HC/CP/ALS/DS/Stroke:", out["pd_pairwise_d"],
      "(submitted range 0.62-1.01)")

# ---- 4.1: severity-matched moderates: CP-vs-stroke-cluster homogeneity d=0.15?
mod = defaultdict(list)
for r in rows:
    g = G6.get(r["aetiology"])
    if g and g != "HC" and r["severity_label"] == "moderate":
        c = comp(r)
        if c == c:
            mod[g].append(c)
if len(mod.get("PD", [])) > 2 and all(len(mod.get(g, [])) > 0 for g in EXEC):
    exm = np.concatenate([mod[g] for g in EXEC if mod.get(g)])
    dmod = abs(cohens_d(np.array(mod["PD"]), exm))
    out["severity_matched_moderate_PD_vs_exec_d"] = round(float(dmod), 3)
    print(f"     severity-matched (moderate) PD vs exec d = {dmod:.3f} "
          f"(submitted text: clustering persists, d = 0.15 for ALS-vs-CP context)")

# ---- severity correlations ----------------------------------------------------
sev_map = {"mild": 1, "moderate": 2, "severe": 3}
xs, ys = [], []
for r in rows:
    if r["aetiology"] in G6 and r["aetiology"] != "healthy":
        s = sev_map.get(r["severity_label"])
        c = comp(r)
        if s and c == c:
            xs.append(s)
            ys.append(c)
rho = spearmanr(xs, ys).statistic
check("severity Spearman rho (labelled dys, composite)", -0.543, rho, 0.02)
# SAP excluded
xs2, ys2 = [], []
for r in rows:
    if r["dataset"].startswith("SAP"):
        continue
    if r["aetiology"] in G6 and r["aetiology"] != "healthy":
        s = sev_map.get(r["severity_label"])
        c = comp(r)
        if s and c == c:
            xs2.append(s)
            ys2.append(c)
check("severity rho SAP-excluded", -0.410, spearmanr(xs2, ys2).statistic, 0.02)

# ---- 5.2 quartile correlations -------------------------------------------------
np_all = [(f(r["n_phones"]), sev_map.get(r["severity_label"]), comp(r))
          for r in rows if r["aetiology"] in G6 and r["aetiology"] != "healthy"]
np_all = [(n, s, c) for n, s, c in np_all if n == n and s and c == c]
qs = np.percentile([n for n, _, _ in np_all], [25, 50, 75])
subQ = {1: [], 2: [], 3: [], 4: []}
for n, s, c in np_all:
    q = 1 + int(n > qs[0]) + int(n > qs[1]) + int(n > qs[2])
    subQ[q].append((s, c))
for q, target in ((1, -0.14), (2, -0.18), (3, -0.39), (4, -0.71)):
    r_ = spearmanr([s for s, _ in subQ[q]], [c for _, c in subQ[q]]).statistic
    check(f"quartile Q{q} rho", target, r_, 0.04)

# ---- Table 4 cosines: 5-feat and 9-feat, cells >=3 -----------------------------
def cell_cos(feats):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = G6.get(r["aetiology"])
        if g:
            by[g][r["language"]].append(r)
    res = {}
    for g in ("PD", "CP", "ALS", "HC"):
        langs = [l for l, rs in by[g].items() if len(rs) >= 3]
        profs = {}
        for l in langs:
            m = []
            for ft in feats:
                v = np.array([f(r[ft]) for r in by[g][l]])
                v = v[~np.isnan(v)]
                m.append(v.mean() if len(v) else np.nan)
            profs[l] = np.array(m)
        sims = []
        for l1, l2 in combinations(sorted(langs), 2):
            a, b = profs[l1], profs[l2]
            mk = ~(np.isnan(a) | np.isnan(b))
            if mk.sum() >= 3:
                sims.append(float(a[mk] @ b[mk] /
                                  (np.linalg.norm(a[mk]) * np.linalg.norm(b[mk]))))
        if sims:
            res[g] = {"mean": round(float(np.mean(sims)), 4),
                      "min": round(float(np.min(sims)), 4),
                      "max": round(float(np.max(sims)), 4),
                      "n_langs": len(langs)}
    return res

t4_5 = cell_cos(CONS)
t4_9 = cell_cos(FEATS9)
out["table4_5feat"] = t4_5
out["table4_9feat"] = t4_9
check("Table4 PD mean cosine (9-feat hypothesis)", 0.973, t4_9["PD"]["mean"], 0.002)
check("Table4 CP mean cosine (9-feat)", 0.987, t4_9["CP"]["mean"], 0.002)
check("Table4 ALS mean cosine (9-feat)", 0.979, t4_9["ALS"]["mean"], 0.002)
check("Table4 HC mean cosine (9-feat)", 0.958, t4_9["HC"]["mean"], 0.002)
check("Contribution(b) PD cosine (5-feat)", 0.979, t4_5["PD"]["mean"], 0.002)
check("Contribution(b) CP cosine (5-feat)", 0.987, t4_5["CP"]["mean"], 0.002)
check("Contribution(b) ALS cosine (5-feat)", 0.985, t4_5["ALS"]["mean"], 0.002)
check("Contribution(b) HC cosine (5-feat)", 0.979, t4_5["HC"]["mean"], 0.002)

# ---- X-3: Swahili anecdote traces ----------------------------------------------
def prof(rs, feats):
    m = []
    for ft in feats:
        v = np.array([f(r[ft]) for r in rs])
        v = v[~np.isnan(v)]
        m.append(v.mean() if len(v) else np.nan)
    return np.array(m)


def coss(a, b):
    mk = ~(np.isnan(a) | np.isnan(b))
    return float(a[mk] @ b[mk] / (np.linalg.norm(a[mk]) * np.linalg.norm(b[mk])))


sw_pd = [r for r in rows if r["language"] == "sw" and r["aetiology"] == "parkinsons"]
sk_pd = [r for r in rows if r["language"] == "sk" and r["aetiology"] == "parkinsons"]
sw_cp = [r for r in rows if r["language"] == "sw" and r["aetiology"] == "cerebral_palsy"]
zh_cp = [r for r in rows if r["language"] == "zh" and r["aetiology"] == "cerebral_palsy"]
pd_multi = [r for r in rows if r["aetiology"] == "parkinsons" and r["language"] != "sw"]
x3 = {}
for feats, tag in ((CONS, "5feat"), (FEATS9, "9feat")):
    x3[f"swPD_vs_skPD_{tag}"] = round(coss(prof(sw_pd, feats), prof(sk_pd, feats)), 4)
    x3[f"swPD_vs_multilangPD_{tag}"] = round(coss(prof(sw_pd, feats),
                                                  prof(pd_multi, feats)), 4)
    x3[f"swCP_vs_zhCP_{tag}"] = round(coss(prof(sw_cp, feats), prof(zh_cp, feats)), 4)
out["x3_swahili_traces"] = x3
print("     X-3 traces:", json.dumps(x3))

# ---- HC>=5 sensitivity: PD cosine excluding sw ---------------------------------
# (paper: PD 0.979 remains, CP 0.984 3 langs, ALS 0.985 at HC>=5)
hc_by_lang = defaultdict(int)
for r in rows:
    if r["aetiology"] == "healthy":
        hc_by_lang[r["language"]] += 1
ok_langs = {l for l, n in hc_by_lang.items() if n >= 5}
def cell_cos_hc5(feats, g_target):
    by = defaultdict(list)
    for r in rows:
        if G6.get(r["aetiology"]) == g_target and r["language"] in ok_langs:
            by[r["language"]].append(r)
    langs = [l for l, rs in by.items() if len(rs) >= 3]
    sims = []
    for l1, l2 in combinations(sorted(langs), 2):
        sims.append(coss(prof(by[l1], feats), prof(by[l2], feats)))
    return (round(float(np.mean(sims)), 4), len(langs)) if sims else (np.nan, len(langs))
for g_t, target in (("PD", 0.979), ("CP", 0.984), ("ALS", 0.985)):
    val, nl = cell_cos_hc5(CONS, g_t)
    check(f"HC>=5 sensitivity {g_t} cosine (5-feat)", target, val, 0.003)

# ---- min n>=10 PD cosine (conclusion: 0.976) ------------------------------------
by = defaultdict(list)
for r in rows:
    if r["aetiology"] == "parkinsons":
        by[r["language"]].append(r)
langs10 = [l for l, rs in by.items() if len(rs) >= 10]
sims = [coss(prof(by[l1], CONS), prof(by[l2], CONS))
        for l1, l2 in combinations(sorted(langs10), 2)]
check("PD cosine with cell n>=10 (5-feat)", 0.976, float(np.mean(sims)), 0.003)

# ---- severe CP retention (4.2: en/zh 18-29%, sw 55-74%) -------------------------
def retention(lang):
    hc = [r for r in rows if r["language"] == lang and r["aetiology"] == "healthy"]
    sev = [r for r in rows if r["language"] == lang and
           r["aetiology"] == "cerebral_palsy" and r["severity_label"] == "severe"]
    if not hc or not sev:
        return None
    ph, ps = prof(hc, CONS), prof(sev, CONS)
    ratio = ps / np.where(ph == 0, 1, ph)
    return [round(float(x), 3) for x in ratio]
out["severe_cp_retention"] = {l: retention(l) for l in ("en", "zh", "sw")}
print("     severe CP retention en/zh/sw:", json.dumps(out["severe_cp_retention"]))

out["checks"] = report
n_fail = sum(1 for c in report if c["verdict"] == "FAIL")
print(f"\n==== {len(report)} checks, {n_fail} FAIL ====")
OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("written", OUT)
