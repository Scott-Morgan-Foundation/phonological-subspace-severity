#!/usr/bin/env python
"""Test 4 local analysis — consumes test4_speaker_dprime.csv from the DGX rebuild.

Per language and variant:
  1. FAITHFULNESS (pooled variant only): Spearman + median |diff| of per-speaker
     d' vs the frozen master (cache-vs-original check; if this fails for a language,
     its variant comparisons are still internally valid but not anchored to the paper).
  2. WEIGHTING EFFECT: Spearman of composite d' (mean of the 5 consonant d')
     variant vs pooled, on the same cached speakers.
  3. GROUP-LEVEL STABILITY (en only, cached subsample, same speakers across variants):
     Kruskal-Wallis epsilon-squared across HC/PD/CP/ALS/DS/Stroke on composite;
     Cohen's d PD vs execution cluster (CP+DS+Stroke).
Decision rule 4 (§11): if a reasonable weighting choice materially changes a headline
result (sign flip / boundary crossed), report the dependence and narrow interpretation.

Output: results/test4_rebuilds_report.md + results/test4_rebuilds_summary.json
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kruskal

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MASTER = BASE / "frozen_package" / "results" / "track4_master.csv"
REBUILT = BASE / "results" / "test4_speaker_dprime.csv"
OUT_MD = BASE / "results" / "test4_rebuilds_report.md"
OUT_JSON = BASE / "results" / "test4_rebuilds_summary.json"

FEATS = ["nasal", "voicing", "sonorant", "strident", "manner"]
AET6 = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
        "down_syndrome": "DS", "stroke": "Stroke"}
EXEC = {"CP", "DS", "Stroke"}


def safe(s):
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")


def fnum(v):
    try:
        f = float(v)
        return f if f == f else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    master = {}
    meta = {}
    with open(MASTER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["dataset"], safe(r["speaker_id"]))
            master[k] = np.array([fnum(r[ft + "_dprime"]) for ft in FEATS])
            meta[k] = AET6.get(r["aetiology"])

    reb = defaultdict(dict)   # (lang, variant) -> {key: vec}
    with open(REBUILT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["dataset"], r["speaker_id"])
            reb[(r["language"], r["variant"])][k] = np.array(
                [fnum(r[ft]) for ft in FEATS])

    out = {}
    lines = ["# Test 4 — HC-direction rebuild analysis", ""]

    langs = sorted({l for (l, v) in reb})
    for lang in langs:
        variants = sorted({v for (l, v) in reb if l == lang})
        res = {}
        # 1. faithfulness of pooled vs master
        pooled = reb.get((lang, "pooled"), {})
        both = [k for k in pooled if k in master]
        pv, mv = [], []
        for k in both:
            for i in range(5):
                a, b = pooled[k][i], master[k][i]
                if a == a and b == b:
                    pv.append(a); mv.append(b)
        if len(pv) > 20:
            rho = float(spearmanr(pv, mv).statistic)
            mad = float(np.median(np.abs(np.array(pv) - np.array(mv))))
            res["faithfulness"] = {"n_values": len(pv), "spearman": round(rho, 4),
                                   "median_abs_diff": round(mad, 4)}
        # 2. composite Spearman variant vs pooled
        def comp(vec):
            v = vec[~np.isnan(vec)]
            return float(v.mean()) if len(v) else np.nan
        res["variant_vs_pooled_spearman"] = {}
        for v in variants:
            if v == "pooled":
                continue
            ks = [k for k in reb[(lang, v)] if k in pooled]
            a = [comp(reb[(lang, v)][k]) for k in ks]
            b = [comp(pooled[k]) for k in ks]
            pair = [(x, y) for x, y in zip(a, b) if x == x and y == y]
            if len(pair) > 10:
                res["variant_vs_pooled_spearman"][v] = round(
                    float(spearmanr([p[0] for p in pair], [p[1] for p in pair]).statistic), 4)
        # 3. group stats per variant (en only)
        if lang == "en":
            res["group_stats"] = {}
            for v in variants:
                groups = defaultdict(list)
                for k, vec in reb[(lang, v)].items():
                    a = meta.get(k)
                    c = comp(vec)
                    if a and c == c:
                        groups[a].append(c)
                gl = [np.array(groups[g]) for g in sorted(groups) if len(groups[g]) >= 3]
                names = [g for g in sorted(groups) if len(groups[g]) >= 3]
                if len(gl) >= 3:
                    H, p = kruskal(*gl)
                    n = sum(len(g) for g in gl)
                    eps2 = float((H - len(gl) + 1) / (n - len(gl)))
                    pd_ = np.array(groups.get("PD", []))
                    ex = np.concatenate([np.array(groups.get(g, [])) for g in EXEC]) \
                        if any(g in groups for g in EXEC) else np.array([])
                    d = np.nan
                    if len(pd_) > 2 and len(ex) > 2:
                        sp = np.sqrt(((len(pd_) - 1) * pd_.var(ddof=1) +
                                      (len(ex) - 1) * ex.var(ddof=1)) /
                                     (len(pd_) + len(ex) - 2))
                        d = float((pd_.mean() - ex.mean()) / sp) if sp > 1e-12 else np.nan
                    res["group_stats"][v] = {
                        "groups": {g: len(groups[g]) for g in names},
                        "kw_eps2": round(eps2, 4), "kw_p": float(p),
                        "cohen_d_PD_vs_exec": round(d, 4) if d == d else None}
        out[lang] = res

    lines.append("Per-language results (cached subsample; see JSON for full detail):\n")
    for lang, res in out.items():
        lines.append(f"## {lang}")
        f = res.get("faithfulness")
        if f:
            lines.append(f"- faithfulness (pooled vs master, per-speaker-feature values): "
                         f"Spearman {f['spearman']}, median |diff| {f['median_abs_diff']} "
                         f"(n={f['n_values']})")
        vs = res.get("variant_vs_pooled_spearman", {})
        if vs:
            lines.append("- composite-d' Spearman vs pooled: " +
                         ", ".join(f"{k} {v}" for k, v in sorted(vs.items())))
        gs = res.get("group_stats")
        if gs:
            lines.append("- en group stats per variant (KW eps2 / PD-vs-exec d):")
            for v, g in sorted(gs.items()):
                lines.append(f"    {v}: eps2 {g['kw_eps2']}, d {g['cohen_d_PD_vs_exec']}, "
                             f"groups {g['groups']}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
