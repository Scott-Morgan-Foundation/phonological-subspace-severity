#!/usr/bin/env python3
"""v9 number registry and single-rounding formatter.

Extends scripts/v8/format_numbers.py (imported unchanged; its registry, rounding rules and loaders are
reused). v9 changes:
  * removes the v8 entries whose source was a re-implementation that v9 replaces with an ORIGINAL script
    re-run unchanged on the corrected inputs (minimum-n cross-lingual CIs from v6_crossling; Table 6 from
    v6_fixed_token_dgx);
  * adds entries sourced from results/v9/v9_originals.json (robustness_analyses.py Analysis 2 for
    contribution (b); reviewer_experiments.py Experiment 2 and 3 for the minimum-n cosines and the Holm
    post hoc; fixed_token_dprime.py Experiment A for Table 6; robustness_pass5.py Experiments 1 and 2 for
    the common-speaker fixed-token set and the minimum-HC sensitivity);
  * binds every number changed in v9 to its sentence context (ctx), so check_consistency verifies the
    number in place rather than merely locating it somewhere in the outputs.
Every value is rounded ONCE from the saved output. The original scripts print 3-4 decimals; the printed
value is the original's own output at the precision it printed (rule 3dp/4dp applied to that value).
"""
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("format_numbers_v8",
                                               Path(__file__).resolve().parents[1] / "v8" / "format_numbers.py")
V8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V8)

REV = V8.REV
fmt, get, load = V8.fmt, V8.get, V8.load
V9 = "results/v9/v9_originals.json"
C = ["states", "corrected", "parsed"]

DROP = {"mn3_CP_lo_3dp", "mn3_CP_hi_3dp", "mn10_CP_lo_3dp", "mn10_CP_hi_3dp"} | \
       {f"t6row_{b}" for b in ("20", "50", "100", "200")} | {f"t6eps_{b}" for b in ("20", "50", "100", "200")}
E = [e for e in V8.E if e["id"] not in DROP]


def add(id_, path, rule, ctx=(), src=V9, note=""):
    E.append({"id": id_, "src": src, "path": list(path), "rule": rule, "ctx": list(ctx), "note": note})


# ---- contribution (b): robustness_analyses.py Analysis 2 --------------------------------------
A2 = C + ["robustness_analyses", "a2_crosslingual_bootstrap"]
for g in ("PD", "CP", "ALS", "HC"):
    add(f"b_{g}_mean", A2 + [g, "mean_cosine"], "3dp")
    add(f"b_{g}_lo", A2 + [g, "mean_of_pair_ci_lower"], "3dp")
    add(f"b_{g}_hi", A2 + [g, "mean_of_pair_ci_upper"], "3dp")
add("b_PD_tight_cos", A2 + ["PD", "tightest_pair", "cos"], "3dp")
add("b_PD_tight_lo", A2 + ["PD", "tightest_pair", "ci95", 0], "3dp")
for g, lab in (("PD", "PD"), ("CP", "CP"), ("ALS", "ALS"), ("HC", "HC")):
    pass

# ---- Section 4.5 minimum-n: reviewer_experiments.py Experiment 2 -----------------------------
MN = C + ["reviewer_experiments", "exp2_min_n_language_pair_bootstrap"]
for t in ("3", "5", "10"):
    for g in ("PD", "CP", "HC"):
        add(f"mn{t}_{g}_mean", MN + [t, g, "mean"], "3dp")
        add(f"mn{t}_{g}_lo", MN + [t, g, "ci95", 0], "3dp")
        add(f"mn{t}_{g}_hi", MN + [t, g, "ci95", 1], "3dp")
add("mn3_PD_min", MN + ["3", "PD", "min"], "3dp")

# ---- Holm post hoc: reviewer_experiments.py Experiment 3 --------------------------------------
H = C + ["reviewer_experiments", "exp3_holm"]
add("holm_n_HC", H + ["group_sizes", "HC"], "int")
add("holm_n_PD", H + ["group_sizes", "PD"], "int")


def _holm_idx(pair):
    pairs = get(load(V9), H + ["pairs"])
    return [i for i, p in enumerate(pairs) if p["pair"] == pair][0]


def _add_holm():
    for pair, key in (("HC vs PD", "hcpd"), ("CP vs PD", "cppd"), ("ALS vs PD", "alspd"), ("DS vs PD", "dspd"),
                      ("PD vs Stroke", "pdst"), ("CP vs Stroke", "cpst"), ("CP vs DS", "cpds"), ("DS vs Stroke", "dsst")):
        i = _holm_idx(pair)
        add(f"holm_{key}_d_2dp", H + ["pairs", i, "cohen_d_first_minus_second"], "2dp")
        add(f"holm_{key}_r_2dp", H + ["pairs", i, "r_rb"], "2dp")
        add(f"holm_{key}_r_3dp", H + ["pairs", i, "r_rb"], "3dp")
        add(f"holm_{key}_p", H + ["pairs", i, "p_holm"], "2dp")


if (REV / V9).exists():
    _add_holm()

# ---- Table 6 and the common-speaker set: fixed_token_dprime.py A, robustness_pass5.py 1 --------
FT = C + ["fixed_token_dprime", "expA_per_budget"]
P5 = C + ["robustness_pass5"]
for b in ("20", "50", "100", "200"):
    add(f"t6row_{b}", FT + [b], "t6row", ctx=[b + "\n{v}"])
    add(f"t6eps_{b}", FT + [b, "aet_eps2"], "3dp")
    add(f"t6rho_{b}_3dp", FT + [b, "rho"], "3dp")
    add(f"t6n_{b}", FT + [b, "n"], "int")
    add(f"cs_rho_{b}_3dp", P5 + ["exp1_common_set", "per_budget", b, "rho"], "3dp")
add("cs_n", P5 + ["exp1_common_set", "n_qualifying"], "int")
add("t6eps_min_2dp", C + ["fixed_token_dprime", "derived", "eps2_min"], "2dp")
add("t6eps_max_2dp", C + ["fixed_token_dprime", "derived", "eps2_max"], "2dp")
add("cs_rho_min_3dp", P5 + ["exp1_common_set", "derived", "rho_min"], "3dp")
add("cs_rho_max_3dp", P5 + ["exp1_common_set", "derived", "rho_max"], "3dp")
add("t6_200_p_bound", FT + ["200", "p"], "lt10:-79", note="p of the 200-token budget below the printed bound")
add("cs_p_bound", P5 + ["exp1_common_set", "derived", "p_max"], "lt10:-79", note="all common-set p below the printed bound")
for thr in ("1", "10", "20"):
    for g in ("PD", "CP", "ALS"):
        add(f"mhc{thr}_{g}_cos", P5 + ["exp2_min_hc", thr, g, "cos"], "3dp")
        add(f"mhc{thr}_{g}_nl", P5 + ["exp2_min_hc", thr, g, "n_languages"], "int")
        add(f"mhc{thr}_{g}_lo", P5 + ["exp2_min_hc", thr, g, "ci95", 0], "3dp")
        add(f"mhc{thr}_{g}_hi", P5 + ["exp2_min_hc", thr, g, "ci95", 1], "3dp")
for thr in ("5",):
    for g in ("PD", "CP", "ALS"):
        add(f"mhc{thr}_{g}_cos", P5 + ["exp2_min_hc", thr, g, "cos"], "3dp")
        add(f"mhc{thr}_{g}_lo", P5 + ["exp2_min_hc", thr, g, "ci95", 0], "3dp")
        add(f"mhc{thr}_{g}_hi", P5 + ["exp2_min_hc", thr, g, "ci95", 1], "3dp")
        add(f"mhc{thr}_{g}_nl", P5 + ["exp2_min_hc", thr, g, "n_languages"], "int")


for s, pth in (("mean", ["mean"]), ("lo", ["ci95", 0]), ("hi", ["ci95", 1])):
    add(f"mn1_CP_{s}", MN + ["1", "CP"] + pth, "3dp")
add("mn5_ALS_nl", MN + ["5", "ALS", "n_languages"], "int")
add("mn5_ALS_np", MN + ["5", "ALS", "n_pairs"], "int")


# ---- sentence bindings for every number changed in v9 (filled from the formatted values) -------
BIND = [
    ("gives PD {b_PD_mean} {{[}}{b_PD_lo}, {b_PD_hi}{{]}}, CP {b_CP_mean} {{[}}{b_CP_lo}, {b_CP_hi}{{]}}, "
     "ALS {b_ALS_mean} {{[}}{b_ALS_lo}, {b_ALS_hi}{{]}} and HC {b_HC_mean} {{[}}{b_HC_lo}, {b_HC_hi}{{]}}"),
    ("The lowest PD language-pair cosine is {mn3_PD_min}, and the pair with the lowest bootstrap bound, "
     "Dutch--Portuguese PD (cosine {b_PD_tight_cos}), has a lower bound of {b_PD_tight_lo}."),
    ("For PD, mean cosine similarity is {mn3_PD_mean} {{[}}{mn3_PD_lo}, {mn3_PD_hi}{{]}} across 6 languages at "
     "n $\\geq$ 3 and {mn10_PD_mean} {{[}}{mn10_PD_lo}, {mn10_PD_hi}{{]}} across 5 languages at n $\\geq$ 10"),
    ("CP retains cosine {mn3_CP_mean} across the same 4 languages at every threshold ({{[}}{mn3_CP_lo}, {mn3_CP_hi}{{]}} "
     "at n $\\geq$ 3 and n $\\geq$ 5, {{[}}{mn10_CP_lo}, {mn10_CP_hi}{{]}} at n $\\geq$ 10)"),
    ("HC profiles show cosine {mn10_HC_mean} {{[}}{mn10_HC_lo}, {mn10_HC_hi}{{]}} across the same 11 languages at "
     "n $\\geq$ 3, 5 and 10."),
    "e.g. HC {holm_n_HC}, PD {holm_n_PD})",
    "PD vs. HC d = +{holm_hcpd_d_2dp}, PD vs. CP d = {holm_cppd_d_2dp}, PD vs. ALS d = {holm_alspd_d_2dp}.",
    "(r = {holm_cppd_r_2dp}--{holm_dspd_r_2dp})",
    "near-zero separation within it (|r| $\\leq$ {holm_cpst_r_3dp})",
]
BIND_DGX = [
    "In the {t6n_200} speakers with at least 200 tokens per phonological class",
    "preserves the severity correlation (rho = {t6rho_200_3dp})",
    "At the strictest budget (200 tokens), rho = {t6rho_200_3dp} (p < $10^{{-79}}$, n = {t6n_200}) and epsilon-squared = {t6eps_200}.",
    "we repeated the analysis on the {cs_n} speakers who qualify at all token budgets",
    ("rho = {cs_rho_20_3dp} at 20 tokens, {cs_rho_50_3dp} at 50, {cs_rho_100_3dp} at 100, and {cs_rho_200_3dp} at 200 "
     "(all p < $10^{{-79}}$)"),
    "(rho = {cs_rho_min_3dp} to {cs_rho_max_3dp} on a common speaker set of {cs_n} speakers qualifying at all budgets)",
    "survives at all budgets (epsilon-squared = {t6eps_min_2dp}--{t6eps_max_2dp})",
    "(rho = {cs_rho_min_3dp} to {cs_rho_max_3dp} on a common speaker set across all token budgets)",
    ("PD cosine remains {mhc5_PD_cos} {{[}}{mhc5_PD_lo}, {mhc5_PD_hi}{{]}} across {mhc5_PD_nl} languages, CP is "
     "{mhc5_CP_cos} {{[}}{mhc5_CP_lo}, {mhc5_CP_hi}{{]}} across {mhc5_CP_nl} languages (against {mhc1_CP_cos} across "
     "{mhc1_CP_nl} languages with Swahili), and ALS remains {mhc5_ALS_cos} {{[}}{mhc5_ALS_lo}, {mhc5_ALS_hi}{{]}}."),
    "(only the lower PD bound moves, to {mhc20_PD_lo}, at HC $\\geq$ 20)",
]


def raw(e):
    V8._derive()
    return get(load(e["src"]), e["path"])


def claims(F):
    """Derived statements in the text that are not a single number (equalities, counts)."""
    out = []

    def eq(a, b, what):
        out.append((what, F.get(a) == F.get(b), f"{a}={F.get(a)} {b}={F.get(b)}"))
    for s in ("mean", "lo", "hi"):
        eq(f"mn3_CP_{s}", f"mn5_CP_{s}", "CP n>=3 and n>=5 give the same cosine/interval")
        eq(f"mn3_HC_{s}", f"mn5_HC_{s}", "HC n>=3 and n>=5 give the same cosine/interval")
        eq(f"mn3_HC_{s}", f"mn10_HC_{s}", "HC n>=3 and n>=10 give the same cosine/interval")
    for t in ("1", "5", "10"):
        eq("mn3_CP_mean", f"mn{t}_CP_mean", f"CP cosine identical at n>={t}")
    out.append(("ALS has 2 languages / 1 pair at n>=5", F.get("mn5_ALS_nl") == "2" and F.get("mn5_ALS_np") == "1",
                f"{F.get('mn5_ALS_nl')}/{F.get('mn5_ALS_np')}"))
    try:
        mn10 = get(load(V9), MN + ["10"])
        out.append(("ALS absent (<2 languages) at n>=10", "ALS" not in mn10, str(sorted(mn10))))
        pairs = get(load(V9), H + ["pairs"])
        sig = [p for p in pairs if p["p_holm"] < 0.001]
        ns = sorted(p["pair"] for p in pairs if p["p_holm"] >= 0.05)
        out.append(("12 of 15 Holm pairs p < 0.001", len(sig) == 12 and len(pairs) == 15, f"{len(sig)}/{len(pairs)}"))
        out.append(("non-significant pairs are CP-DS, CP-Stroke, DS-Stroke, each p_Holm = 1.00",
                    ns == ["CP vs DS", "CP vs Stroke", "DS vs Stroke"] and
                    all(round(p["p_holm"], 2) == 1.0 for p in pairs if p["pair"] in ns), str(ns)))
        within = [abs(p["r_rb"]) for p in pairs if p["pair"] in ns]
        out.append(("within-cluster |r| <= printed bound", max(within) <= float(F["holm_cpst_r_3dp"]), str(within)))
        pdx = [abs(p["r_rb"]) for p in pairs if p["pair"] in ("CP vs PD", "DS vs PD", "PD vs Stroke")]
        out.append(("PD vs execution-group r within printed range",
                    float(F["holm_cppd_r_2dp"]) == round(min(pdx), 2) and float(F["holm_dspd_r_2dp"]) == round(max(pdx), 2),
                    str(pdx)))
    except (KeyError, FileNotFoundError) as exc:
        out.append(("originals parsed", False, repr(exc)))
    out += claims_dgx(F)
    return out


def claims_dgx(F):
    out = []
    try:
        ft = get(load(V9), FT)
        for b, e in ft.items():
            m = e["severity_means"]
            mono = [m[s]["mean"] for s in ("control", "mild", "moderate", "severe")]
            out.append((f"fixed-token budget {b}: monotonic control > mild > moderate > severe",
                        all(x > y for x, y in zip(mono, mono[1:])), str(mono)))
            out.append((f"fixed-token budget {b}: n with severity equals n processed", e["n_rho"] == e["n"],
                        f"{e['n_rho']}/{e['n']}"))
        out.append(("common set size equals the 200-token budget n", F.get("cs_n") == F.get("t6n_200"),
                    f"{F.get('cs_n')}/{F.get('t6n_200')}"))
        mh = get(load(V9), P5 + ["exp2_min_hc"])
        for g in ("PD", "CP", "ALS"):
            out.append((f"min-HC {g}: cosine identical at HC>=5, 10, 20",
                        len({mh[t][g]["cos"] for t in ("5", "10", "20")}) == 1, str([mh[t][g]["cos"] for t in ("5", "10", "20")])))
            out.append((f"min-HC {g}: interval identical at HC>=5 and 10", mh["5"][g]["ci95"] == mh["10"][g]["ci95"],
                        f"{mh['5'][g]['ci95']} {mh['10'][g]['ci95']}"))
        for g in ("CP", "ALS"):
            out.append((f"min-HC {g}: interval identical at HC>=20 (only PD's lower bound moves)",
                        mh["5"][g]["ci95"] == mh["20"][g]["ci95"], f"{mh['5'][g]['ci95']} {mh['20'][g]['ci95']}"))
        out.append(("min-HC PD: only the lower bound moves at HC>=20",
                    mh["5"]["PD"]["ci95"][1] == mh["20"]["PD"]["ci95"][1] and mh["5"]["PD"]["ci95"][0] != mh["20"]["PD"]["ci95"][0],
                    f"{mh['5']['PD']['ci95']} {mh['20']['PD']['ci95']}"))
        out.append(("min-HC PD and ALS unchanged from HC>=1 to HC>=5",
                    all(mh["1"][g]["cos"] == mh["5"][g]["cos"] for g in ("PD", "ALS")), ""))
    except (KeyError, FileNotFoundError) as exc:
        out.append(("DGX originals parsed", False, repr(exc)))
    return out


def bind():
    F = formatted()
    ids = {e["id"]: e for e in E}
    import re as _re
    for tpl in BIND + BIND_DGX:
        names = _re.findall(r"(?<!\{)\{([A-Za-z0-9_|]+)\}(?!\})", tpl)
        try:
            filled = tpl.format(**F)
        except KeyError:
            continue
        for n in names:
            if n in ids and filled not in ids[n]["ctx"]:
                ids[n]["ctx"].append(filled)
    return F


def formatted():
    out = {}
    for e in E:
        try:
            out[e["id"]] = fmt(raw(e), e["rule"])
        except (KeyError, IndexError, TypeError) as exc:
            out[e["id"]] = f"MISSING({type(exc).__name__}:{exc})"
    return out


if __name__ == "__main__":
    F = bind()
    out = REV / "results" / "v9" / "v9_formatted_numbers.json"
    out.write_text(json.dumps({e["id"]: {"printed": F[e["id"]], "src": e["src"], "path": e["path"], "rule": e["rule"],
                                         "ctx": e["ctx"]} for e in E}, indent=1, default=str), encoding="utf-8")
    miss = [k for k, x in F.items() if x.startswith("MISSING")]
    print(f"{len(E)} registry entries; {len(miss)} missing; wrote {out}")
    for k in miss[:40]:
        print("  ", k, F[k])
