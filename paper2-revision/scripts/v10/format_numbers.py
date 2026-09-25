#!/usr/bin/env python3
"""v10 number registry and single-rounding formatter.

Extends scripts/v9/format_numbers.py (imported unchanged; which extends v8). v10 adds entries for every number
introduced or changed in fix round B5, each bound to its sentence context (ctx), and sourced from:
  results/v10/fig4_adjacent_intervals_v10.json   Figure 4 (master metadata for every panel)
  results/v10/v10_facts.json                      coverage (87.8%, 151), Dutch exclusion for all six groups,
                                                  VTA ratios, effective leave-one-dataset-out folds
  results/v10/v10_fixed_token_groups.json         groups compared by the fixed-token aetiology test
  results/v10/v10_originals_reproduction.json     originals run on the submitted inputs (permutation code)
  results/v9/v9_originals.json                    original bootstrap and leave-one-dataset-out scripts (corrected)
Every value is rounded ONCE from the saved output.
"""
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("format_numbers_v9",
                                               Path(__file__).resolve().parents[1] / "v9" / "format_numbers.py")
V9M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V9M)

REV = V9M.REV
fmt, get, load = V9M.fmt, V9M.get, V9M.load
E = V9M.E            # the v9 registry (list, shared; v10 entries are appended)
claims_v9 = V9M.claims

F4 = "results/v10/fig4_adjacent_intervals_v10.json"
FA = "results/v10/v10_facts.json"
FG = "results/v10/v10_fixed_token_groups.json"
RP = "results/v10/v10_originals_reproduction.json"
V9 = "results/v9/v9_originals.json"
RA = ["states", "corrected", "parsed", "robustness_analyses"]


def add(id_, src, path, rule, ctx=(), note=""):
    E.append({"id": id_, "src": src, "path": list(path), "rule": rule, "ctx": list(ctx), "note": note})


# ---- Figure 4 (caption) ------------------------------------------------------------------------------
for bb, key in (("XLS-R", "xlsr"), ("MMS", "mms")):
    base = ["backbones", bb, "adjacent", "mild-moderate"]
    add(f"f4_{key}_mm_margin", F4, base + ["margin"], "2dp")
    add(f"f4_{key}_mm_lo", F4, base + ["ci95", 0], "2dp")
    add(f"f4_{key}_mm_hi", F4, base + ["ci95", 1], "2dp")

# ---- Section 4.1: original stratified bootstrap (recorded labels) ------------------------------------
add("a1orig_rho", V9, RA + ["a1_stratified_bootstrap", "mean_rho"], "3dp")
add("a1orig_lo", V9, RA + ["a1_stratified_bootstrap", "ci95", 0], "3dp")
add("a1orig_hi", V9, RA + ["a1_stratified_bootstrap", "ci95", 1], "3dp")

# ---- Section 4.3: leave-one-dataset-out (effective deletions; original range) ------------------------
LE = ["lodo_effective"]
add("lodo_n_labels", FA, LE + ["n_labels"], "int")
add("lodo_n_eff", FA, LE + ["n_effective"], "int")
add("lodo_rho_lo", FA, LE + ["rho_range", 0], "3dp")
add("lodo_rho_hi", FA, LE + ["rho_range", 1], "3dp")
add("lodo_rho_mean", FA, LE + ["rho_mean"], "3dp")
add("lodo_eps_lo", FA, LE + ["eps2_range", 0], "3dp")
add("lodo_eps_hi", FA, LE + ["eps2_range", 1], "3dp")
add("lodo_eps_mean", FA, LE + ["eps2_mean"], "3dp")
add("lodo_p_rho_bound", FA, LE + ["max_p_rho"], "lt10:-62")
add("lodo_p_kw_bound", FA, LE + ["max_kw_p"], "lt10:-52")
add("a3orig_lo", V9, RA + ["a3_lodo", "rho_range", 0], "3dp")
add("a3orig_hi", V9, RA + ["a3_lodo", "rho_range", 1], "3dp")

# ---- coverage ------------------------------------------------------------------------------------------
add("cov_nasal_pct_1dp", FA, ["coverage", "nasal_valid_pct"], "1dp")
add("cov_absent_no_hb_int", FA, ["coverage", "absent_with_no_hubert_base_dprime"], "int")

# ---- Dutch exclusion (six groups) -------------------------------------------------------------------
add("nl6_maxchange_3dp", FA, ["dutch_exclusion_all", "max_abs_change"], "3dp")

# ---- VTA ratios -------------------------------------------------------------------------------------------
add("vta_ratio_CP_1dp", FA, ["vta", "hc_ratio", "CP"], "1dp")
add("vta_ratio_ALS_1dp", FA, ["vta", "hc_ratio", "ALS"], "1dp")
add("vta_ratio_DS_1dp", FA, ["vta", "hc_ratio", "DS"], "1dp")

# ---- fixed-token groups ------------------------------------------------------------------------------------
add("ftg_N_200", FG, ["budgets", "200", "kruskal_N"], "int")
add("ftg_PD_200", FG, ["budgets", "200", "by_group", "PD"], "int")

# ---- permutation code retained with the submission, on the submitted inputs (frozen log) ------------------
BQ = ["by_quantity"]
add("expB_sub_obs", RP, BQ + ["5.1 permutation: PD observed cosine", "original_on_submitted_inputs"], "3dp")
add("expB_sub_null", RP, BQ + ["5.1 permutation: PD null mean", "original_on_submitted_inputs"], "3dp")
add("expB_sub_p", RP, BQ + ["5.1 permutation: PD p", "original_on_submitted_inputs"], "2dp")
add("sapx_sub_eps_printed", RP, BQ + ["SAP-excluded composite eps2", "submitted_printed"], "3dp",
    note="value printed in the submitted manuscript (quoted in the letter)")
add("sapx_sub_n_printed", RP, BQ + ["SAP-excluded n", "submitted_printed"], "int",
    note="value printed in the submitted manuscript (quoted in the letter)")

# ---- sentence bindings for every v10 edit --------------------------------------------------------------------
BIND10 = [
    "tex:the original bootstrap script retained with the submission codes each speaker's recorded severity label instead and, on the same inputs, gives rho = {a1orig_rho} {{[}}{a1orig_lo}, {a1orig_hi}{{]}}.",
    "tex:after dropping each of the {lodo_n_labels} dataset labels in turn",
    "tex:so {lodo_n_eff} deletions change the sample; all {lodo_n_eff} yielded significant severity correlations (rho range: {lodo_rho_lo} to {lodo_rho_hi}, mean {lodo_rho_mean}; all p < 10$^{{{lodo_p_rho_bound}}}$)",
    "tex:(epsilon-squared range: {lodo_eps_lo} to {lodo_eps_hi}, mean {lodo_eps_mean}; all p < 10$^{{{lodo_p_kw_bound}}}$)",
    "tex:gives rho from {a3orig_lo} to {a3orig_hi} over the same {lodo_n_eff} deletions.",
    "tex:the mild--moderate step for XLS-R (difference {f4_xlsr_mm_margin}, 95\\% CI [{f4_xlsr_mm_lo}, {f4_xlsr_mm_hi}]) and MMS ({f4_mms_mm_margin} [{f4_mms_mm_lo}, {f4_mms_mm_hi}])",
    "tex:2,961 ({cov_nasal_pct_1dp}\\%) yielded a valid nasality estimate",
    "tex:{cov_absent_no_hb_int} of these also have no HuBERT-base d-prime",
    "tex:per-aetiology profile-cosine minima change by at most {nl6_maxchange_3dp} (PD",
    "tex:about twice the triangle area of the ALS and DS groups, {vta_ratio_CP_1dp} times that of CP and {vta_ratio_PD_1dp} times that of PD",
    "tex:at 200 tokens PD also drops out ({ftg_PD_200} qualifying speakers), leaving HC, CP, ALS and DS (N = {ftg_N_200})",
    "tex:(on the submitted inputs it gives observed {expB_sub_obs}, null mean {expB_sub_null}, p = {expB_sub_p})",
    "letter:on the submitted inputs it gives null mean {expB_sub_null}, p = {expB_sub_p})",
    "letter:not the submitted composite effect size (ε² {sapx_sub_eps_printed}), H, nasality values or n ({sapx_sub_n_printed})",
    "tex:(epsilon-squared = {t6eps_min_2dp}--{t6eps_max_2dp}; stroke is absent at every budget and PD at 200 tokens)",
]
# v8/v9 sentence contexts whose sentences were rewritten in v10 (the new contexts are in BIND10)
STALE_CTX = {"({v} times that of PD)", "(1.3 times that of PD)", "survives at all budgets (epsilon-squared = 0.36--0.58)"}

# ---- sentence bindings for body passages targeted by the v9 checker's planted-error test ---------------------------
MA = "results/v6/v6_master_analyses.json"
HV = "results/v10/v10_hr_values.json"
for g, key in (("HC", "HC"), ("PD", "PD"), ("CP", "CP"), ("ALS", "ALS"), ("DS", "DS"), ("Stroke", "Stroke")):
    add(f"g41_{key}", MA, ["coverage", "group_counts_nasal_valid", g], "int")
add("holm_n_sig_int", HV, ["values", "holm_n_sig", "value"], "int")
add("holm_n_pairs_int", HV, ["values", "holm_n_pairs", "value"], "int")
CL6 = "results/v6/v6_classifier.json"
for g in ("HC", "CP", "ALS"):
    add(f"clf_f1_{g}", CL6, ["corrected", "raw_cosine_train_mean", "per_class_f1", g], "3dp")
for s_ in ("control", "mild", "moderate", "severe"):
    add(f"ctc_mean_{s_}", MA, ["ctc_conf_severity", "means", s_], "3dp")
XL = "results/v6/v6_crossling.json"
add("sw_pd_all", XL, ["swahili", "pd_sw_vs_all_pd"], "3dp")
add("sw_pd_sk", XL, ["swahili", "pd_sw_vs_sk_pd"], "3dp")
add("sw_cp_zh", XL, ["swahili", "cp_sw_vs_zh_cp"], "3dp")
BIND10 += [
    "tex:healthy controls (HC, n = {g41_HC}), Parkinson's disease (PD, n = {g41_PD}), cerebral palsy (CP, n = {g41_CP}), amyotrophic lateral sclerosis (ALS, n = {g41_ALS}), Down syndrome (DS, n = {g41_DS}), and stroke (n = {g41_Stroke})",
    "tex:control vs. mild (n = {tm_cm_n_int} matched pairs, Cohen's d = {tm_cm_d_2dp},",
    "tex:mild vs. moderate (n = {tm_mm_n_int}, d = {tm_mm_d_2dp},",
    "tex:moderate vs. severe (n = {tm_ms_n_int}, d = {tm_ms_d_2dp},",
    "tex:PD {pc_min_PD_3dp} [{pc_lo_PD_3dp}, {pc_hi_PD_3dp}], CP {pc_min_CP_3dp} [{pc_lo_CP_3dp}, {pc_hi_CP_3dp}], ALS {pc_min_ALS_3dp} [{pc_lo_ALS_3dp}, {pc_hi_ALS_3dp}], DS {pc_min_DS_3dp} [{pc_lo_DS_3dp}, {pc_hi_DS_3dp}] and stroke {pc_min_Stroke_3dp} [{pc_lo_Stroke_3dp}, {pc_hi_Stroke_3dp}]",
    "tex:and HC {pc_min_HC_3dp} [{pc_lo_HC_3dp}, {pc_hi_HC_3dp}]",
    "tex:(Kendall's W = {kw_PD_2dp} PD, {kw_CP_2dp} CP, {kw_ALS_2dp} ALS, {kw_DS_2dp} DS, {kw_Stroke_2dp} stroke)",
    "tex:across all {holm_n_pairs_int} pairwise aetiology comparisons on composite consonant d-prime. After correction, {holm_n_sig_int} of {holm_n_pairs_int} pairs remain significant",
    "tex:XLS-R-300M rho = {sev_rho_xlsr_3dp}, HuBERT-base rho = {sev_rho_hubert-base_3dp}, MMS-300M rho = {sev_rho_mms_3dp}, wav2vec2-base rho = {sev_rho_wav2vec2_3dp}, WavLM-base rho = {sev_rho_wavlm_3dp}, and HuBERT-large rho = {sev_rho_hubert-large_3dp} (all p < 10$^{{{sev_p_bound}}}$",
    "tex:Q3 (185--2,222) rho = {q3_rho_2dp}, p < 10$^{{-16}}$, n = {q3_n_int}, {q3_dys_int} dysarthric; and Q4 (above 2,222) rho = {q4_rho_2dp}",
    "tex:HC speakers are best identified (F1 = {clf_f1_HC}), followed by CP ({clf_f1_CP}) and ALS ({clf_f1_ALS})",
    "tex:control {ctc_mean_control}, mild {ctc_mean_mild}, moderate {ctc_mean_moderate}, severe {ctc_mean_severe}",
    "tex:(HC {vta_mean_HC_1dp}, PD {vta_mean_PD_1dp}, DS {vta_mean_DS_1dp}, ALS {vta_mean_ALS_1dp}, CP {vta_mean_CP_1dp})",
    "tex:cosine similarity {sw_pd_all} to the multi-language PD mean and {sw_pd_sk} to the Slovak PD mean, and the 21 Swahili CP speakers show cosine {sw_cp_zh}",
    "tex:CP is {mhc5_CP_cos} {{[}}{mhc5_CP_lo}, {mhc5_CP_hi}{{]}} across {mhc5_CP_nl} languages (against {mhc1_CP_cos} across {mhc1_CP_nl} languages with Swahili)",
]


def claims(F):
    out = list(claims_v9(F))
    f4 = load(F4)["backbones"]
    ok = all(v["adjacent"][k]["excludes_zero"] == (not (bb in ("XLS-R", "MMS") and k == "mild-moderate"))
             for bb, v in f4.items() for k in v["adjacent"])
    out.append(("Figure 4: every adjacent-severity interval excludes zero except mild-moderate for XLS-R and MMS", ok, ""))
    out.append(("Figure 4: all six panels monotonic on point estimates", all(v["monotonic_point"] for v in f4.values()), ""))
    fa = load(FA)
    out.append(("VTA 'about twice' ALS and DS (1.9-2.1)", all(1.9 <= fa["vta"]["hc_ratio"][g] <= 2.1 for g in ("ALS", "DS")),
                str({g: fa["vta"]["hc_ratio"][g] for g in ("ALS", "DS")})))
    de = fa["dutch_exclusion_all"]["abs_change"]
    out.append(("Dutch exclusion: CP, ALS, DS and stroke unchanged at 3 decimals",
                all(round(de[g], 3) == 0 for g in ("CP", "ALS", "DS", "Stroke")), str(de)))
    le = fa["lodo_effective"]
    out.append(("LODO: the labels removing no speaker are CDSD and Domotica",
                le["labels_removing_no_speaker"] == ["CDSD", "Domotica"], str(le["labels_removing_no_speaker"])))
    tex = (REV / "paper2-csl_v10.tex").read_text(encoding="utf-8")
    import re as _re
    m = _re.search(r"We conducted (\w+) robustness analyses to address potential confounds: ([^.]*?), plus an aetiology-template", tex)
    words = {"eight": 8, "nine": 9, "ten": 10, "seven": 7}
    n_items = len(_re.split(r",\s*(?:and\s+)?", m.group(2))) if m else -1
    out.append(("Section 4.5: the stated number of robustness analyses equals the listed items",
                bool(m) and words.get(m.group(1)) == n_items, f"{m.group(1) if m else None} vs {n_items}"))
    fg = load(FG)
    out.append(("fixed-token: stroke absent from the aetiology test at every budget",
                all("Stroke" in v["groups_absent"] for v in fg["budgets"].values()), ""))
    out.append(("fixed-token: PD present at 20-100 tokens, absent at 200",
                all("PD" in fg["budgets"][b]["kruskal_groups"] for b in ("20", "50", "100")) and
                "PD" not in fg["budgets"]["200"]["kruskal_groups"], ""))
    out.append(("fixed-token: group totals equal the original's Processed counts", fg.get("totals_match_log") is True, ""))
    return out


def formatted():
    out = {}
    for e in E:
        try:
            out[e["id"]] = fmt(V9M.raw(e), e["rule"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            out[e["id"]] = f"MISSING({type(exc).__name__}:{exc})"
    return out


def raw(e):
    return V9M.raw(e)


def bind():
    V9M.bind()                       # v9 (and v8) templates, filled into the shared registry's ctx
    for e in E:
        e["ctx"] = [c for c in e["ctx"] if c not in STALE_CTX]
    F = formatted()
    ids = {e["id"]: e for e in E}
    import re as _re
    for tpl in BIND10:
        names = _re.findall(r"(?<!\{)\{([A-Za-z0-9_|]+)\}(?!\})", tpl)
        filled = tpl.format(**F)
        for n in names:
            if filled not in ids[n]["ctx"]:
                ids[n]["ctx"].append(filled)
    return F


if __name__ == "__main__":
    F = bind()
    out = REV / "results" / "v10" / "v10_formatted_numbers.json"
    out.write_text(json.dumps({e["id"]: {"printed": F[e["id"]], "src": e["src"], "path": e["path"], "rule": e["rule"],
                                         "ctx": e["ctx"]} for e in E}, indent=1, default=str), encoding="utf-8")
    miss = [k for k, x in F.items() if x.startswith("MISSING")]
    print(f"{len(E)} registry entries; {len(miss)} missing; wrote {out}")
    for k in miss[:40]:
        print("  ", k, F[k])
