#!/usr/bin/env python3
"""v8 number registry and single-rounding formatter.

Every reported value is read from its saved output at full precision and rounded ONCE here
(v7 check N-15..N-18, N-28, N-29, N-33 were double roundings of 4-decimal JSON values).
The registry (ENTRIES) is also the machine-readable body of NUMBERS_v8: each entry names the saved
output, the path to the value inside it, the rounding rule, the printed string, and the manuscript /
letter / matrix contexts where it appears. scripts/v8/check_consistency.py re-reads every source,
re-applies the rule, and checks the printed string and every context.

Rules: "Ndp" = N decimals; "int" = integer with thousands separator; "pct" = x*100 rounded to an
integer; "word" = integer spelled out; "sci" = mantissa x 10^exp with 1 decimal; "exact" = the
printed string must equal the stored value's string form.
Run directly to write results/v8/v8_formatted_numbers.json and NUMBERS_v8_registry.md.
"""
import json
import re
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
_CACHE = {}


def load(src):
    if src not in _CACHE:
        p = REV / src
        _CACHE[src] = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else p.read_text(encoding="utf-8")
    return _CACHE[src]


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def fmt(x, rule):
    if rule == "exact":
        return str(x)
    if rule == "int":
        return f"{int(Decimal(str(x)).quantize(Decimal('1'), ROUND_HALF_EVEN)):,}"
    if rule == "word":
        return WORDS[int(x)]
    if rule == "pct":
        return str(int(Decimal(str(float(x) * 100)).quantize(Decimal('1'), ROUND_HALF_EVEN)))
    if rule.startswith("lt10:"):
        e = int(rule.split(":")[1])
        return str(e) if float(x) < 10.0 ** e else f"NOT<10^{e}(max={x})"
    if rule == "sci1":
        m_, e_ = f"{float(x):.15e}".split("e")
        mm = Decimal(m_).quantize(Decimal("0.1"), ROUND_HALF_EVEN)
        ee = int(e_)
        if mm >= 10:
            mm, ee = (mm / 10).quantize(Decimal("0.1")), ee + 1
        return f"${mm} \\times 10^{{{ee}}}$"
    if rule == "t3row":   # Table 3 row: H, p, eps2, N
        return (f" & {fmt(x['H'], '1dp')}\n & {fmt(x['p'], 'sci1')}\n & {fmt(x['epsilon_squared'], '3dp')}\n"
                f" & {fmt(x['n_total_used'], 'int')}\n")
    if rule == "t4row":   # Table 4 row: mean, min, max
        return f" & {fmt(x['mean'], '3dp')}\n & {fmt(x['min'], '3dp')}\n & {fmt(x['max'], '3dp')}\n"
    if rule == "t6row":   # Table 6 row: n, rho, (p bound), eps2
        return f" & {fmt(x['n'], 'int')}\n & {fmt(x['rho'], '3dp')}\n"
    m = re.fullmatch(r"(\d)dp", rule)
    if m:
        q = Decimal(1).scaleb(-int(m.group(1)))
        v = Decimal(repr(float(x))).quantize(q, ROUND_HALF_EVEN)
        if v == 0:
            v = abs(v)
        s = f"{v:,}" if abs(v) >= 1000 else f"{v}"
        return s
    raise ValueError(rule)


def get(d, path):
    for k in path:
        d = d[k]
    return d


E = []  # registry


def add(id_, src, path, rule, ctx=(), note=""):
    E.append({"id": id_, "src": src, "path": list(path), "rule": rule, "ctx": list(ctx), "note": note})


T6 = "results/v8/v8_test6.json"
T2 = "results/test2_delta_corrected.json"
T2S = "results/v8/v8_test2_sensitivity.json"
ORIG = "results/v8/v8_originals.json"
MF = "results/v8/v8_master_facts.json"
QT = "results/v8/v8_token_quartiles.json"
RET = "results/v8/v8_cp_retention.json"
T3 = "results/rebuild_2026-09-23/table3_regenerated.json"
CL = "results/v6/v6_crossling.json"

# ---- cross-backbone (v8_test6) ---------------------------------------------------------
for a in ["PD", "CP", "ALS", "DS", "Stroke", "HC"]:
    add(f"pc_min_{a}_3dp", T6, ["profile_cosine_min_per_aetiology", a], "3dp")
    add(f"pc_min_{a}_2dp", T6, ["profile_cosine_min_per_aetiology", a], "2dp")
    pair = {"HC": "xlsr|mms"}.get(a, "hubert-large|mms")
    add(f"pc_lo_{a}_3dp", T6, ["profile_cosines", a, pair, "ci95", 0], "3dp")
    add(f"pc_hi_{a}_3dp", T6, ["profile_cosines", a, pair, "ci95", 1], "3dp")
add("pc_min_all_2dp", T6, ["profile_cosine_min_overall"], "2dp")
for a in ["PD", "CP", "ALS", "DS", "Stroke"]:
    add(f"kw_{a}_2dp", T6, ["kendalls_w", a], "2dp")
add("kw_min_2dp", T6, ["kendalls_w_range", 0], "2dp")
add("kw_max_2dp", T6, ["kendalls_w_range", 1], "2dp")
for a in ["PD", "CP", "ALS", "DS", "Stroke"]:
    add(f"grp_n_{a}_int", T6, ["group_n_per_backbone", "xlsr", a], "int")
add("sev_n_min_int", T6, ["severity_rho_per_backbone", "xlsr", "n"], "int", note="five non-HuBERT-base backbones share n")
add("sev_n_max_int", T6, ["severity_rho_per_backbone", "hubert-base", "n"], "int")
for b, lab in [("xlsr", "XLS-R"), ("hubert-base", "HuBERT-base"), ("mms", "MMS"), ("wav2vec2", "wav2vec2"),
               ("wavlm", "WavLM"), ("hubert-large", "HuBERT-large")]:
    add(f"sev_rho_{b}_3dp", T6, ["severity_rho_per_backbone", b, "rho"], "3dp")
for k in ["hubert-base|hubert-large", "hubert-base|wavlm", "hubert-base|wav2vec2", "hubert-base|xlsr", "hubert-base|mms",
          "hubert-large|wavlm", "hubert-large|wav2vec2", "hubert-large|xlsr", "hubert-large|mms", "wavlm|wav2vec2",
          "wavlm|xlsr", "wavlm|mms", "wav2vec2|xlsr", "wav2vec2|mms", "xlsr|mms"]:
    add(f"t5_{k}_3dp", T6, ["per_pair", k, "rho"], "3dp")
    add(f"t5_{k}_n_int", T6, ["per_pair", k, "n"], "int")
add("xb_min_2dp", T6, ["rho_min"], "2dp")
add("xb_max_2dp", T6, ["rho_max"], "2dp")
add("xb_min_3dp", T6, ["rho_min"], "3dp")
add("nl_min_without_3dp", T6, ["dutch_exclusion", "rho_min_without_nl"], "3dp")
add("nl_n_without_int", T6, ["dutch_exclusion", "n_range_without_nl", 0], "int")
add("nl_pc_PD_with_3dp", T6, ["dutch_exclusion", "profile_cos_min_with_nl", "PD"], "3dp")
add("nl_pc_PD_without_3dp", T6, ["dutch_exclusion", "profile_cos_min_without_nl", "PD"], "3dp")
add("nl_pc_HC_with_3dp", T6, ["dutch_exclusion", "profile_cos_min_with_nl", "HC"], "3dp")
add("nl_pc_HC_without_3dp", T6, ["dutch_exclusion", "profile_cos_min_without_nl", "HC"], "3dp")
add("nl_pc_maxchange_3dp", T6, ["_derived", "nl_pc_maxchange"], "3dp")

add("sev_p_bound", T6, ["_derived", "sev_p_max"], "lt10:-120", note="all per-backbone severity p < 10^-120")
add("tamil_pooled_tokens_int", "results/v6/v6_tamil_provenance.json", ["_derived", "pooled_tokens"], "int",
    note="21,710 SLR65 + 142,490 SSNCE control tokens = the 164,200 of the extraction logs")
add("tamil_slr65_tokens_int", "results/v6/v6_tamil_provenance.json", ["files", "track4_results_xlsr.csv", "slr65_18_n_phones_sum"], "int")

# ---- Test 2 ---------------------------------------------------------------------------
add("t2_raw_lo_3dp", T2, ["primary_raw", "ci95", 0], "3dp")
add("t2_raw_hi_3dp", T2, ["primary_raw", "ci95", 1], "3dp")
add("t2_4d_lo_3dp", T2, ["sensitivities", "no_manner_4dim", "ci95", 0], "3dp")
add("t2_4d_hi_3dp", T2, ["sensitivities", "no_manner_4dim", "ci95", 1], "3dp")
add("t2_2s_lo_3dp", T2, ["sensitivities", "two_stage_bootstrap", "ci95", 0], "3dp")
add("t2_2s_hi_3dp", T2, ["sensitivities", "two_stage_bootstrap", "ci95", 1], "3dp")
add("t2_mc5_d_4dp", T2S, ["min_cell_5", "delta"], "4dp")
add("t2_mc5_lo_4dp", T2S, ["min_cell_5", "ci95", 0], "4dp")
add("t2_mc5_hi_4dp", T2S, ["min_cell_5", "ci95", 1], "4dp")
add("t2_mc5_nperm_int", T2S, ["n_perm"], "int")

# ---- original scripts (corrected inputs) ----------------------------------------------
P = ["parsed", "corrected"]
for key, nm in [("cm", "Control vs Mild"), ("mm", "Mild vs Moderate"), ("ms", "Moderate vs Severe")]:
    add(f"tm_{key}_n_int", ORIG, P + ["token_matched", nm, "n_matched"], "int")
    add(f"tm_{key}_d_2dp", ORIG, P + ["token_matched", nm, "d"], "2dp")
add("tm_d_min_2dp", ORIG, P + ["token_matched", "Mild vs Moderate", "d"], "2dp")
add("tm_d_max_2dp", ORIG, P + ["token_matched", "Moderate vs Severe", "d"], "2dp")
R = P + ["feature_ranking_by_abs_severity_rho"]
add("rank_first_nasal_word", ORIG, R + ["first_counts", "nasal"], "word")
add("rank_top3_nasal_word", ORIG, R + ["top3_count_per_feature", "nasal"], "word")
add("rank_top3_voicing_word", ORIG, R + ["top3_count_per_feature", "voicing"], "word")
add("rank_top3_strident_word", ORIG, R + ["top3_count_per_feature", "strident"], "word")
S = P + ["sap_excluded"]
add("sx_rho_excl_3dp", ORIG, S + ["sap_excluded", "severity", "rho"], "3dp")
add("sx_n_excl_int", ORIG, S + ["sap_excluded", "severity", "n"], "int")
add("sx_rho_full_3dp", ORIG, S + ["full", "severity", "rho"], "3dp")
add("sx_n_full_int", ORIG, S + ["full", "severity", "n"], "int")
add("sx_comp_eps_excl_3dp", ORIG, S + ["sap_excluded", "composite_kw", "eps2"], "3dp")
add("sx_comp_H_excl_int", ORIG, S + ["sap_excluded", "composite_kw", "H"], "int")
add("sx_comp_eps_full_3dp", ORIG, S + ["full", "composite_kw", "eps2"], "3dp")
add("sx_n_speakers_excl_int", ORIG, S + ["sap_excluded", "n_speakers"], "int")
for f, nm in [("nasal", "nasal_dprime"), ("high", "high_dprime"), ("round", "round_dprime")]:
    add(f"sx_{f}_full_3dp", ORIG, S + ["full", "per_feature_kw_dysarthric_groups", nm, "eps2"], "3dp")
    add(f"sx_{f}_excl_3dp", ORIG, S + ["sap_excluded", "per_feature_kw_dysarthric_groups", nm, "eps2"], "3dp")

# ---- master facts ---------------------------------------------------------------------
add("vta_ratio_PD_1dp", MF, ["vta", "ratio_hc_to_group", "PD"], "1dp")
for g in ["HC", "PD", "DS", "ALS", "CP"]:
    add(f"vta_mean_{g}_1dp", MF, ["vta", "mean", g], "1dp")
add("vta_n_any_int", MF, ["vta", "n_any_group_valid"], "int")
add("yt_fr_n_int", MF, ["table1", "YouTube_French", "n"], "int")
add("yt_fr_als_int", MF, ["table1", "YouTube_French", "aetiology", "als"], "int")
add("yt_fr_hc_int", MF, ["table1", "YouTube_French", "aetiology", "healthy"], "int")
add("yt_de_n_int", MF, ["table1", "YouTube_German", "n"], "int")
add("yt_de_als_int", MF, ["table1", "YouTube_German", "aetiology", "als"], "int")
add("yt_de_hc_int", MF, ["table1", "YouTube_German", "aetiology", "healthy"], "int")
add("blocks_int", MF, ["test2_blocks", "n_blocks"], "int")
add("blocks_mixing_int", MF, ["test2_blocks", "n_mixing_blocks"], "int")
add("blocks_perm_int", MF, ["test2_blocks", "n_speakers_in_mixing_blocks"], "int")
add("blocks_sap_int", MF, ["test2_blocks", "n_speakers_in_mixing_blocks_SAP"], "int")
add("svd_n_int", MF, ["svd", "n"], "int")
add("svd_hc_int", MF, ["svd", "aetiology", "healthy"], "int")
add("svd_unknown_int", MF, ["svd", "aetiology", "unknown"], "int")

# ---- quartiles, retention, Table 3, crossling ----------------------------------------
Q = ["corrected_min3"]
add("q_n_int", QT, Q + ["n"], "int")
for i in (1, 2, 3):
    add(f"q_b{i}_int", QT, Q + ["boundaries", i - 1], "int")
for q in (1, 2, 3, 4):
    add(f"q{q}_rho_2dp", QT, Q + ["quartiles", f"Q{q}", "rho"], "2dp")
    add(f"q{q}_n_int", QT, Q + ["quartiles", f"Q{q}", "n"], "int")
    add(f"q{q}_dys_int", QT, Q + ["quartiles", f"Q{q}", "n_dysarthric"], "int")
    add(f"q{q}_lo_int", QT, Q + ["quartiles", f"Q{q}", "n_phones_range", 0], "int")
add("q1_p_3dp", QT, Q + ["quartiles", "Q1", "p"], "3dp")
add("q2_p_3dp", QT, Q + ["quartiles", "Q2", "p"], "3dp")
for sev, k in [("mild", "mild"), ("moderate", "mod")]:
    add(f"ret_en_{k}_lo_pct", RET, ["retention", "en", sev, "range_5c", 0], "pct")
    add(f"ret_en_{k}_hi_pct", RET, ["retention", "en", sev, "range_5c", 1], "pct")
for f, col, dp in [("low_eps", "low_dprime", "3dp"), ("back_H", "back_dprime", "1dp"),
                   ("xpos_H", "cross_position_cosim", "1dp"), ("rate_H", "speech_rate", "1dp")]:
    add(f"t3_{f}_{dp}", T3, ["_rows", col, "epsilon_squared" if "eps" in f else "H"], dp)
for t in (3, 10):
    add(f"mn{t}_CP_lo_3dp", CL, [f"min_n_{t}", "CP", "ci95", 0], "3dp")
    add(f"mn{t}_CP_hi_3dp", CL, [f"min_n_{t}", "CP", "ci95", 1], "3dp")

# ---- table rows bound to their labels (context templates; {v} = printed row) ----------
T5LAB = {"hubert-base|wavlm": "HuBERT-base vs WavLM-base", "hubert-large|wavlm": "HuBERT-large vs WavLM-base",
         "hubert-base|hubert-large": "HuBERT-base vs HuBERT-large", "hubert-base|wav2vec2": "HuBERT-base vs wav2vec2-base",
         "wavlm|wav2vec2": "WavLM-base vs wav2vec2-base", "hubert-large|wav2vec2": "HuBERT-large vs wav2vec2-base",
         "hubert-base|mms": "HuBERT-base vs MMS-300M", "wavlm|mms": "WavLM-base vs MMS-300M",
         "hubert-large|mms": "HuBERT-large vs MMS-300M", "hubert-base|xlsr": "HuBERT-base vs XLS-R-300M",
         "wavlm|xlsr": "WavLM-base vs XLS-R-300M", "hubert-large|xlsr": "HuBERT-large vs XLS-R-300M",
         "wav2vec2|xlsr": "Wav2vec2-base vs XLS-R-300M", "wav2vec2|mms": "Wav2vec2-base vs MMS-300M",
         "xlsr|mms": "XLS-R-300M vs MMS-300M"}
for k, lab in T5LAB.items():
    add(f"t5row_{k}", T6, ["per_pair", k, "rho"], "3dp", ctx=[lab + "\n & {v}\n & 3,101"])
T3LAB = {"nasal_dprime": "Nasality d$'$", "voicing_dprime": "Voicing d$'$", "sonorant_dprime": "Sonorance d$'$",
         "strident_dprime": "Stridency d$'$", "manner_dprime": "Manner d$'$", "high_dprime": "Height d$'$",
         "low_dprime": "Lowness d$'$", "back_dprime": "Backness d$'$", "round_dprime": "Rounding d$'$",
         "vowel_triangle_area": "Vowel triangle area", "boundary_sharpness": "Boundary sharpness",
         "cross_position_cosim": "Cross-position cosine", "speech_rate": "Speech rate", "pause_rate": "Pause rate",
         "vowel_duration_cv": "Vowel duration CV"}
for col, lab in T3LAB.items():
    add(f"t3row_{col}", T3, ["_rows", col], "t3row", ctx=[lab + "\n{v}"])
for g, lab in [("CP", "CP\n & 4 (en, zh, sw, ta)\n"), ("ALS", "ALS\n & 4 (en, de, fr, pt)\n"),
               ("PD", "PD\n & 6 (en, es, it, nl, pt, sk)\n"), ("HC", "HC\n & 11 (de, en, es, fr, hu, it, nl, pt, sk, ta, zh)\n")]:
    add(f"t4row_{g}", CL, ["min_n_3", g], "t4row", ctx=[lab + "{v}"])
for b in ("20", "50", "100", "200"):
    add(f"t6row_{b}", "results/v6/v6_fixed_token_dgx.json", ["per_budget", b], "t6row", ctx=[b + "\n{v}"])
    add(f"t6eps_{b}", "results/v6/v6_fixed_token_dgx.json", ["per_budget", b, "eps2"], "3dp")

# ---- context bindings for the numbers edited in v8 ------------------------------------
CTX = {
    "tm_cm_n_int": ["control vs. mild (n = {v} matched pairs"], "tm_cm_d_2dp": ["matched pairs, Cohen's d = {v}, p"],
    "tm_mm_n_int": ["mild vs. moderate (n = {v}, d"], "tm_ms_n_int": ["moderate vs. severe (n = {v}, d"],
    "tm_d_min_2dp": ["(d = {v}--"], "sx_rho_excl_3dp": ["weaker: rho = {v} (n = "], "sx_n_excl_int": ["(n = {v}, p < $10^{-63}$)"],
    "sx_comp_eps_excl_3dp": ["epsilon-squared = {v} (H = "], "sx_comp_eps_full_3dp": ["versus {v} for the six groups"],
    "pc_min_PD_3dp": ["PD {v} ["], "pc_min_ALS_3dp": ["ALS {v} ["], "pc_min_Stroke_3dp": ["stroke {v} ["],
    "kw_PD_2dp": ["(Kendall's W = {v} PD,"], "kw_min_2dp": ["Kendall's W = {v}--"], "pc_min_all_2dp": ["is at least {v}; fine-grained"],
    "q1_n_int": ["n = {v}, of whom"], "q2_rho_2dp": ["rho = {v}, p = 0.016"], "q_b3_int": ["(above {v}) rho"],
    "ret_en_mild_lo_pct": ["labelled mild ({v}--"], "ret_en_mod_lo_pct": ["or moderate ({v}--"],
    "nl_pc_maxchange_3dp": ["change by at most {v} (PD"], "vta_ratio_PD_1dp": ["({v} times that of PD)"],
    "t2_mc5_d_4dp": ["(Delta = +{v}, ["], "t2_mc5_nperm_int": ["p < 0.001 ({v} permutations"],
    "sev_n_min_int": ["n = {v}-1,761"], "grp_n_DS_int": ["DS {v} and stroke"],
    "rank_first_nasal_word": ["nasality ranks first in {v} of six"],
    "t3_low_eps_3dp": ["Lowness d$'$\n & 961.7\n & $1.2 \\times 10^{-205}$\n & {v}\n"],
    "t5_hubert-base|wavlm_3dp": ["HuBERT-base vs WavLM-base\n & {v}\n"], "t5_wav2vec2|xlsr_3dp": ["Wav2vec2-base vs XLS-R-300M\n & {v}\n"],
    "yt_fr_als_int": ["French-speaking ({v} with ALS"], "yt_de_als_int": ["German-speaking ({v} with ALS"],
}
for e in E:
    if e["id"] in CTX:
        e["ctx"] = CTX[e["id"]]


def _derive():
    tp = load("results/v6/v6_tamil_provenance.json")
    f0 = tp["files"]["track4_results_xlsr.csv"]
    tp.setdefault("_derived", {})["pooled_tokens"] = f0["slr65_18_n_phones_sum"] + f0["ssnce_control_n_phones_sum"]
    t6 = load(T6)
    t6.setdefault("_derived", {})["sev_p_max"] = max(v["p"] for v in t6["severity_rho_per_backbone"].values())
    w, wo = t6["dutch_exclusion"]["profile_cos_min_with_nl"], t6["dutch_exclusion"]["profile_cos_min_without_nl"]
    t6.setdefault("_derived", {})["nl_pc_maxchange"] = max(abs(w[a] - wo[a]) for a in w)
    t3 = load(T3)
    t3["_rows"] = {r["column"]: r for r in t3["rows"]}


def raw(e):
    _derive()
    return get(load(e["src"]), e["path"])


def formatted():
    return {e["id"]: fmt(raw(e), e["rule"]) for e in E}


if __name__ == "__main__":
    F = formatted()
    out = REV / "results" / "v8" / "v8_formatted_numbers.json"
    out.write_text(json.dumps({e["id"]: {"printed": F[e["id"]], "raw": raw(e), "src": e["src"], "path": e["path"],
                                         "rule": e["rule"]} for e in E}, indent=1, default=str), encoding="utf-8")
    print(f"{len(E)} registry entries; wrote {out}")
