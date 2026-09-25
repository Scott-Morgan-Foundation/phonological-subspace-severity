#!/usr/bin/env python3
"""v8 executable consistency check: numbers in the manuscript, Highlights and working-draft letter
against the saved outputs. Exits non-zero on ANY failure. Do not weaken it to pass.

Scope. Scanned: paper2-csl_v8.tex from \\begin{abstract} to the reference list; submission_highlights_v2.txt;
response_letter_v8.md. Every numeric token is scanned except single-digit integers (design descriptors
such as "6 backbones"), which are outside this parser's scope and are listed as such in the report.

Checks.
 A. Registry entries (scripts/v8/format_numbers.py, ENTRIES): each source file exists; the value at the
    stated path, rounded once by the stated rule, equals the printed string. (Mismatch = FAIL.)
 B. Every scanned numeric token must be (in order):
    1. a registry value (A), or
    2. a whitelisted non-result token (citation, section/figure/table number, year, identifier, method
       parameter, external descriptor) listed in WHITELIST below with its reason, or
    3. located automatically: a numeric leaf in one of the saved outputs in SOURCES that rounds to the
       token at the token's own printed precision (same sign when the token is signed); p-value bounds
       "10^{-X}" are located as a p-value leaf in [10^-(X+1), 10^-X).
    Anything else is UNMAPPED = FAIL.
 C. Repeated quantities: registry entries that read the same value (same source and path) must print
    consistently (e.g. 0.98 and 0.980), which single rounding guarantees; any violation = FAIL.
 D. Every source named in NUMBERS_v8.md exists.
Output: results/v8/check_consistency_report.json and .md (full token table with the source located).
"""
import csv
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import format_numbers as FN  # noqa: E402

REV = Path(__file__).resolve().parents[2]
OUTJ = REV / "results" / "v8" / "check_consistency_report.json"
OUTM = REV / "results" / "v8" / "check_consistency_report.md"

SOURCES = [
    "results/v8/v8_test6.json", "results/v8/v8_test2_sensitivity.json", "results/v8/v8_originals.json",
    "results/v8/v8_master_facts.json", "results/v8/v8_token_quartiles.json", "results/v8/v8_cp_retention.json",
    "results/v8/v8_tamil_hubert_provenance.json", "results/v8/v8_fig_text_sizes.json", "results/v8/v8_cite_order.json",
    "results/test2_delta_corrected.json", "results/test3_effects_corrected.csv",
    "results/rebuild_2026-09-23/table3_regenerated.json", "results/rebuild_2026-09-23/severity_counts.json",
    "results/rebuild_2026-09-23/aetiology_undetermined.json", "results/rebuild_2026-09-23/pairwise_d_extracted.json",
    "results/rebuild_2026-09-23/pairwise_moderate_only.json", "results/rebuild_2026-09-23/copas_recompute_v4.json",
    "results/rebuild_2026-09-23/group_counts_section_4_1.json",
    "results/v6/v6_master_analyses.json", "results/v6/v6_crossling.json", "results/v6/v6_classifier.json",
    "results/v6/v6_fixed_token_dgx.json", "results/v6/v6_test45_analysis.json", "results/v6/v6_misc.json",
    "results/v6/v6_figures_1_3.json", "results/v6/v6_tamil_provenance.json", "results/v6/v6_table1_check.json",
    "results/v5/fig2_group_means.json", "results/v5/fig4_adjacent_intervals.json",
    "results/v5/hubert_tamil_swap_impact.json", "results/v5/unsourced_numbers.json",
    "results/v5/swap_run/swapped/results/test2_delta_corrected.json",
    "corrected/CHANGES_step1.json", "results/test6_backbones.json", "results/test6_backbones_corrected.json",
    "results/test7_lodo_repro.json",
    "results/v8/originals/corrected/reviewer_experiments.log",
    "results/v8/originals/corrected/cross_model_comparison.log",
    "results/v8/originals/corrected/sap_excluded_sensitivity.log",
]
TEXT_SOURCES = {"../../Paper 1/arxiv/manuscript.tex": "companion paper [2] (published preprint)",
                "frozen_package/paper2_csl_submission.zip:paper2-csl.tex": "submitted manuscript (historical values)"}

# Non-result tokens: (token regex, regex on the 18 characters BEFORE the token, regex on the 18
# characters AFTER it, reason). None = no condition. Kept narrow on purpose: a result count such as
# "10 of 13 features" or "n = 322" must not match here.
WHITELIST = [
    (r"[\d.,]+", r"(Joshy|dependent and|and$|reaches|SRCC|SALR|; |\(r = |accuracy on UA-Speech \(\[6\]; )$", r"^(\\?%)? ?(speaker-(in)?dependent|accuracy|across five|between SSL)", "result quoted from cited literature"),
    (r"(93\.97|49\.22|70\.48|0\.761|0\.81)", None, None, "result quoted from cited literature (Joshy & Rajan; SALR [6]; Bae et al. [15]; Cho et al. [4])"),
    (r"\d+", r"(YCSLA-D-|YCSLA-D-\d\d-)$", None, "manuscript identifier"),
    (r"\d+", r"\[\s*$|[\[,\-–]\s*\d*\s*$", r"^\s*[\],\-–]", "citation number"),
    (r"\d+(\.\d+)*", r"(Section|Sections|Table|Figure|Fig\.|§|item|Items?|\$\\S\$|Rule|Outcome|R[12]-|D-)\s*$", None, "cross-reference number"),
    (r"(19|20)\d\d", None, None, "year"),
    (r"\d+", r"(syn|LDC|doi\.org/|DOI: |10\.\d{4,}/|[a-z]-|[A-Za-z]|_|\.)$", None, "identifier / URL / DOI / model id"),
    (r"\d+", None, r"^(m|M\b|k\b|K\b|-ft|-ll|h\b|th\b|[A-Za-z_])", "identifier / model id / unit-bearing descriptor / ordinal"),
    (r"(768|1024|94\.7|316\.6|95\.0|315\.4|960|60,000|94,000|436|128|1,100)", None, r"^(\s*(M params|hours|h\b|languages|,|\s*$)|\s*&|\s*\\\\|K h|\+ languages)", "Table 2 / model card descriptor (external, not computed)"),
    (r"(16|50|20)", None, r"^\s*(kHz|Hz|ms)", "audio parameter"),
    (r"(3\.1\.3|3\.1\.0|2\.1\.0)", r"v$", None, "software version"),
    (r"(42|1,000|10,000|2,000|50|20|100|200|150|16|10|0\.8|1\.25|15|3|94|85|70|84|89|790|5)", r"(seed |penalty |at |and |\\geq\$? ?|≥ ?|> ?|< ?|>\s*|<\s*|\(|between |exactly |, |Directive 2019/|Article |from |\$\\pm\$|\+/-|−|-|^)\s*$",
     r"^\s*(\\?%|resamples|speaker resamples|permutations|draws|random draws|seeded draws|tokens|ms\b|dB|and 0 dB|times|-fold|per cell|speakers\)|Article|safeguards|\\% = |-(84|94)|\$\^)", "method parameter / legal article / threshold"),
    (r"(13|15|12|10|25|26|5|9|4|3)", None, r"^\s*-?\s*(languages?\b|datasets?\b|corpora|heterogeneous corpora|backbones?|aetiologies|main aetiologies|dataset (rows|labels)|main-analysis features|segmental|consonant|-dimensional|dimensional|SSL|phonological and prosodic|pairwise|folds|with sufficient|robustness|features\b|contrasts|clinical|seeded|analysis groups|groups)", "design count (description of the study design)"),
]


def fmt_dp(x, d):
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(float(x))).quantize(q, ROUND_HALF_EVEN)
    return abs(v) if v == 0 else v


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield path, float(obj)
    elif isinstance(obj, str):
        try:
            yield path, float(obj)
        except ValueError:
            return


def load_leaves():
    out = []
    for s in SOURCES:
        p = REV / s
        if not p.exists():
            out.append((s, None, None))
            continue
        if p.suffix == ".json":
            for path, x in leaves(json.loads(p.read_text(encoding="utf-8"))):
                out.append((s, path, x))
        elif p.suffix == ".csv":
            with open(p, encoding="utf-8") as fh:
                for i, row in enumerate(csv.DictReader(fh)):
                    for k, v in row.items():
                        try:
                            out.append((s, f"row{i}.{k}", float(v)))
                        except (TypeError, ValueError):
                            pass
        else:
            txt = p.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", txt):
                out.append((s, f"char{m.start()}", float(m.group())))
    return out


NUM = re.compile(r"(?P<sign>(?<![\w\d.)\]}\-–])[-−](?=\d))?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)")
SCI = re.compile(r"(\d+(?:\.\d+)?)\s*\$?\s*(?:\\times|×)\s*\$?\s*10\s*\$?\s*\^?\s*\{?\s*([-−]\d+)\s*\}?")
BOUND = re.compile(r"(?:<|&lt;)\s*\$?\s*10\s*\$?\s*\^?\s*\{?\s*([-−]\d+)\s*\}?\$?")


def scan(name, text):
    toks = []
    consumed = set()
    for m in SCI.finditer(text):
        toks.append({"doc": name, "pos": m.start(), "kind": "sci", "tok": f"{m.group(1)}e{m.group(2).replace(chr(0x2212), '-')}"})
        consumed.update(range(m.start(), m.end()))
    for m in BOUND.finditer(text):
        if m.start() in consumed:
            continue
        toks.append({"doc": name, "pos": m.start(), "kind": "bound", "tok": m.group(1).replace("−", "-")})
        consumed.update(range(m.start(), m.end()))
    for m in NUM.finditer(text):
        if any(i in consumed for i in range(m.start(), m.end())):
            continue
        n = m.group("num")
        if re.fullmatch(r"\d", n):
            continue
        prev = text[max(0, m.start() - 1):m.start()]
        if prev.isalpha() or prev in "_\\":
            continue
        toks.append({"doc": name, "pos": m.start(), "kind": "num",
                     "tok": ("-" if m.group("sign") else "") + n})
    for t in toks:
        t["ctx"] = text[max(0, t["pos"] - 45):t["pos"] + 45].replace("\n", " ")
        t["before"] = text[max(0, t["pos"] - 18):t["pos"]].replace("\n", " ")
        end = t["pos"] + len(t["tok"].lstrip("-"))
        t["after"] = text[end:end + 18].replace("\n", " ")
    return toks


CTX_DOCS = {"tex": (REV / "paper2-csl_v8.tex").read_text(encoding="utf-8"),
            "letter": (REV / "response_letter_v8.md").read_text(encoding="utf-8")}


def main():
    fails, report = [], {"registry": [], "tokens": [], "scope_note": "single-digit integers are not scanned"}
    # ---- A: registry
    F = FN.formatted()
    printed = {}
    for e in FN.E:
        p = REV / e["src"]
        if not p.exists():
            fails.append(f"A missing source {e['src']} ({e['id']})")
            continue
        val = F[e["id"]]
        if val.startswith("NOT<"):
            fails.append(f"A bound violated {e['id']}: {val}")
        for tpl in e.get("ctx", []):
            doc = "letter" if tpl.startswith("letter:") else "tex"
            snippet = tpl.split(":", 1)[1] if tpl.startswith(("letter:", "tex:")) else tpl
            snippet = snippet.replace("{v}", val)
            if snippet not in CTX_DOCS[doc]:
                fails.append(f"A context not found for {e['id']} ({val!r}) in {doc}: {snippet[:120]!r}")
        report["registry"].append({"id": e["id"], "printed": val, "src": e["src"], "path": e["path"], "rule": e["rule"]})
        printed.setdefault(val.replace(",", ""), []).append(e["id"])
    # ---- C: same (src,path) consistency
    by_sp = {}
    for e in FN.E:
        by_sp.setdefault((e["src"], json.dumps(e["path"])), []).append(e)
    for k, es in by_sp.items():
        raws = {json.dumps(FN.raw(e)) for e in es}
        if len(raws) != 1:
            fails.append(f"C inconsistent raw for {k}")
        for e in es:
            if e["rule"].endswith("dp") and FN.fmt(FN.raw(e), e["rule"]) != F[e["id"]]:
                fails.append(f"C rounding mismatch {e['id']}")
    # ---- B: tokens
    tex = (REV / "paper2-csl_v8.tex").read_text(encoding="utf-8")
    a0, a1 = tex.find("\\begin{abstract}"), tex.find("\\section*{References}")
    docs = {"tex": tex[a0:a1], "highlights": (REV / "submission_highlights_v2.txt").read_text(encoding="utf-8"),
            "letter": (REV / "response_letter_v8.md").read_text(encoding="utf-8")}
    L = load_leaves()
    missing_src = sorted({s for s, pth, x in L if pth is None})
    for s in missing_src:
        fails.append(f"B missing auto-search source {s}")
    import math
    L = [x for x in L if x[1] is not None and math.isfinite(x[2])]
    texts = {}
    for k, v in TEXT_SOURCES.items():
        if ":" in k and k.endswith(".tex") and ".zip:" in k:
            import zipfile
            zp, inner = k.split(":", 1)
            try:
                texts[k] = zipfile.ZipFile(REV / zp).read(inner).decode("utf-8")
            except (FileNotFoundError, KeyError):
                fails.append(f"B missing text source {k}")
        else:
            p = (REV / k).resolve()
            if p.exists():
                texts[k] = p.read_text(encoding="utf-8")
            else:
                fails.append(f"B missing text source {k}")
    for name, text in docs.items():
        for t in scan(name, text):
            status, where = None, None
            tok = t["tok"]
            if t["kind"] == "num":
                key = tok.lstrip("-").replace(",", "")
                if key in printed or tok.replace(",", "") in printed:
                    status, where = "REGISTRY", printed.get(key) or printed.get(tok.replace(",", ""))
                if status is None:
                    for pat, bre, are, reason in WHITELIST:
                        if re.fullmatch(pat, tok.lstrip("-")) and (bre is None or re.search(bre, t["before"])) \
                                and (are is None or re.search(are, t["after"])):
                            status, where = "WHITELIST", reason
                            break
                if status is None:
                    d = len(tok.split(".")[1]) if "." in tok else 0
                    v = float(tok.replace(",", ""))
                    cands = []
                    for s, pth, x in L:
                        for xx in (x, x * 100):
                            if d == 0 and float(int(v)) == v:
                                ok = abs(xx - v) < 1e-9 or (abs(abs(xx) - abs(v)) < 1e-9 and not tok.startswith("-"))
                            else:
                                rep_ = repr(float(xx))
                                dec = rep_.split(".")[1] if "." in rep_ and "e" not in rep_ else ""
                                if len(dec) == d + 1 and dec.endswith("5"):
                                    ok = False   # a pre-rounded value ending in 5: rounding it again is a double rounding
                                else:
                                    r = fmt_dp(xx, d)
                                    ok = (r == Decimal(tok.replace(",", ""))) or (not tok.startswith("-") and abs(r) == abs(Decimal(tok.replace(",", ""))))
                            if ok:
                                cands.append(f"{s}:{pth}")
                                break
                        if len(cands) >= 3:
                            break
                    if cands:
                        status, where = "LOCATED", cands
                    else:
                        for k, txt in texts.items():
                            if tok in txt:
                                status, where = "LOCATED_TEXT", k
                                break
            elif t["kind"] == "sci":
                mant, ex = tok.split("e")
                d = len(mant.split(".")[1]) if "." in mant else 0
                cands = [f"{s}:{pth}" for s, pth, x in L if x != 0 and
                         abs(int(f"{x:e}".split("e")[1]) - int(ex)) == 0 and fmt_dp(float(f"{x:e}".split("e")[0]), d) == Decimal(mant)][:3]
                if cands:
                    status, where = "LOCATED", cands
            elif t["kind"] == "bound" and tok in printed:
                status, where = "REGISTRY", printed[tok]
            elif t["kind"] == "bound":
                ex = int(tok)
                cands = [f"{s}:{pth}" for s, pth, x in L if re.search(r"(^|[._\[])(p|kw_p|pvalue|p_value|perm\w*p\w*)($|[\]._])", pth.split(".")[-1] if pth else "")
                         and 0 < x < 10.0 ** ex and x >= 10.0 ** (ex - 1)][:3]
                if not cands:
                    cands = [f"{s}:{pth}" for s, pth, x in L if "log" in s and 0 < x < 10.0 ** ex and x >= 10.0 ** (ex - 1)][:3]
                if cands:
                    status, where = "LOCATED_BOUND", cands
            if status is None:
                status = "UNMAPPED"
                fails.append(f"B UNMAPPED {name}: {tok!r} ... {t['ctx']!r}")
            report["tokens"].append({**{k: t[k] for k in ("doc", "kind", "tok", "ctx")}, "status": status, "where": where})
    # ---- D: NUMBERS sources exist
    nb = (REV / "NUMBERS_v8.md").read_text(encoding="utf-8")
    for m in sorted(set(re.findall(r"`((?:results|corrected|scripts|frozen_package|v5_investigation)/[^`]+?)`", nb))):
        path = m.split(" ")[0].split("→")[0].strip()
        if not (REV / path).exists():
            fails.append(f"D NUMBERS_v8 names a missing file: {path}")
    counts = {}
    for t in report["tokens"]:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    report["counts"] = counts
    report["n_registry"] = len(FN.E)
    report["failures"] = fails
    OUTJ.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = ["# check_consistency report (v8)", "", f"Registry entries: {len(FN.E)}. Token status counts: {counts}.",
             "Scope: single-digit integers are not scanned.", "", f"## Failures ({len(fails)})", ""]
    lines += [f"- {f}" for f in fails] or ["- none"]
    lines += ["", "## Token table", "", "| doc | token | status | located in / reason | context |", "|---|---|---|---|---|"]
    for t in report["tokens"]:
        w = t["where"] if isinstance(t["where"], str) else "; ".join(t["where"] or [])
        lines.append(f"| {t['doc']} | {t['tok']} | {t['status']} | {str(w)[:160]} | {t['ctx'].replace('|', '/')} |")
    OUTM.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"tokens: {counts}; failures: {len(fails)}")
    for f in fails[:80]:
        print(" ", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
