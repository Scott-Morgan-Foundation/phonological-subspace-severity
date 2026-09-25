#!/usr/bin/env python3
"""Build NUMBERS_v10.md from (1) the curated source table (v9 rows with v10 changes), (2) the single-rounding registry
(scripts/v10/format_numbers.py), (3) the derived-claim results, (4) the high-risk bindings (scripts/v10/hr_bindings.json)
and (5) the token table produced by scripts/v10/check_consistency.py. Run check_consistency.py first, then this
script, then check_consistency.py again (check D reads this file)."""
import importlib.util
import json
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location("format_numbers_v10", Path(__file__).resolve().parent / "format_numbers.py")
FN = importlib.util.module_from_spec(_s)
_s.loader.exec_module(FN)

V10ROWS = """| Figure 4 (bars, n, adjacent-severity intervals in the caption) | Figure 4 | `results/v10/fig4_adjacent_intervals_v10.json` (`scripts/v10/regen_fig4_v10.py`: speaker metadata from the master for every panel) | corrected |
| Coverage: 2,961 (87.8%) valid nasality estimates; 151 of the 182 speakers absent from the five non-HuBERT-base files have no HuBERT-base d-prime (§3.2, §3.3) | §3.2, §3.3 | `results/v10/v10_facts.json` → `coverage` (`scripts/v10/v10_facts.py`) | corrected |
| Dutch exclusion, profile-cosine minima for all six groups ("at most 0.003") | §3.1 | `results/v10/v10_facts.json` → `dutch_exclusion_all` | corrected |
| VTA group means and HC-to-group ratios (about twice ALS and DS; 2.4 CP; 1.3 PD) | §4.1 | `results/v10/v10_facts.json` → `vta` | corrected |
| Leave-one-dataset-out: 26 labels, 24 deletions that change the sample (CDSD and Domotica remove no speaker); ranges and means | §4.3 | `results/v10/v10_facts.json` → `lodo_effective` (from `results/v6/v6_master_analyses.json` → `lodo.folds`, re-implementation, controls = 0) | corrected |
| Original leave-one-dataset-out script (recorded labels): rho −0.574 to −0.408 | §4.3 | `results/v9/v9_originals.json` → `states.corrected.parsed.robustness_analyses.a3_lodo` | corrected |
| Original stratified bootstrap (recorded labels): rho −0.542 [−0.573, −0.513] | §4.1 | `results/v9/v9_originals.json` → `states.corrected.parsed.robustness_analyses.a1_stratified_bootstrap` | corrected |
| Fixed-token aetiology test: groups compared per budget (stroke absent at every budget; PD absent at 200 tokens, 2 speakers; N = 465) | §4.5, Table 6 caption, §5.2 | `results/v10/v10_fixed_token_groups.json` (`scripts/v10/v10_fixed_token_groups.py`, run on the DGX; totals equal the original's Processed counts) | corrected |
| Permutation code retained with the submission, on the submitted inputs (observed 0.979, null 0.998, p = 1.00) | §5.1, letter 5, D-6 | `results/v10/v10_originals_reproduction.json` → `by_quantity` (frozen log `frozen_package/logs/fixed_token_dprime.log`) | submitted |
| Originals run on the submitted inputs against the submitted manuscript (reproduced / not reproduced), incl. SAP-excluded ε² 0.106 and n 2,161 as printed at submission | letter 2 | `results/v10/v10_originals_reproduction.json` (`scripts/v10/v10_originals_reproduction.py`) | submitted |
| Table 1 severity sources (evidence per dataset) | Table 1, §3.2 | `results/v10/v10_facts.json` → `table1_severity_sources` | corrected + processing scripts |
| High-risk design counts, tallies, thresholds, Table 1 counts, companion-paper values | abstract, contributions, tables, captions, Conclusion | `results/v10/v10_hr_values.json` (`scripts/v10/v10_hr_values.py`; each value with its source) | corrected / scripts / companion manuscript |
| Table 2 model descriptors (external; as printed at submission, not re-verified) | Table 2 | `results/v10/v10_external_descriptors.json` | external |
| Citation first-occurrence order (letter 3, R1-W4) | letter | `results/v10/v10_cite_order.json` (`scripts/v10/v10_cite_order.py`) | v10 manuscript |
"""

REPLACE = {
    "`../../Paper 1/arxiv/manuscript.tex` | published preprint |": "`external/paper1_arxiv_manuscript.tex` (arXiv source copy) | companion manuscript |",
    "`results/v5/fig2_group_means.json`, `results/v5/fig4_adjacent_intervals.json` |": "`results/v5/fig2_group_means.json` (Figure 4: see the v10 row) |",
    "| Citation first-occurrence order (letter 3, R1-W4) | letter | `results/v8/v8_cite_order.json` (`scripts/v8/v8_cite_order.py`) | v8 manuscript |\n": "",
}


def main():
    v9 = (REV / "NUMBERS_v9.md").read_text(encoding="utf-8")
    cur = v9[v9.index("| Quantity | Location |"):v9.index("## 2. Registry")].rstrip() + "\n"
    for a, b in REPLACE.items():
        n = cur.count(a)
        if n != 1:
            raise SystemExit(f"curated-table replacement expected once, found {n}: {a[:70]}")
        cur = cur.replace(a, b)
    header = ("# NUMBERS_v10.md — Paper 2 v10 (2026-09-25)\n\n"
              "Every number in `paper2-csl_v10.tex`, `submission_highlights_v2.txt` and `response_letter_v10.md` is mapped here to\n"
              "the saved output that contains it, or, for values quoted from the submitted manuscript, the submitted passage.\n"
              "Five layers:\n\n"
              "1. **Curated sources** (below): v9 rows and the v10 rows. \"corrected\" = the inputs in `corrected/`.\n"
              "2. **Registry**: values read at full precision from their outputs and rounded once by `scripts/v10/format_numbers.py`.\n"
              "3. **Derived claims** evaluated by `scripts/v10/check_consistency.py`.\n"
              "4. **High-risk bindings**: every number in the abstract, Highlights, contributions, tables, figure captions and\n"
              "   Conclusion is bound to its sentence context and a saved output (`scripts/v10/hr_bindings.json`).\n"
              "5. **Token table**: every numeric token scanned in the three documents with its status from\n"
              "   `scripts/v10/check_consistency.py` (`results/v10/check_consistency_report.md`). Outside the high-risk regions,\n"
              "   LOCATED means a value found at the printed precision in a saved output, which is not a semantic binding.\n\n"
              "Being listed here does not mean a number has been verified; the separate checking pass does that.\n\n"
              "## 1. Curated sources\n\n")
    F = FN.bind()
    lines = [header + cur + V10ROWS, "## 2. Registry (rounded once)", "",
             "| id | printed | output | path | rule | bound to sentence |", "|---|---|---|---|---|---|"]
    for e in FN.E:
        lines.append(f"| {e['id']} | {F[e['id']]} | `{e['src']}` | {'/'.join(map(str, e['path']))} | {e['rule']} | "
                     f"{'yes' if e['ctx'] else ''} |")
    rep = json.loads((REV / "results" / "v10" / "check_consistency_report.json").read_text(encoding="utf-8"))
    lines += ["", "## 3. Derived claims", "", "| claim | holds | detail |", "|---|---|---|"]
    for c in rep.get("claims", []):
        lines.append(f"| {c['claim']} | {'yes' if c['ok'] else 'NO'} | {c['detail']} |")
    hb = json.loads((REV / "scripts" / "v10" / "hr_bindings.json").read_text(encoding="utf-8"))["bindings"]
    lines += ["", f"## 4. High-risk bindings ({len(hb)})", "", "| region | printed | source | context |", "|---|---|---|---|"]
    for b in hb:
        src = b["src"].get("id") or f"{b['src']['src']} → {b['src'].get('path', '')}"
        ctx = (b["before"][-30:] + "[" + b["value_at_build"] + "]" + b["after"]).replace("\n", " ").replace("|", "/")
        lines.append(f"| {b['region']} | {b['value_at_build'].strip()[:40].replace(chr(10), ' ')} | {src} | {ctx} |")
    lines += ["", f"## 5. Token table ({len(rep['tokens'])} tokens; status counts {rep['counts']})", "",
              "| doc | token | status | output / reason | context |", "|---|---|---|---|---|"]
    for t in rep["tokens"]:
        w = t["where"] if isinstance(t["where"], str) else "; ".join(t["where"] or [])
        lines.append(f"| {t['doc']} | {t['tok']} | {t['status']} | {w[:200]} | {t['ctx'].replace('|', '/')} |")
    (REV / "NUMBERS_v10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote NUMBERS_v10.md", len(lines), "lines")


if __name__ == "__main__":
    main()
