#!/usr/bin/env python3
"""v10 edits (fix round B5, v9_check/CHECK_REPORT.md). Applies exact replacements to the v10 copies of the manuscript,
letter and matrix; each old string must occur exactly the stated number of times (default 1), else the script stops.
Writes results/v10/apply_v10_edits.json (one record per edit)."""
import json
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
F = {"tex": REV / "paper2-csl_v10.tex", "letter": REV / "response_letter_v10.md",
     "matrix": REV / "reviewer_response_matrix_v10.md"}
T = {k: p.read_text(encoding="utf-8") for k, p in F.items()}
LOG = []


def rep(doc, item, old, new, count=1):
    n = T[doc].count(old)
    if n != count:
        sys.exit(f"[{item}] expected {count} occurrence(s) in {doc}, found {n}: {old[:90]!r}")
    T[doc] = T[doc].replace(old, new)
    LOG.append({"doc": doc, "item": item, "old": old, "new": new, "count": count})


# ---------------- manuscript ----------------
rep("tex", "NV-1 Dutch exclusion DS/stroke",
    "HC 0.965 to 0.962; CP and ALS unchanged)", "HC 0.965 to 0.962; CP, ALS, DS and stroke unchanged)")

rep("tex", "F-5/F-6/NV-6 Table 1 caption",
    "Severity labels derive from clinical assessment or from intelligibility thresholds (Stipancic), or are unavailable (None); ``None'' with a note means that the dataset's dysarthric speakers carry no severity label.",
    "The severity-source column states where each dataset's severity labels come from: a clinical rating (with the instrument where known), intelligibility percentages mapped to classes with the Stipancic thresholds, thresholded listener intelligibility ratings, the corpus documentation, or an assignment by the authors or a collaborator; ``None'' means that the dataset's dysarthric speakers carry no severity label. The evidence for each entry is recorded in the verification package.")
T1 = [
    ("SAP", " & PD, ALS, CP, DS, Stroke\n & Clinical + Stipancic\n", " & PD, ALS, CP, DS, Stroke\n & SAP listener intelligibility ratings (1--7), thresholded\n"),
    ("TORGO", " & TORGO [22] (15)\n & CP, ALS, HC\n & Clinical intelligibility\n", " & TORGO [22] (15)\n & CP, ALS, HC\n & Corpus documentation\n"),
    ("UA-Speech", " & UA-Speech [23] (15)\n & CP\n & Clinical intelligibility\n", " & UA-Speech [23] (15)\n & CP\n & Corpus documentation (mild/severe only)\n"),
    ("COPAS", " & COPAS [26] (227)\n & Mixed, HC\n & DIA clinical\n", " & COPAS [26] (227)\n & Mixed, cleft palate, laryngectomy, voice disorder, unknown, HC\n & DIA intelligibility \\% (Stipancic)\n"),
    ("CHASING", " & CHASING [27] (8)\n & PD\n & Clinical\n", " & CHASING [27] (8)\n & PD\n & Listener intelligibility ratings, thresholded\n"),
    ("TreasureHunters1", " & TreasureHunters1 [28] (5)\n & PD\n & Clinical\n", " & TreasureHunters1 [28] (5)\n & PD\n & Assigned by the authors from published listener results\n"),
    ("Neurovoz", " & Neurovoz [29] (111)\n & PD, HC\n & Clinical\n", " & Neurovoz [29] (111)\n & PD, HC\n & Clinical (GRBAS G)\n"),
    ("PC-GITA", " & PC-GITA [30] (100)\n & PD, HC\n & Clinical\n", " & PC-GITA [30] (100)\n & PD, HC\n & Clinical (UPDRS speech item)\n"),
    ("IPVS", " & IPVS [31] (65)\n & PD, HC\n & Clinical\n", " & IPVS [31] (65)\n & PD, HC\n & Derived from CPS3 scores\n"),
    ("MDSC", " & MDSC [33] (56)\n & CP, HC\n & Clinical\n", " & MDSC [33] (56)\n & CP, HC\n & Intelligibility \\% (Stipancic)\n"),
    ("SSNCE", " & SSNCE (LDC2021S04) (30)\n & CP, HC\n & Clinical\n", " & SSNCE (LDC2021S04) (30)\n & CP, HC\n & Corpus documentation\n"),
    ("YouTube German", " & YouTube German (collected by authors) (13)\n & ALS, HC\n & Self-reported\n", " & YouTube German (collected by authors) (13)\n & ALS, HC\n & Assigned by the authors (not clinical)\n"),
    ("Hungarian", " & PD, Stroke, Mixed, unknown, HC\n & Clinical\n", " & PD, Stroke, Mixed, unknown, HC\n & Supplied by a collaborator (15 of 39 speakers)\n"),
    ("YouTube French", " & YouTube French (collected by authors) (24)\n & ALS, HC\n & Self-reported\n", " & YouTube French (collected by authors) (24)\n & ALS, HC\n & Assigned by the authors (not clinical)\n"),
    ("CDLI", " & CP, PD, MS, HC\n & Clinical\n", " & CP, PD, MS, HC\n & Corpus documentation\n"),
]
for name, old, new in T1:
    rep("tex", f"F-5/F-6/NV-6 Table 1 {name}", old, new)

rep("tex", "NV-6 §3.2 SAP unknown severity wording",
    "(1,045 speakers whose recordings lack clinical severity ratings)", "(1,045 speakers without intelligibility ratings)")
rep("tex", "NV-6 §3.2 severity sources (i)",
    "Severity labels derive from three sources: (i) clinical ground truth from original dataset documentation (e.g., perceptual rating by speech-language pathologists for COPAS, intelligibility percentages for TORGO and UA-Speech); (ii) Stipancic threshold derivation",
    "Severity labels come from four kinds of source (Table 1 gives the source for each dataset): (i) clinical ratings or severity classes supplied with the dataset (e.g., the UPDRS speech item for PC-GITA, GRBAS G for Neurovoz, the severity classes documented for TORGO and UA-Speech); (ii) Stipancic threshold derivation")
rep("tex", "NV-6 §3.2 severity sources (ii)-(iv)",
    "for datasets reporting intelligibility percentages without categorical labels, using thresholds of >94\\% = control, 85-94\\% = mild, 70-84\\% = moderate, <70\\% = severe; and (iii) healthy control assignment for speakers from normative datasets (LibriSpeech, SLR65 Tamil).",
    "for datasets reporting intelligibility percentages without categorical labels (COPAS, whose percentages come from the Dutch Intelligibility Assessment, and MDSC), using thresholds of >94\\% = control, 85-94\\% = mild, 70-84\\% = moderate, <70\\% = severe; (iii) thresholded listener intelligibility ratings (SAP's 1--7 ratings, CHASING's listener ratings) or labels assigned by the authors or supplied by a collaborator (the YouTube recordings, TreasureHunters1 and 15 of the 39 Hungarian Dysarthria speakers); and (iv) healthy control assignment for speakers from normative datasets (LibriSpeech, SLR65 Tamil).")

rep("tex", "D-2 §4.1 bootstrap re-implementation",
    "to {[}-0.512, -0.443{]} (voicing), none crossing zero.",
    "to {[}-0.512, -0.443{]} (voicing), none crossing zero. This bootstrap is a re-implementation that codes severity by the Statistical conventions (healthy controls = 0); the original bootstrap script retained with the submission codes each speaker's recorded severity label instead and, on the same inputs, gives rho = -0.542 {[}-0.573, -0.513{]}. We report the re-implementation because it uses the coding of every other severity correlation in the paper; both scripts are in the repository.")

rep("tex", "NV-5 VTA ratio",
    "show healthy speakers with about twice the triangle area of the CP, ALS and DS groups (1.3 times that of PD)",
    "show healthy speakers with about twice the triangle area of the ALS and DS groups, 2.4 times that of CP and 1.3 times that of PD")

rep("tex", "F-4 Figure 4 file", "figures/fig4_severity_v5}", "figures/fig4_severity_v10}")
rep("tex", "F-4 Figure 4 caption metadata",
    "are excluded from the severity bins; all six panels are built the same way.",
    "are excluded from the severity bins; severity labels, control status and aetiology come from the master table for every panel, so all six panels are built the same way.")
rep("tex", "F-4 Figure 4 caption intervals",
    "(difference 0.05, 95\\% CI [-0.02, 0.11]) and MMS (0.03 [-0.03, 0.09])",
    "(difference 0.05, 95\\% CI [-0.02, 0.12]) and MMS (0.03 [-0.04, 0.08])")

rep("tex", "D-3/F-3 §4.3 LODO",
    "after dropping each of the 26 dataset labels in turn (UA-Speech and its control set are dropped separately). All 26 folds yielded significant severity correlations (rho range: -0.577 to -0.406, mean -0.541; all p < 10$^{-62}$) and large aetiology effect sizes on the composite in the severity-labelled six-group sample (epsilon-squared range: 0.154 to 0.324, mean 0.290; all p < 10$^{-52}$).",
    "after dropping each of the 26 dataset labels in turn (UA-Speech and its control set are dropped separately). Two of these labels (CDSD and Domotica) have no speaker in the severity-labelled six-group sample, so 24 deletions change the sample; all 24 yielded significant severity correlations (rho range: -0.577 to -0.406, mean -0.541; all p < 10$^{-62}$) and large aetiology effect sizes on the composite in that sample (epsilon-squared range: 0.154 to 0.324, mean 0.290; all p < 10$^{-52}$). This analysis is a re-implementation that codes severity by the Statistical conventions (healthy controls = 0); the original leave-one-dataset-out script retained with the submission codes recorded severity labels and gives rho from -0.574 to -0.408 over the same 24 deletions.")

rep("tex", "G-2/F-7 fixed-token groups",
    "Table 6 shows that the severity correlation and aetiology discrimination survive at all token budgets from 20 to 200 tokens per class.",
    "Table 6 shows that the severity correlation survives at all token budgets from 20 to 200 tokens per class and that the aetiology Kruskal-Wallis test is significant at every budget, but the groups it compares change with the budget. The test uses groups with at least 5 qualifying speakers: stroke never qualifies, so at 20, 50 and 100 tokens it compares HC, PD, CP, ALS and DS, and at 200 tokens PD also drops out (2 qualifying speakers), leaving HC, CP, ALS and DS (N = 465). The PD separation reported in Section 4.1 is therefore not tested at the strictest budget.")
rep("tex", "G-2/F-7 fixed-token scope",
    "Within the speakers who meet each budget, this indicates that the severity and aetiology signals are not artifacts of token-count differences,",
    "Within the speakers who meet each budget, this indicates that the severity signal, and the aetiology signal among the groups compared, are not artifacts of token-count differences,")
rep("tex", "G-2/F-7 Table 6 caption",
    "\\textbf{Table 6.} Fixed-token d-prime: severity correlation and aetiology discrimination at different token budgets per phonological class.",
    "\\textbf{Table 6.} Fixed-token d-prime: severity correlation and aetiology discrimination at different token budgets per phonological class. The aetiology test compares HC, PD, CP, ALS and DS at 20--100 tokens and HC, CP, ALS and DS at 200 tokens (stroke, and PD at 200 tokens, have fewer than 5 qualifying speakers).")
rep("tex", "G-2/F-7 §5.2",
    "and the aetiology discrimination survives at all budgets (epsilon-squared = 0.36--0.58).",
    "and the aetiology discrimination remains significant at all budgets among the groups with at least 5 qualifying speakers (epsilon-squared = 0.36--0.58; stroke is absent at every budget and PD at 200 tokens).")

rep("tex", "G-L1/NV-2 SAP-excluded re-analysis",
    "ALS is thinned to n = 14 outside SAP. SAP contributes 420 PD,",
    "ALS is thinned to n = 14 outside SAP. These values are a re-analysis on the corrected inputs with the SAP-excluded script in the public repository; on the submitted inputs that script does not reproduce all of the SAP-excluded values reported at submission, so we do not treat it as the script that produced them. SAP contributes 420 PD,")

rep("tex", "NV-3 §5.1 permutation history",
    "For PD, the analysis on the originally submitted inputs gave observed 0.979, null mean 0.982, p = 0.84; its permutation draws were not retained. A seeded reconstruction on the corrected inputs gives observed 0.978, null mean 0.975, p = 0.21. In both, the observed cosine does not exceed the within-language null.",
    "For PD, the submitted manuscript reported observed 0.979, null mean 0.982, p = 0.84; the permutation draws behind these values were not retained, and the permutation code retained with the submission does not reproduce them (on the submitted inputs it gives observed 0.979, null mean 0.998, p = 1.00). A seeded reconstruction on the corrected inputs gives observed 0.978, null mean 0.975, p = 0.21. In each case the observed cosine does not exceed the within-language null.")

rep("tex", "H-5/#10 repo tag", "paper2-csl-revision-2026-09-25-r3", "paper2-csl-revision-2026-09-25-r4", count=3)

# ---------------- letter ----------------
rep("letter", "G-L3 header",
    "Every number below is mapped to a saved output in NUMBERS_v9.md.",
    "Every number below is listed in NUMBERS_v10.md with the saved output it comes from (or, for values quoted from the submitted manuscript, the submitted passage).")
rep("letter", "G-L1 originals reproduction",
    "were retained and are re-run unchanged on the corrected inputs.",
    "were retained and are re-run unchanged on the corrected inputs. Run on the submitted inputs, they reproduce the submitted values with three exceptions: the SAP-excluded script reproduces the submitted severity correlation and vowel-feature values but not the submitted composite effect size (ε² 0.106), H, nasality values or n (2,161), so it is not established that it produced them; the per-backbone ranking script reproduces the submitted per-backbone correlations and \"nasality ranks first in 3 of 6\", but not the submitted statement that the same three contrasts are in the top three for all six backbones; and the retained permutation code does not reproduce the submitted PD permutation summary (Section 5.1).")
rep("letter", "G-L1 close-to-submitted",
    "Their results are close to the submitted ones (token-matched: control vs mild n = 185, d = 0.60; mild vs moderate n = 99, d = 0.54; moderate vs severe n = 37, d = 0.94; SAP-excluded: severity ρ -0.408, composite ε² 0.162).",
    "On the corrected inputs the token-matched comparisons are close to the submitted ones (control vs mild n = 185, d = 0.60; mild vs moderate n = 99, d = 0.54; moderate vs severe n = 37, d = 0.94); the SAP-excluded analysis, a re-analysis on the corrected inputs, gives severity ρ -0.408 and composite ε² 0.162.")
rep("letter", "NV-3 item 5 permutation history",
    "and Section 5.1 reports the within-language permutation analysis once for each input state: on the originally submitted inputs (PD observed 0.979, null mean 0.982, p = 0.84; the original permutation draws were not retained) and as a seeded reconstruction on the corrected inputs (observed 0.978, null mean 0.975, p = 0.21); in both, the observed cosine does not exceed the within-language null.",
    "and Section 5.1 gives the within-language permutation analysis as reported in the submission (PD observed 0.979, null mean 0.982, p = 0.84; the draws were not retained, and the permutation code retained with the submission does not reproduce these values: on the submitted inputs it gives null mean 0.998, p = 1.00) and as a seeded reconstruction on the corrected inputs (observed 0.978, null mean 0.975, p = 0.21); in each case the observed cosine does not exceed the within-language null.")
rep("letter", "#10 repo tag", "paper2-csl-revision-2026-09-25-r3", "paper2-csl-revision-2026-09-25-r4")
rep("letter", "#10 script dirs",
    "(including `scripts/v8/` and `scripts/v9/` with unchanged copies of the retained original scripts they re-run)",
    "(including `scripts/v8/`, `scripts/v9/` and `scripts/v10/`; `scripts/v8/originals/` and `scripts/v9/originals/` hold unchanged copies of the retained original scripts they re-run)")
rep("letter", "G-L3 closing",
    "and every number in it is mapped in NUMBERS_v9.md to a saved analysis output, which an executable consistency check (`scripts/v9/check_consistency.py`) compares against the manuscript.",
    "and every number in it is listed in NUMBERS_v10.md with the saved analysis output it comes from (or, for values quoted from the submitted manuscript, the submitted passage); an executable consistency check (`scripts/v10/check_consistency.py`) compares them against the manuscript.")

# ---------------- matrix ----------------
rep("matrix", "header", "> WORKING COPY for v9. Line and page anchors regenerated from the final v9 compile (paper2-csl_v9.pdf).",
    "> WORKING COPY for v10. Line and page anchors regenerated from the final v10 compile (paper2-csl_v10.pdf).")
rep("matrix", "header NUMBERS", "and NUMBERS_v9.md maps every reported number to its output.",
    "and NUMBERS_v10.md maps every reported number to its output.")
rep("matrix", "#10 repo tag", "paper2-csl-revision-2026-09-25-r3", "paper2-csl-revision-2026-09-25-r4")
rep("matrix", "G-M1 R1-W4 script",
    "`scripts/v8/v8_cite_order.py` reports the first-occurrence order of the v8 manuscript.",
    "`scripts/v10/v10_cite_order.py` reports the first-occurrence order of the v10 manuscript.")
rep("matrix", "G-M1 R1-W4 where", "| References; `scripts/cite_audit.py` |", "| References; `scripts/v10/v10_cite_order.py` |")
rep("matrix", "G-M2 D-1",
    "; their summaries were recomputed with the corrected labels and are unchanged except the Dutch faithfulness check",
    "; their summaries were recomputed with the corrected labels and are reported in Section 4.5")
rep("matrix", "NV-3 D-6",
    "Seeded reconstruction on the corrected inputs reproduces the observed value (0.978) but not the null (0.975, p = 0.21); same qualitative conclusion; stated once per input state in §5.1",
    "The permutation code retained with the submission does not reproduce the submitted values (on the submitted inputs: observed 0.979, null mean 0.998, p = 1.00); a seeded reconstruction on the corrected inputs gives observed 0.978, null mean 0.975, p = 0.21; in each case the observed cosine does not exceed the null; §5.1 gives the submitted values as reported, then the reconstruction")

for k, p in F.items():
    p.write_text(T[k], encoding="utf-8")
(REV / "results" / "v10" / "apply_v10_edits.json").write_text(json.dumps(LOG, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(LOG)} edits applied")
