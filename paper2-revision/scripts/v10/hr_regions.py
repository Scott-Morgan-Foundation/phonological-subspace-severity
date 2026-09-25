#!/usr/bin/env python3
"""High-risk regions of the v10 manuscript for scripts/v10/check_consistency.py: abstract, contributions, every
table (caption + body), every figure caption and the Conclusion; plus the Highlights file. Also the token scanner
used inside them (every digit sequence, single digits included, plus 'x 10^{-k}' forms and '< 10^{-k}' bounds)."""
import re
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
TEX = REV / "paper2-csl_v10.tex"
HL = REV / "submission_highlights_v2.txt"


def regions(tex):
    R = {}

    def span(name, a_pat, b_pat, from_a=True):
        a = tex.find(a_pat)
        if a < 0:
            raise SystemExit(f"region start not found: {name}: {a_pat!r}")
        b = tex.find(b_pat, a + (len(a_pat) if from_a else 0))
        if b < 0:
            raise SystemExit(f"region end not found: {name}: {b_pat!r}")
        R[name] = (a, b)

    span("abstract", "\\begin{abstract}", "\\end{abstract}")
    span("contributions", "Our contributions are:", "\\end{enumerate}")
    span("conclusion", "\\section{Conclusion}", "\\subsection{Declaration of Generative AI Use}")
    for n in "123456":
        span(f"table{n}", f"\\textbf{{Table {n}.", "\\end{longtable}")
    for n in "12345":
        span(f"figure{n}", f"\\emph{{\\textbf{{Figure {n}.}}", "\n")
    return R


NUM = re.compile(r"(?P<sign>(?<![\w\d.)\]}\-–])[-−](?=\d))?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)")
SCI = re.compile(r"(\d+(?:\.\d+)?)\s*\$?\s*(?:\\times|×)\s*\$?\s*10\s*\$?\s*\^?\s*\{?\s*([-−]\d+)\s*\}?")
BOUND = re.compile(r"(?:<|&lt;)\s*\$?\s*10\s*\$?\s*\^?\s*\{?\s*([-−]\d+)\s*\}?\$?")


def structural(full, t):
    """Reason string if the token is structural (not a reported quantity), else None. Deliberately narrow:
    citation numbers, cross-references, table layout widths, identifiers inside \\texttt{}/\\url{},
    publication years of cited works, and the confidence level '95\\%'."""
    s, e = t["s"], t["e"]
    before, after = full[max(0, s - 24):s], full[e:e + 12]
    tok = t["tok"]
    if t["kind"] == "ident":
        return "identifier (digits attached to letters)"
    if re.search(r"\[\s*(\d+\s*[,–-]\s*)*$", before) and re.match(r"\s*([,–-]\s*\d+\s*)*\]", after):
        return "citation number"
    if re.search(r"(Table|Figure|Section|Sections|Fig\.|\\S|§)\s*(\d+(\.\d+)?\s*(--|–|-)\s*)?$", before):
        return "cross-reference"
    if after.startswith("\\linewidth"):
        return "table layout width"
    ot, cu = full.rfind("\\texttt{", 0, s), full.rfind("\\url{", 0, s)
    for o in (ot, cu):
        if o >= 0 and full.find("}", o) >= e:
            return "identifier inside \\texttt{} or \\url{}"
    if re.fullmatch(r"(19|20)\d\d", tok) and re.search(r"(al\.,?|et al\.)\s*$", before):
        return "publication year of a cited work"
    if tok == "95" and re.match(r"\\?%", after):
        return "confidence level"
    return None


def tokens(text, offset=0):
    """Every numeric token with its absolute [start, end) span. Tokens glued to a preceding letter, underscore or
    backslash (identifiers such as wav2vec2, LDC2021S04, \\section) are identifiers and are returned with kind
    'ident' so the checker can account for them explicitly."""
    out, used = [], set()
    for m in SCI.finditer(text):
        out.append({"kind": "sci", "tok": m.group(0), "s": offset + m.start(), "e": offset + m.end()})
        used.update(range(m.start(), m.end()))
    for m in BOUND.finditer(text):
        if m.start() in used:
            continue
        out.append({"kind": "bound", "tok": m.group(0), "s": offset + m.start(), "e": offset + m.end()})
        used.update(range(m.start(), m.end()))
    for m in NUM.finditer(text):
        if any(i in used for i in range(m.start(), m.end())):
            continue
        prev = text[max(0, m.start() - 1):m.start()]
        nxt = text[m.end():m.end() + 1]
        kind = "ident" if (prev.isalpha() or prev in "_\\" or nxt.isalpha() or nxt == "_") else "num"
        out.append({"kind": kind, "tok": m.group(0), "s": offset + m.start(), "e": offset + m.end()})
    return sorted(out, key=lambda t: t["s"])
