#!/usr/bin/env python3
"""v8: first-occurrence order of numeric citations [n] in the v8 manuscript body (from the Introduction heading to the reference list; the abstract has no numeric citations). Writes results/v8/v8_cite_order.json with the order and every position where the
first-cited number differs from strictly sequential numbering."""
import json
import re
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
tex = (REV / "paper2-csl_v8.tex").read_text(encoding="utf-8")
cut = tex.find("\\section*{References}")
start = tex.find("\\section{Introduction}")
body = tex[start:cut] if cut >= 0 else tex[start:]
order = []
for m in re.finditer(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]", body):
    for part in re.split(r"\s*,\s*", m.group(1)):
        rng = re.split(r"\s*[\-–]\s*", part)
        nums = range(int(rng[0]), int(rng[-1]) + 1) if len(rng) == 2 else [int(rng[0])]
        for n in nums:
            if n not in order:
                order.append(n)
diff = [[i + 1, n] for i, n in enumerate(order) if n != i + 1]
res = {"first_occurrence_order": order, "n_distinct": len(order),
       "positions_differing_from_sequential": diff,
       "strictly_sequential": order == list(range(1, len(order) + 1))}
(REV / "results" / "v8" / "v8_cite_order.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(order[:12], "... n", len(order), "| differing positions:", diff)
