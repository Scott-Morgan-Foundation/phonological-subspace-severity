#!/usr/bin/env python3
"""Freeze the high-risk bindings: for each spec item (scripts/v10/hr_spec.py) find the printed value at the token's
place in the v10 manuscript / Highlights, and record the value together with its surrounding text (30 characters
before, 15 after, within the region). The frozen contexts are what scripts/v10/check_consistency.py later requires to
be present with the value re-derived from the saved output, so a changed number fails the check.
Run once after the text is final; it refuses to build if any spec item does not match the text or the saved value.
Output: scripts/v10/hr_bindings.json"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hr_regions as H          # noqa: E402
from hr_resolve import resolve  # noqa: E402
from hr_spec import SPEC        # noqa: E402

_s = importlib.util.spec_from_file_location("fn10", HERE / "format_numbers.py")
FN = importlib.util.module_from_spec(_s)
_s.loader.exec_module(FN)
F = FN.bind()

BEFORE, AFTER = 30, 15


def build():
    tex = H.TEX.read_text(encoding="utf-8")
    hl = H.HL.read_text(encoding="utf-8")
    R = H.regions(tex)
    R["highlights"] = (0, len(hl))
    out, errors = [], []
    for region, items in SPEC.items():
        full = hl if region == "highlights" else tex
        a, b = R[region]
        toks = [t for t in H.tokens(full[a:b], a) if not H.structural(full, t)]
        spans, i = [], 0
        for t in toks:
            if any(s < t["e"] and t["s"] < e for s, e in spans):
                continue
            if i >= len(items):
                errors.append(f"{region}: token {t['tok']!r} at {t['s']} has no spec item")
                continue
            want, ref = items[i]
            i += 1
            if want is not None and t["tok"] != want:
                errors.append(f"{region}: spec expects {want!r}, text has {t['tok']!r} at {t['s']}")
                continue
            val, src = resolve(ref, F, FN.fmt)
            hits = []
            j = full.find(val, max(a, t["s"] - len(val)))
            while 0 <= j < min(b, t["e"] + 1):
                if j < t["e"] and t["s"] < j + len(val) and j + len(val) <= b:
                    hits.append(j)
                j = full.find(val, j + 1)
            if not hits:
                errors.append(f"{region}: value {val!r} ({ref}) not at token {t['tok']!r} ({full[t['s']-20:t['e']+10]!r})")
                continue
            j = hits[0]
            spans.append((j, j + len(val)))
            out.append({"region": region, "ref": ref, "src": src, "value_at_build": val,
                        "before": full[max(a, j - BEFORE):j], "after": full[j + len(val):min(b, j + len(val) + AFTER)]})
        if i != len(items):
            errors.append(f"{region}: {len(items) - i} spec items unused")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    (HERE / "hr_bindings.json").write_text(json.dumps({"definition": __doc__, "bindings": out}, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"{len(out)} bindings frozen")


if __name__ == "__main__":
    build()
