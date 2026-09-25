#!/usr/bin/env python3
"""v10: which aetiology groups enter the fixed-token analysis (Table 6) at each token budget.

The original fixed_token_dprime.py (Experiment A) admits a speaker at a budget when the speaker is in one of
the six analysis groups with a recorded severity label, and at least three of the five consonant contrasts have
>= budget tokens in both phone classes; the aetiology Kruskal-Wallis then uses groups with >= 5 speakers.
Group membership depends only on these token counts (the random subsampling changes the d-prime values, not
who enters), so it is computed here from the phone lists in the same embedding files, reusing the original
module's own constants and helpers (imported, not copied). The per-budget totals must equal the "Processed"
counts in the original's log on the same inputs; the script asserts this when the log is given.

Run on the DGX with DYSARTHRIA_BASE pointing at a directory holding results/track4/track4_master.csv
(the corrected master), results/track4/embeddings and config:
    DYSARTHRIA_BASE=<dir> python v10_fixed_token_groups.py <fixed_token_dprime.py> <out.json> [<its log>]
"""
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

orig_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("ftd", orig_path)
ftd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ftd)

rows = ftd.load_master()
meta = {f"{r['dataset']}__{r['speaker_id']}": r for r in rows}
pfs = {lang: ftd.load_phone_features(lang) for lang in set(ftd.LANG_MAP.values())}
counts = {}  # key -> list of (n_pos, n_neg) per contrast, or None for a contrast with no phone classes
aet_of = {}
for npz in sorted(ftd.EMB_DIR.glob("*.npz")):
    m = meta.get(npz.stem)
    if m is None:
        continue
    pf = pfs.get(ftd.LANG_MAP.get(m.get("language", "en"), "en"))
    if pf is None:
        continue
    if m.get("severity_label", "unknown") not in ftd.SEV_MAP or m.get("aetiology", "unknown") not in ftd.MAIN_AETS:
        continue
    phones = np.load(npz, allow_pickle=True)["phones"]
    c = []
    for contrast in ftd.CONS_CONTRASTS:
        pos, neg = ftd.get_phone_classes(pf, contrast)
        if not pos or not neg:
            c.append(None)
            continue
        c.append((sum(1 for p in phones if p in pos), sum(1 for p in phones if p in neg)))
    counts[npz.stem] = c
    aet_of[npz.stem] = m["aetiology"]

out = {"definition": __doc__, "original_sha256": hashlib.sha256(orig_path.read_bytes()).hexdigest(),
       "master_sha256": hashlib.sha256(ftd.MASTER.read_bytes()).hexdigest(), "budgets": {}}
for b in ftd.TOKEN_BUDGETS:
    ok = [k for k, c in counts.items() if sum(1 for x in c if x is not None and x[0] >= b and x[1] >= b) >= 3]
    grp = Counter(ftd.AET_SHORT[aet_of[k]] for k in ok)
    kw = {g: n for g, n in grp.items() if n >= 5}
    out["budgets"][b] = {"n_admitted": len(ok), "by_group": dict(sorted(grp.items())),
                         "kruskal_groups": dict(sorted(kw.items())), "kruskal_k": len(kw),
                         "kruskal_N": sum(kw.values()),
                         "groups_absent": sorted(set(ftd.AET_SHORT.values()) - set(kw))}
if len(sys.argv) > 3:
    log = Path(sys.argv[3]).read_text(encoding="utf-8")
    processed = {int(b): int(n) for b, n in
                 re.findall(r"--- Token budget: (\d+) per class ---\n\s+Processed: (\d+)", log)}
    out["log_sha256"] = hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
    out["log_processed"] = processed
    for b, v in out["budgets"].items():
        assert v["n_admitted"] == processed[b], (b, v["n_admitted"], processed[b])
    out["totals_match_log"] = True
out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({b: (v["n_admitted"], v["kruskal_k"], v["kruskal_N"], v["groups_absent"])
                  for b, v in out["budgets"].items()}))
