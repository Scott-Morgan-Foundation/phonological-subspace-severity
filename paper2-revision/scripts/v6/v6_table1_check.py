"""v6: compare every speaker count printed in Table 1 of paper2-csl_v6.tex with the corrected master.
Language totals are compared with master rows per language; each dataset count is compared with the
master rows of that dataset in the row's language (datasets spanning languages are split by language,
as the Table 1 caption states) and with the dataset total."""
import re

import pandas as pd

from common import MASTER, REV, write

TEX = REV / "paper2-csl_v6.tex"
t = TEX.read_text(encoding="utf-8")
blk = t[t.index("\\textbf{Language (n)}"):]
blk = blk[blk.index("\\midrule") + 8:blk.index("\\bottomrule")]
blk = blk.replace("\\_", "_").replace("(collected by authors)", "").replace("(LDC2021S04)", "")
m = pd.read_csv(MASTER)
LANG = {"English": "en", "Slovak": "sk", "Portuguese": "pt", "Dutch": "nl", "Spanish": "es", "Italian": "it",
        "Mandarin": "zh", "Tamil": "ta", "German": "de", "Hungarian": "hu", "French": "fr", "Swahili": "sw"}
DS = {"SAP": "SAP", "LibriSpeech": "LibriSpeech_English", "TORGO": "TORGO", "UA-Speech": "UASPEECH",
      "UA-Speech controls": "UASPEECH_control", "EWA-DB": "EWA-DB", "AVFAD": "AVFAD", "COPAS": "COPAS",
      "Domotica": "Domotica", "CHASING": "CHASING", "TreasureHunters1": "TreasureHunters1", "Neurovoz": "Neurovoz",
      "PC-GITA": "PC-GITA", "IPVS": "IPVS", "EasyCall": "EasyCall", "MDSC": "MDSC", "CDSD": "CDSD",
      "SLR65": "SLR65_Tamil", "SSNCE": "SSNCE_Tamil", "SVD": "SVD", "YouTube German": "YouTube_German",
      "Hungarian Dys.": "Hungarian_Dysarthria", "CV_Hungarian": "CV_Hungarian", "Hungarian_HC": "Hungarian_HC",
      "YouTube French": "YouTube_French", "CDLI Kenyan": "CDLI_Kenyan_Swahili"}
rows, lang = [], None
for line in blk.split("\\\\"):
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 2:
        continue
    lm = re.match(r"([A-Za-z]+) \((\d[\d,]*)\)", cells[0])
    if lm:
        lang = lm.group(1)
        n = int(lm.group(2).replace(",", ""))
        rows.append({"kind": "language", "name": lang, "table": n, "master": int((m.language == LANG[lang]).sum())})
    for name, n in re.findall(r"([A-Za-z][A-Za-z0-9_ .\-]*?)\s*(?:\[\d+\])?\s*\((\d[\d,]*)\)", cells[1]):
        name, n = name.strip().rstrip(","), int(n.replace(",", ""))
        d = DS.get(name)
        tot = int((m.dataset == d).sum()) if d else None
        inlang = int(((m.dataset == d) & (m.language == LANG[lang])).sum()) if d else None
        rows.append({"kind": "dataset", "name": name, "language": lang, "table": n,
                     "master_total": tot, "master_in_language": inlang})
for r in rows:
    r["match"] = r["table"] == r["master"] if r["kind"] == "language" else r["table"] in (r["master_total"], r["master_in_language"])
out = {"definition": __doc__, "rows": rows, "n_rows": len(rows), "n_match": sum(r["match"] for r in rows),
       "mismatches": [r for r in rows if not r["match"]],
       "master_datasets_not_in_table": sorted(set(m.dataset) - {DS[r["name"]] for r in rows if r["kind"] == "dataset" and r["name"] in DS})}
write("v6_table1_check.json", out, [MASTER, TEX])
print(out["n_rows"], out["n_match"], out["mismatches"], out["master_datasets_not_in_table"])
