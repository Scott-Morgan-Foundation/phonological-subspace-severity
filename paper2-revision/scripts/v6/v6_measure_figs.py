"""v6: measure placed text sizes of Figures 1-5 in the compiled v6 PDF and render the figure pages
and the Table 5 page at 150 dpi. Sizes are PyMuPDF span sizes in placed points. The figure page is
the caption page or the page before it with the most vector drawings; figure text = spans on that
page smaller than 11 pt (body text is 12 pt), outside the left line-number margin, and above the
caption when the caption shares the page; lineno margin numbers (5.98 pt digits) are excluded."""
from pathlib import Path

import fitz

from common import REV, write

PDF = REV / "paper2-csl_v6.pdf"
REN = REV / "results" / "v6" / "renders"
REN.mkdir(parents=True, exist_ok=True)
doc = fitz.open(PDF)


def btext(b):
    return "".join(s["text"] for l in b["lines"] for s in l["spans"])


out = {"pdf": PDF.name, "pages": doc.page_count, "figures": {}}
for k in range(1, 6):
    cap = f"Figure {k}."
    cps = [i for i, p in enumerate(doc) if any(b["type"] == 0 and btext(b).startswith(cap)
                                                 for b in p.get_text("dict")["blocks"])]
    cp = cps[0]
    pno = max([i for i in (cp - 1, cp) if i >= 0], key=lambda i: len(doc[i].get_drawings()))
    pg = doc[pno]
    blocks = pg.get_text("dict")["blocks"]
    cap_y = min(b["bbox"][1] for b in blocks if b["type"] == 0 and btext(b).startswith(cap)) if pno == cp else None
    sizes = []
    for b in blocks:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                if not s["text"].strip() or s["size"] >= 11 or s["bbox"][0] < 60:
                    continue
                if s["text"].strip().isdigit() and abs(s["size"] - 5.98) < 0.01:   # lineno margin numbers
                    continue
                if cap_y is not None and s["bbox"][1] >= cap_y - 1:
                    continue
                sizes.append((round(s["size"], 2), s["text"].strip()[:30]))
    png = REN / f"fig{k}_page{pno + 1}.png"
    pg.get_pixmap(dpi=150).save(png)
    mn = min(sizes)[0] if sizes else None
    out["figures"][f"fig{k}"] = {"page": pno + 1, "caption_page": cp + 1, "min_pt": mn,
                                 "sizes": sorted({s for s, _ in sizes}),
                                 "smallest_examples": sorted(sizes)[:5], "render": png.name}
t5 = [i for i, p in enumerate(doc) if "Table 5." in p.get_text()][0]
doc[t5].get_pixmap(dpi=150).save(REN / f"table5_page{t5 + 1}.png")
out["table5_page"] = t5 + 1
write("v6_fig_text_sizes.json", out, [PDF])
for k, v in out["figures"].items():
    print(k, v["page"], v["min_pt"], v["sizes"], v["smallest_examples"][:3])
print("table5 page", t5 + 1)
