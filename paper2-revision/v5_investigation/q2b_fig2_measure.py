"""Phase A Q2b: measure placed text size of Figure 2 in the compiled v4 PDF.

Method: locate the page whose text contains the Figure 2 caption, take the largest image/xobject
bbox on that page as the figure, list text spans falling inside it with their rendered font size
(PyMuPDF reports span size in placed points, i.e. after the \\includegraphics scaling), and render
the page to PNG at 150 dpi for visual inspection. Also reports the source-figure native sizes
(figures/fig2_aetiology_radars.pdf) and the linear scale factor between native and placed width.
"""
import json
from pathlib import Path
import fitz

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
pdf = fitz.open(BASE / "paper2-csl_v4.pdf")

page_no = None
for i, p in enumerate(pdf):
    if "Pairwise aetiology comparison" in p.get_text():
        page_no = i
        break
page = pdf[page_no]

# figure region: union of drawings/images; prefer image/xobject blocks
blocks = page.get_text("dict")["blocks"]
img_rects = [fitz.Rect(b["bbox"]) for b in blocks if b["type"] == 1]
draw_rects = [fitz.Rect(d["rect"]) for d in page.get_drawings()]
region = None
for r in img_rects + draw_rects:
    region = r if region is None else region | r
cap_y = None
for b in blocks:
    if b["type"] == 0:
        t = "".join(s["text"] for l in b["lines"] for s in l["spans"])
        if "Pairwise aetiology comparison" in t:
            cap_y = b["bbox"][1]
if region is not None and cap_y is not None and region.y1 > cap_y:
    region.y1 = cap_y

spans = []
for b in blocks:
    if b["type"] != 0:
        continue
    for l in b["lines"]:
        for s in l["spans"]:
            r = fitz.Rect(s["bbox"])
            if region is not None and region.contains(r) and s["text"].strip():
                spans.append({"text": s["text"].strip(), "size_pt": round(s["size"], 2), "bbox": [round(x, 1) for x in s["bbox"]]})

src = fitz.open(BASE / "figures" / "fig2_aetiology_radars.pdf")
sp = src[0]
native = []
for b in sp.get_text("dict")["blocks"]:
    if b["type"] == 0:
        for l in b["lines"]:
            for s in l["spans"]:
                if s["text"].strip():
                    native.append({"text": s["text"].strip(), "size_pt": round(s["size"], 2)})
native_w_in = sp.rect.width / 72
placed_w_in = 4.99435
scale = placed_w_in / native_w_in

pix = page.get_pixmap(dpi=150)
pix.save(OUT / "fig2_v4_page.png")
clip = region if region is not None else page.rect
page.get_pixmap(dpi=300, clip=clip).save(OUT / "fig2_v4_figure_crop_300dpi.png")

sizes = sorted({s["size_pt"] for s in spans})
res = {"pdf_page_index0": page_no, "pdf_page_label": page_no + 1, "figure_region_pt": [round(x, 1) for x in region] if region else None,
       "placed_span_sizes_pt": sizes, "placed_spans": spans,
       "native_figure_width_in": round(native_w_in, 3), "placed_width_in": placed_w_in, "linear_scale": round(scale, 4),
       "native_span_sizes_pt": sorted({n["size_pt"] for n in native}),
       "native_x_scale_implied_placed_sizes_pt": sorted({round(n["size_pt"] * scale, 2) for n in native}),
       "native_spans": native}
(OUT / "q2b_fig2_measure.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("page", page_no + 1, "region", res["figure_region_pt"])
print("placed span sizes", sizes, "n spans", len(spans))
print("native sizes", res["native_span_sizes_pt"], "scale", res["linear_scale"], "-> placed", res["native_x_scale_implied_placed_sizes_pt"])
for s in native:
    print("  native", s["size_pt"], "->", round(s["size_pt"] * scale, 2), s["text"][:40])
