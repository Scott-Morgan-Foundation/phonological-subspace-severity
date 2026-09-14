import io
import re

s = io.open("paper2-csl_v2.tex", encoding="utf-8").read()
refs_i = s.find("\\section*{References}")
body, refs = s[:refs_i], s[refs_i:]

items = re.findall(r"\\item (.{0,100})", refs)
print(f"{len(items)} reference items")
for i, t in enumerate(items, 1):
    print(f"[{i}] {t[:95]}")

CITE = re.compile(r"(?:\{\[\}|\[)(\d{1,2}(?:\s*,\s*\d{1,2})*)(?:\{\]\}|\])")
order = []
for m in CITE.finditer(body):
    for n in re.split(r"\s*,\s*", m.group(1)):
        n = int(n)
        if n not in order:
            order.append(n)
print("\nfirst-occurrence order:", order)
uncited = [i for i in range(1, len(items) + 1) if i not in order]
print("uncited items:", uncited)
print("cited beyond list:", [n for n in order if n > len(items)])
for n in (39, 40, 47):
    m = re.search(r"[^\n]{0,130}(?:\{\[\}|\[)" + str(n) + r"(?:\{\]\}|\])[^\n]{0,130}", body)
    print(f"\n[{n}] first ctx:", (m.group(0)[:240] if m else "??"))
