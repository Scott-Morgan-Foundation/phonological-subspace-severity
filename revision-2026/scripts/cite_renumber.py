# R1-W4: sequential-by-first-occurrence renumbering + author-year -> numbered cites.
# Usage: python cite_renumber.py [apply]
import io
import re
import sys

APPLY = len(sys.argv) > 1 and sys.argv[1] in ("apply", "apply2")
SKIP_STEP1 = len(sys.argv) > 1 and sys.argv[1] == "apply2"
p = "paper2-csl_v2.tex"
s = io.open(p, encoding="utf-8").read()
fails = []


def rep(old, new, cnt=1):
    global s
    c = s.count(old)
    if c != cnt:
        fails.append((old[:60], c, cnt))
        return
    s = s.replace(old, new)


# --- step 1: fix wrong pointers + number the author-year-only citations (current numbers) ---
if SKIP_STEP1:
    def rep(old, new, cnt=1):
        pass
rep("COPAS [47]", "COPAS [24]", 2)                     # item 47 is Westfall&Yarkoni; COPAS corpus = Middag et al. [24]
rep("(MFA; McAuliffe et al., 2017)", "(MFA; McAuliffe et al., 2017 [45])")
rep("recommendation of Westfall and Yarkoni (2016) to report", "recommendation of Westfall and Yarkoni (2016) [47] to report")
rep("recommendation of Westfall and Yarkoni (2016) for confounds", "recommendation of Westfall and Yarkoni (2016) [47] for confounds")
rep("adapted from Cohen (1988).", "adapted from Cohen (1988) [43].")
rep("the COPAS manual (Van Nuffelen et al.)", "the COPAS manual (Van Nuffelen et al. [46])")
rep("WavLM-base was pre-trained on 960 hours of LibriSpeech", "WavLM-base [42] was pre-trained on 960 hours of LibriSpeech")
rep("SSNCE Tamil through the Linguistic Data Consortium (LDC2021S04)", "SSNCE Tamil [44] through the Linguistic Data Consortium (LDC2021S04)")

# context check for [24]
m = re.search(r"[^\n]{0,100}(?:\{\[\}|\[)24(?:\{\]\}|\])[^\n]{0,100}", s)
print("[24] first ctx now:", m.group(0)[:200] if m else "??")

# --- step 2: split refs, compute order, renumber ---
refs_i = s.find("\\section*{References}")
body, refs = s[:refs_i], s[refs_i:]

items = re.split(r"\n\\item ", refs)
head = items[0]
tail_m = re.search(r"(\n\\end\{enumerate\}.*)$", items[-1], re.S)
items[-1] = items[-1][: tail_m.start()]
tail = tail_m.group(1)
entries = items[1:]
print(f"{len(entries)} entries parsed")

CITE = re.compile(r"(\{\[\}|\[)(\d{1,2}(?:\s*,\s*\d{1,2})*)(\{\]\}|\])")
order = []
for m in CITE.finditer(body):
    for n in re.split(r"\s*,\s*", m.group(2)):
        n = int(n)
        if n not in order:
            order.append(n)
print("order:", order)
assert sorted(order) == list(range(1, len(entries) + 1)), (
    f"cited set != items: missing {set(range(1,len(entries)+1))-set(order)}, extra {set(order)-set(range(1,len(entries)+1))}")
mapping = {old: i + 1 for i, old in enumerate(order)}
changed = {o: n for o, n in mapping.items() if o != n}
print("renumbered (old->new):", changed)


def sub(m):
    nums = [mapping[int(n)] for n in re.split(r"\s*,\s*", m.group(2))]
    return m.group(1) + ", ".join(str(n) for n in nums) + m.group(3)


new_body = CITE.sub(sub, body)
new_entries = [entries[old - 1] for old in order]
new_refs = head + "".join("\n\\item " + e for e in new_entries) + tail

out = new_body + new_refs
# sanity: same length class, same number of items
assert out.count("\\item ") == s.count("\\item "), "item count changed"
print("fails:", fails if fails else "NONE")
if APPLY and not fails:
    io.open(p, "w", encoding="utf-8", newline="").write(out)
    print("APPLIED")
else:
    print("dry run only" if not APPLY else "NOT applied due to fails")
