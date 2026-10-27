"""Build paper/latex/main.bib from paper/verified_refs.tsv using arXiv metadata (titles and authors are taken
from arXiv, never typed by hand). Entries whose arXiv record cannot be found are written with author={TODO}
so they are obvious in the PDF.

  python3 paper/make_bib.py      (standard library only)
Also writes paper/latex/cite_keys.tsv: title -> bibtex key.
"""
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
API = "http://export.arxiv.org/api/query?"
NS = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch(query):
    with urllib.request.urlopen(API + query, timeout=60) as r:
        return ET.fromstring(r.read())


def entries(root):
    out = []
    for e in root.findall("a:entry", NS):
        aid = ARXIV_ID.search(e.find("a:id", NS).text or "")
        out.append({"id": aid.group(1) if aid else "",
                    "title": " ".join((e.find("a:title", NS).text or "").split()),
                    "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                    "year": (e.find("a:published", NS).text or "")[:4]})
    return out


def main():
    rows = [line.rstrip("\n").split("\t") for line in open(os.path.join(HERE, "verified_refs.tsv"))]
    rows = [r for r in rows if len(r) >= 4]
    ids = {i: ARXIV_ID.search(r[3]).group(1) for i, r in enumerate(rows) if ARXIV_ID.search(r[3])}
    meta = {}
    id_list = sorted(set(ids.values()))
    for k in range(0, len(id_list), 50):
        for e in entries(fetch("id_list=" + ",".join(id_list[k:k + 50]) + "&max_results=50")):
            meta[e["id"]] = e
        time.sleep(3)
    for i, r in enumerate(rows):  # no arXiv id in the URL: search by exact title
        if i in ids:
            continue
        q = urllib.parse.quote(f'ti:"{r[1][:120]}"')
        try:
            cands = entries(fetch(f"search_query={q}&max_results=3"))
        except Exception:
            cands = []
        hit = next((c for c in cands if norm(c["title"])[:40] == norm(r[1])[:40]), None)
        if hit:
            ids[i] = hit["id"]
            meta[hit["id"]] = hit
        time.sleep(3)

    os.makedirs(os.path.join(HERE, "latex"), exist_ok=True)
    bib, keys, used = [], [], set()
    for i, (_, title, venue, url) in enumerate(rows):
        m = meta.get(ids.get(i, ""), None)
        authors = m["authors"] if m else []
        year = (m["year"] if m else "") or (re.search(r"(20\d\d)", venue).group(1) if re.search(r"(20\d\d)", venue) else "")
        last = re.sub(r"[^A-Za-z]", "", authors[0].split()[-1]).lower() if authors else "anon"
        word = next((w for w in re.findall(r"[A-Za-z]+", title) if len(w) > 3), "paper").lower()
        key = f"{last}{year}{word}"
        while key in used:
            key += "x"
        used.add(key)
        venue_clean = re.sub(r"\s*\(.*?\)", "", venue).strip()
        fields = [f"  title = {{{{{title if not m else m['title']}}}}}",
                  f"  author = {{{' and '.join(authors) if authors else 'TODO'}}}",
                  f"  year = {{{year or 'TODO'}}}",
                  f"  note = {{{venue_clean}" + (f", arXiv:{m['id']}" if m else "") + "}",
                  f"  url = {{{url}}}"]
        bib.append(f"@misc{{{key},\n" + ",\n".join(fields) + "\n}\n")
        keys.append(f"{key}\t{title}")
    open(os.path.join(HERE, "latex", "main.bib"), "w").write("\n".join(bib))
    open(os.path.join(HERE, "latex", "cite_keys.tsv"), "w").write("\n".join(keys) + "\n")
    print(f"{len(bib)} entries, {sum(1 for i in range(len(rows)) if ids.get(i) in meta)} with arXiv metadata")


if __name__ == "__main__":
    main()
