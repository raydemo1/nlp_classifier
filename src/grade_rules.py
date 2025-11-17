import re
import json
from pathlib import Path

def load_grade_taxonomy(path="labels/grade_taxonomy.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _regex_hits(text):
    hits = 0
    patterns = [
        r"\b1[3-9]\d{9}\b",
        r"\b\d{15}(\d{2}[0-9X])?\b",
        r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+",
        r"[0-9]{16,19}"
    ]
    for pat in patterns:
        if re.search(pat, text):
            hits += 1
    return hits

def score(text, meta=None, tax_path="labels/grade_taxonomy.json"):
    tx = load_grade_taxonomy(tax_path)
    grades = list(tx.get("grades", {}).keys())
    kw = {g: set(tx["grades"][g].get("keywords", [])) for g in grades}
    scores = {g: 0.0 for g in grades}
    t = text or ""
    t = t.lower()
    rh = _regex_hits(t)
    for g in grades:
        for k in kw[g]:
            if k.lower() in t:
                scores[g] += 1.0
    if rh > 0:
        scores[grades[0]] += 1.5
    if meta:
        dt = (meta.get("datatype") or "").lower()
        if any(x in dt for x in ["id", "card", "phone", "mobile"]):
            scores[grades[0]] += 1.0
        nullok = (meta.get("null") or "").lower()
        if nullok in ["否", "no", "false"]:
            scores[grades[0]] += 0.5
    total = sum(scores.values())
    if total <= 0:
        return [(g, 1.0/len(grades)) for g in grades]
    out = [(g, scores[g]/total) for g in grades]
    out.sort(key=lambda x: -x[1])
    return out

