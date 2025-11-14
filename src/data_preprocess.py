import os
import json
from pathlib import Path
import pandas as pd

def _norm(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

def _find_col(row, keys):
    for k in keys:
        if k in row:
            return row[k]
    return None

def _canonical_row(row):
    dbname = _find_col(row, ["数据库名", "数据库名dbname", "dbname"])
    tablename = _find_col(row, ["数据表名", "数据表名tablename", "tablename", "表名"])
    fieldname = _find_col(row, ["字段名", "fieldname", "字段"])
    fielddesc = _find_col(row, ["字段描述", "fielddesc", "描述"])
    example = _find_col(row, ["数据示例", "示例", "example"])
    datatype = _find_col(row, ["数据类型", "datatype", "类型"])
    nullok = _find_col(row, ["是否可为空", "NULL", "可空"])
    enumv = _find_col(row, ["枚举值", "enum", "枚举"])
    cls = _find_col(row, ["分类", "category", "业务分类"])
    return {
        "dbname": _norm(dbname),
        "tablename": _norm(tablename),
        "fieldname": _norm(fieldname),
        "fielddesc": _norm(fielddesc),
        "example": _norm(example),
        "datatype": _norm(datatype),
        "null": _norm(nullok),
        "enum": _norm(enumv),
        "classification": _norm(cls)
    }

def _compose_text(x):
    parts = []
    if x.get("tablename"):
        parts.append(x["tablename"])
    if x.get("fieldname"):
        parts.append(x["fieldname"])
    if x.get("fielddesc"):
        parts.append("描述:" + x["fielddesc"])
    if x.get("example"):
        parts.append("示例:" + x["example"])
    if x.get("datatype"):
        parts.append("类型:" + x["datatype"])
    if x.get("null"):
        parts.append("可空:" + x["null"])
    if x.get("enum"):
        parts.append("枚举:" + x["enum"])
    return " | ".join([p for p in parts if p])

def _load_taxonomy(tax_path):
    with open(tax_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_reverse_maps(tax):
    domain_to_sub1 = {}
    sub1_to_sub2 = {}
    sub2_to_sub1_domain = {}
    for d, val in tax["domains"].items():
        submap = val.get("一级子类", {})
        domain_to_sub1[d] = list(submap.keys())
        for s1, s2list in submap.items():
            sub1_to_sub2[s1] = s2list
            for s2 in s2list:
                sub2_to_sub1_domain[s2] = (s1, d)
    return domain_to_sub1, sub1_to_sub2, sub2_to_sub1_domain

def _parse_classification(raw, tax):
    if not raw:
        return None, None, None
    raw = raw.replace(" ", "").replace("-", "-")
    parts = [p for p in raw.replace("\\", "/").split("/") if p]
    domain_to_sub1, sub1_to_sub2, sub2_to_sub1_domain = _build_reverse_maps(tax)
    if len(parts) == 3:
        d, s1, s2 = parts
        return d, s1, s2
    if len(parts) == 2:
        a, b = parts
        if a in tax["domains"]:
            return a, b, None
        if b in sub1_to_sub2:
            for d, subs in domain_to_sub1.items():
                if b in subs:
                    return d, b, None
        if b in sub2_to_sub1_domain:
            s1, d = sub2_to_sub1_domain[b]
            return d, s1, b
        return None, None, None
    if len(parts) == 1:
        a = parts[0]
        if a in tax["domains"]:
            return a, None, None
        if a in sub1_to_sub2:
            for d, subs in domain_to_sub1.items():
                if a in subs:
                    return d, a, None
        if a in sub2_to_sub1_domain:
            s1, d = sub2_to_sub1_domain[a]
            return d, s1, a
    return None, None, None

def build_dataset(input_root="data_", tax_path="labels/taxonomy.json", output_path="outputs/dataset.jsonl"):
    tax = _load_taxonomy(tax_path)
    root = Path(input_root)
    rows = []
    for p in root.rglob("*.xlsx"):
        try:
            df = pd.read_excel(p)
        except Exception:
            continue
        for _, r in df.iterrows():
            x = _canonical_row(r)
            text = _compose_text(x)
            d, s1, s2 = _parse_classification(x.get("classification"), tax)
            rows.append({
                "source": str(p),
                "dbname": x.get("dbname"),
                "tablename": x.get("tablename"),
                "fieldname": x.get("fieldname"),
                "text": text,
                "domain": d or "",
                "sub1": s1 or "",
                "sub2": s2 or ""
            })
    outdir = Path(output_path).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return output_path

if __name__ == "__main__":
    build_dataset()