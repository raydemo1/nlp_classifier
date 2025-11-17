import os
import json
from pathlib import Path
import pandas as pd


def _norm(v):
    try:
        import pandas as _pd

        if isinstance(v, _pd.Series):
            for vv in v.values:
                if not _pd.isna(vv):
                    return str(vv).strip()
            return ""
    except Exception:
        pass
    if pd.isna(v):
        return ""
    return str(v).strip()


def _find_col(row, keys):
    for k in keys:
        if k in row:
            val = row[k]
            try:
                import pandas as _pd

                if isinstance(val, _pd.Series):
                    for vv in val.values:
                        if not _pd.isna(vv):
                            return vv
                    return None
            except Exception:
                pass
            return val
    return None


def _normalize_header_key(name: str):
    if not name:
        return ""
    s = str(name).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("（", "(").replace("）", ")")
    if "数据库名" in s or "dbname" in s:
        return "dbname"
    if "数据表名" in s or "tablename" in s or s == "表名":
        return "tablename"
    if "字段名" in s or s == "字段" or s == "fieldname" or s == "time":
        return "fieldname"
    if "字段描述" in s or "fielddesc" in s or s == "描述":
        return "fielddesc"
    if "数据类型" in s or "datatype" in s or s == "类型":
        return "datatype"
    if "最大长度" in s or "length" in s:
        return "length"
    if "可为空" in s or "null" in s:
        return "null"
    if "主键" in s or "外键" in s or "pk" in s or "fk" in s:
        return "keyflag"
    if "枚举" in s or "enum" in s:
        return "enum"
    if "数据示例" in s or s == "示例" or "example" in s:
        return "example"
    if "分类" in s:
        return "classification"
    if "分级" in s or "grade" in s:
        return "grade"
    return s


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        rename_map[c] = _normalize_header_key(c) or c
    return df.rename(columns=rename_map)


def _canonical_row(row):
    dbname = _find_col(row, ["dbname", "数据库名dbname"])
    tablename = _find_col(row, ["tablename", "数据表名tablename", "表名"])
    fieldname = _find_col(row, ["fieldname", "time", "字段名"])
    fielddesc = _find_col(row, ["fielddesc", "字段描述fielddesc", "字段描述"])
    example = _find_col(row, ["example", "数据示例", "示例"])
    datatype = _find_col(row, ["datatype", "数据类型datatype", "数据类型"])
    nullok = _find_col(row, ["null", "是否可为空NULL", "可空", "NULL"])
    cls = _find_col(
        row,
        [
            "classification",
            "分类",
            "分类（包括数据的业务归属分类和业务扩展类，参考《铁路数据分类分级指南》的分类要求）",
        ],
    )
    grade = _find_col(row, ["grade", "分级"])
    return {
        "dbname": _norm(dbname),
        "tablename": _norm(tablename),
        "fieldname": _norm(fieldname),
        "fielddesc": _norm(fielddesc),
        "example": _norm(example),
        "datatype": _norm(datatype),
        "null": _norm(nullok),
        "classification": _norm(cls),
        "grade": _norm(grade),
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
    if x.get("grade"):
        parts.append("分级:" + x["grade"])
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


def build_dataset(
    input_root="data_",
    tax_path="labels/taxonomy.json",
    output_path="outputs/dataset.jsonl",
):
    tax = _load_taxonomy(tax_path)
    candidate_roots = []
    r = Path(input_root)
    if r.exists():
        candidate_roots.append(r)
    r2 = Path("data")
    if r2.exists() and r2 not in candidate_roots:
        candidate_roots.append(r2)
    # fallback: 直接搜索工作目录
    candidate_roots.append(Path("."))
    rows = []
    for root in candidate_roots:
        for p in root.rglob("*.xlsx"):
            try:
                sheets = pd.read_excel(p, sheet_name=None)
            except Exception:
                continue
            for sname, df in sheets.items():
                if df is None or df.empty:
                    continue
                df = _normalize_df(df)
                cols = set(df.columns)
                if not (
                    "fieldname" in cols
                    or "fielddesc" in cols
                    or "classification" in cols
                ):
                    continue
                for _, r in df.iterrows():
                    x = _canonical_row(r)
                    if not x.get("fieldname") and not x.get("fielddesc"):
                        continue

                    # 如果分类为空，删除这条数据
                    if not x.get("classification") or x.get("classification").strip() == "":
                        continue

                    # 归一化分级到 S1/S2/S3（S1最低重要性，S3最高重要性）
                    def _normalize_grade_s(raw: str) -> str:
                        if not raw:
                            return ""
                        s = str(raw).strip().upper()
                        s = s.replace("（", "(").replace("）", ")")
                        s_n = s.replace(" ", "")
                        if "S1" in s_n:
                            return "S1"
                        if "S2" in s_n:
                            return "S2"
                        if "S3" in s_n:
                            return "S3"
                        if "S4" in s_n:
                            return "S4"
                        if any(k in s_n for k in ["G-1", "G1"]) or ("一级" in s):
                            return "S1"
                        if any(k in s_n for k in ["G-2", "G2"]) or ("二级" in s):
                            return "S2"
                        if any(k in s_n for k in ["G-3", "G3"]) or ("三级" in s):
                            return "S3"
                        if any(k in s_n for k in ["G-4", "G4"]) or ("四级" in s):
                            return "S4"

                    x["grade"] = _normalize_grade_s(x.get("grade", ""))
                    text = _compose_text(x)
                    d, s1, s2 = _parse_classification(x.get("classification"), tax)
                    rows.append(
                        {
                            "source": str(p),
                            "dbname": x.get("dbname"),
                            "tablename": x.get("tablename"),
                            "fieldname": x.get("fieldname"),
                            "text": text,
                            "domain": d or "",
                            "sub1": s1 or "",
                            "sub2": s2 or "",
                            "grade": x.get("grade", ""),
                        }
                    )
    outdir = Path(output_path).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return output_path


if __name__ == "__main__":
    build_dataset()
