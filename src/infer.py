import os
import json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

def load_taxonomy(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_thresholds(path):
    defaults = {"domain": 0.45, "sub1": 0.40, "sub2": 0.35}
    p = Path(path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k in defaults:
                if k in cfg:
                    defaults[k] = float(cfg[k])
        except Exception:
            pass
    return defaults

def load_priors(path):
    p = Path(path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def build_text(sample):
    parts = []
    if sample.get("tablename"):
        parts.append(sample["tablename"])
    if sample.get("fieldname"):
        parts.append(sample["fieldname"])
    if sample.get("fielddesc"):
        parts.append("描述:" + sample["fielddesc"])
    if sample.get("example"):
        parts.append("示例:" + sample["example"])
    if sample.get("datatype"):
        parts.append("类型:" + sample["datatype"])
    if sample.get("null"):
        parts.append("可空:" + sample["null"])
    if sample.get("enum"):
        parts.append("枚举:" + sample["enum"])
    return " | ".join([p for p in parts if p])

def load_model_or_none(model_dir, device):
    try:
        if Path(model_dir).exists():
            return SentenceTransformer(str(model_dir), device=device)
    except Exception:
        return None
    return None

def softmax(x):
    x = np.array(x)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def encode(model, texts):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def predict_with_labels(model, labels_json, text):
    labels = list(labels_json.keys())
    idx = {labels_json[k]: k for k in labels_json}
    emb = encode(model, [text])[0]
    protos = encode(model, labels)
    sims = protos @ emb
    probs = (sims + 1) / 2
    probs = probs / probs.sum()
    order = np.argsort(-probs)
    top = [(labels[i], float(probs[i])) for i in order]
    return top

def predict_by_prototype(model, candidates, text):
    emb = encode(model, [text])[0]
    protos = encode(model, candidates)
    sims = protos @ emb
    probs = (sims + 1) / 2
    probs = probs / probs.sum()
    order = np.argsort(-probs)
    top = [(candidates[i], float(probs[i])) for i in order]
    return top

def apply_priors(top_list, priors_layer, text):
    if not priors_layer:
        return top_list
    scores = {l: s for l, s in top_list}
    for label, cfg in priors_layer.items():
        terms = cfg.get("terms", [])
        w = float(cfg.get("weight", 0.0))
        if w <= 0:
            continue
        hit = 0
        for t in terms:
            if t and t in text:
                hit += 1
        if hit > 0 and label in scores:
            scores[label] = scores[label] * (1.0 + w * hit)
    total = sum(scores.values())
    if total <= 0:
        return top_list
    norm = [(l, scores[l] / total) for l, _ in top_list]
    norm.sort(key=lambda x: -x[1])
    return norm

def decide_unknown(top_list, threshold):
    if not top_list:
        return True
    return float(top_list[0][1]) < float(threshold)

def read_xlsx(path):
    df = pd.read_excel(path)
    out = []
    for _, r in df.iterrows():
        row = {k: ("" if pd.isna(r[k]) else str(r[k]).strip()) for k in r.index}
        sample = {}
        for k in ["数据库名", "数据库名dbname", "dbname"]:
            if k in row:
                sample["dbname"] = row[k]
                break
        for k in ["数据表名", "数据表名tablename", "tablename", "表名"]:
            if k in row:
                sample["tablename"] = row[k]
                break
        for k in ["字段名", "fieldname", "字段"]:
            if k in row:
                sample["fieldname"] = row[k]
                break
        for k in ["字段描述", "fielddesc", "描述"]:
            if k in row:
                sample["fielddesc"] = row[k]
                break
        for k in ["数据示例", "示例", "example"]:
            if k in row:
                sample["example"] = row[k]
                break
        for k in ["数据类型", "datatype", "类型"]:
            if k in row:
                sample["datatype"] = row[k]
                break
        for k in ["是否可为空", "NULL", "可空"]:
            if k in row:
                sample["null"] = row[k]
                break
        for k in ["枚举值", "enum", "枚举"]:
            if k in row:
                sample["enum"] = row[k]
                break
        out.append(sample)
    return out

def infer_file(xlsx_path, tax_path="labels/taxonomy.json", domain_model_dir="outputs/models/domain", sub1_model_dir="outputs/models/subclass1", thresholds_path="labels/infer_thresholds.json", priors_path="labels/term_priors.json"):
    tax = load_taxonomy(tax_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain_model = load_model_or_none(domain_model_dir, device)
    sub1_model = load_model_or_none(sub1_model_dir, device)
    base_model = SentenceTransformer("shibing624/text2vec-base-chinese-nli", device=device)
    thresholds = load_thresholds(thresholds_path)
    priors = load_priors(priors_path)
    domain_candidates = list(tax["domains"].keys())
    samples = read_xlsx(xlsx_path)
    fields_out = []
    for s in samples:
        txt = build_text(s)
        if domain_model and Path(domain_model_dir, "labels.json").exists():
            with open(Path(domain_model_dir, "labels.json"), "r", encoding="utf-8") as f:
                dlabels = json.load(f)
            dtop = predict_with_labels(domain_model, dlabels, txt)
        else:
            dtop = predict_by_prototype(base_model, domain_candidates, txt)
        dtop = apply_priors(dtop, priors.get("domain", {}), txt)
        d_unknown = decide_unknown(dtop, thresholds.get("domain", 0.45))
        pred_domain = dtop[0]
        if not d_unknown:
            sub1_candidates = list(tax["domains"][pred_domain[0]]["一级子类"].keys())
            if sub1_model and Path(sub1_model_dir, "labels.json").exists():
                with open(Path(sub1_model_dir, "labels.json"), "r", encoding="utf-8") as f:
                    s1labels = json.load(f)
                s1top_all = predict_with_labels(sub1_model, s1labels, txt)
                s1top = [x for x in s1top_all if x[0] in sub1_candidates]
                if not s1top:
                    s1top = predict_by_prototype(base_model, sub1_candidates, txt)
            else:
                s1top = predict_by_prototype(base_model, sub1_candidates, txt)
            s1top = apply_priors(s1top, priors.get("sub1", {}), txt)
            s1_unknown = decide_unknown(s1top, thresholds.get("sub1", 0.40))
            pred_sub1 = s1top[0]
            if not s1_unknown:
                sub2_candidates = tax["domains"][pred_domain[0]]["一级子类"][pred_sub1[0]]
                if sub2_candidates:
                    s2top = predict_by_prototype(base_model, sub2_candidates, txt)
                    s2top = apply_priors(s2top, priors.get("sub2", {}), txt)
                    s2_unknown = decide_unknown(s2top, thresholds.get("sub2", 0.35))
                    pred_sub2 = s2top[0]
                else:
                    s2top = []
                    s2_unknown = True
                    pred_sub2 = ("", 0.0)
            else:
                s2top = []
                s2_unknown = True
                pred_sub2 = ("", 0.0)
        else:
            s1top = []
            s2top = []
            pred_sub1 = ("", 0.0)
            pred_sub2 = ("", 0.0)
            s1_unknown = True
            s2_unknown = True
        fields_out.append({
            "fieldname": s.get("fieldname", ""),
            "text": txt,
            "pred_domain": {"label": ("unknown" if d_unknown else pred_domain[0]), "score": float(pred_domain[1]), "is_unknown": bool(d_unknown)},
            "pred_subclass1": {"label": ("unknown" if s1_unknown else pred_sub1[0]), "score": float(pred_sub1[1]), "is_unknown": bool(s1_unknown)},
            "pred_subclass2": {"label": ("unknown" if s2_unknown else pred_sub2[0]), "score": float(pred_sub2[1]), "is_unknown": bool(s2_unknown)},
            "candidates": {
                "domain_top3": [{"label": l, "score": float(sc)} for l, sc in dtop[:3]],
                "subclass1_top3": [{"label": l, "score": float(sc)} for l, sc in s1top[:3]],
                "subclass2_top3": [{"label": l, "score": float(sc)} for l, sc in s2top[:3]]
            }
        })
    return {
        "dbname": samples[0].get("dbname", "") if samples else "",
        "tablename": samples[0].get("tablename", "") if samples else "",
        "fields": fields_out
    }

def run_dir(input_root="data_", output_root="outputs/predictions"):
    outdir = Path(output_root)
    outdir.mkdir(parents=True, exist_ok=True)
    for p in Path(input_root).rglob("*.xlsx"):
        res = infer_file(str(p))
        outp = outdir / (p.stem + ".json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_dir()
