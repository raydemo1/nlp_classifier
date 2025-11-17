import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report

def load_dataset(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            items.append(x)
    return items

def build_key(x):
    return (x.get("source", ""), x.get("tablename", ""), x.get("fieldname", ""))

def load_predictions(pred_dir):
    out = {}
    for p in Path(pred_dir).glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        tab = data.get("tablename", "")
        src = str(p)
        for r in data.get("fields", []):
            key = (src, tab, r.get("fieldname", ""))
            out[key] = r
    return out

def evaluate(dataset_path="outputs/dataset.jsonl", pred_dir="outputs/predictions"):
    ds = load_dataset(dataset_path)
    preds = load_predictions(pred_dir)
    y_true = {"domain": [], "sub1": [], "sub2": [], "grade": []}
    y_pred = {"domain": [], "sub1": [], "sub2": [], "grade": []}
    for x in ds:
        key = build_key(x)
        pr = preds.get(key)
        if not pr:
            continue
        y_true["domain"].append(x.get("domain", ""))
        y_true["sub1"].append(x.get("sub1", ""))
        y_true["sub2"].append(x.get("sub2", ""))
        y_true["grade"].append(x.get("grade", ""))
        y_pred["domain"].append(pr.get("pred_domain", {}).get("label", ""))
        y_pred["sub1"].append(pr.get("pred_subclass1", {}).get("label", ""))
        y_pred["sub2"].append(pr.get("pred_subclass2", {}).get("label", ""))
        y_pred["grade"].append(pr.get("pred_grade", {}).get("label", ""))
    reports = {}
    for k in ["domain", "sub1", "sub2", "grade"]:
        if y_true[k] and y_pred[k]:
            reports[k] = classification_report(y_true[k], y_pred[k], digits=4)
    outp = Path("outputs/evaluation.txt")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for k, rep in reports.items():
            f.write("== " + k + " ==\n")
            f.write(rep + "\n\n")
    return str(outp)

if __name__ == "__main__":
    evaluate()

