import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, losses, InputExample
import torch
from torch.utils.data import DataLoader

def load_dataset(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x.get("domain"):
                items.append(x)
    return items

def build_label_map(items):
    labels = sorted(list({x["domain"] for x in items}))
    return {l: i for i, l in enumerate(labels)}

def build_examples(items, label_map):
    ex = []
    for x in items:
        y = label_map[x["domain"]]
        ex.append(InputExample(texts=[x["text"]], label=y))
    return ex

def train(input_path="outputs/dataset.jsonl", outdir="outputs/models/domain", batch_size=32, epochs=6, lr=2e-5, device=None):
    items = load_dataset(input_path)
    if not items:
        return None
    label_map = build_label_map(items)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("shibing624/text2vec-base-chinese-nli", device=device)
    train_ex = build_examples(items, label_map)
    loader = DataLoader(train_ex, batch_size=batch_size, shuffle=True)
    loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label_map))
    warmup = int(len(loader) * epochs * 0.1)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=warmup)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    model.save(outdir)
    with open(Path(outdir) / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False)
    return outdir

if __name__ == "__main__":
    train()
