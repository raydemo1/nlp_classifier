import os
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, losses, InputExample
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

def setup_logging(log_dir="outputs/logs"):
    """设置日志记录"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"train_subclass1_{int(time.time())}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset(path, logger=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="加载数据集", unit="条"):
            x = json.loads(line)
            if x.get("sub1"):
                items.append(x)
    if logger:
        logger.info(f"加载了 {len(items)} 条有效的subclass1数据")
    return items

def build_label_map(items, logger=None):
    labels = sorted(list({x["sub1"] for x in items}))
    label_map = {l: i for i, l in enumerate(labels)}
    if logger:
        logger.info(f"构建了 {len(label_map)} 个subclass1标签: {list(labels)}")
    return label_map

def build_examples(items, label_map):
    ex = []
    for x in items:
        y = label_map[x["sub1"]]
        ex.append(InputExample(texts=[x["text"]], label=y))
    return ex

def train(input_path="outputs/dataset.jsonl", outdir="outputs/models/subclass1", batch_size=32, epochs=6, lr=2e-5, device=None):
    # 设置日志
    logger = setup_logging()
    logger.info("开始subclass1模型训练")

    # 加载数据
    items = load_dataset(input_path, logger)
    if not items:
        logger.error("没有找到有效的训练数据")
        return None

    # 构建标签映射
    label_map = build_label_map(items, logger)

    # 检测GPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输出GPU使用情况
    if device == "cuda":
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
        print(f"🚀 使用GPU训练: {gpu_count}个设备, 主设备: {gpu_name}, 显存: {gpu_memory:.1f}GB")
        logger.info(f"使用GPU训练: {gpu_count}个设备, 主设备: {gpu_name}, 显存: {gpu_memory:.1f}GB")
    else:
        print("💻 使用CPU训练")
        logger.info("使用CPU训练")

    logger.info(f"训练参数: batch_size={batch_size}, epochs={epochs}, lr={lr}, device={device}")

    # 创建模型
    logger.info("加载预训练模型...")
    model = SentenceTransformer("shibing624/text2vec-base-chinese-nli", device=device)

    # 构建训练数据
    logger.info("构建训练样本...")
    train_ex = build_examples(items, label_map)
    loader = DataLoader(train_ex, batch_size=batch_size, shuffle=True)

    # 设置损失函数
    logger.info("设置损失函数...")
    loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=len(label_map)
    )

    # 训练设置
    warmup = int(len(loader) * epochs * 0.1)
    logger.info(f"训练设置: {len(loader)}个batch, warmup={warmup}步")

    # 开始训练
    logger.info("开始训练...")
    start_time = time.time()

    # 添加进度条的回调函数
    class ProgressCallback:
        def __init__(self, total_epochs):
            self.pbar = tqdm(total=total_epochs, desc="训练进度", unit="epoch")
            self.epoch = 0

        def __call__(self, score, epoch, steps):
            if epoch > self.epoch:
                self.epoch = epoch
                self.pbar.update(1)
                logger.info(f"Epoch {epoch}/{total_epochs} 完成")

    total_steps = len(loader) * epochs
    logger.info(f"总训练步数: {total_steps}")

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        output_path=outdir,
        callback=ProgressCallback(epochs)
    )

    training_time = time.time() - start_time
    logger.info(f"训练完成，耗时: {training_time:.2f}秒")

    # 保存模型
    logger.info("保存模型...")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    model.save(outdir)

    # 保存标签映射
    with open(Path(outdir) / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False)

    logger.info(f"模型已保存到: {outdir}")
    print(f"✅ Subclass1模型训练完成，保存到: {outdir}")
    return outdir

if __name__ == "__main__":
    train()
