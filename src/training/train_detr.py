import json
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

EXPERIMENT   = "baseline"
PROJECT_DIR  = "results/detr"
WEIGHTS_DIR  = f"{PROJECT_DIR}/{EXPERIMENT}/weights"
PLOTS_DIR    = f"{PROJECT_DIR}/{EXPERIMENT}/plots"

ANN_TRAIN        = "data/processed/annotations_train.json"
ANN_VAL          = "data/processed/annotations_val.json"
IMAGES_DIR_TRAIN = "data/raw/coco/train2017"
IMAGES_DIR_VAL   = "data/raw/coco/val2017"

SURVEILLANCE_CLASSES = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "backpack", "handbag", "suitcase",
]
NUM_CLASSES = len(SURVEILLANCE_CLASSES)

EPOCHS     = 5
BATCH_SIZE = 4
LR         = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
MAX_IMAGES = 2000
PRINT_FREQ = 50
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "facebook/detr-resnet-50"

processor = DetrImageProcessor.from_pretrained(MODEL_NAME)


class COCODetectionDataset(Dataset):
    def __init__(self, ann_file, images_dir, class_names, augment=False):
        self.images_dir = images_dir
        self.augment    = augment
        self.name_to_id = {name: idx for idx, name in enumerate(class_names)}

        with open(ann_file, "r") as f:
            data = json.load(f)

        self.coco_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

        from collections import defaultdict
        anns_by_image = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        self.samples = []
        for img_info in data["images"]:
            img_id   = img_info["id"]
            img_path = os.path.join(images_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            anns = anns_by_image.get(img_id, [])
            valid_anns = [
                a for a in anns
                if self.coco_id_to_name.get(a["category_id"]) in self.name_to_id
                and a.get("iscrowd", 0) == 0
            ]
            if valid_anns:
                self.samples.append({
                    "path":   img_path,
                    "anns":   valid_anns,
                    "width":  img_info["width"],
                    "height": img_info["height"],
                    "id":     img_id,
                })

        print(f"Датасет: {len(self.samples)} изображений ({images_dir})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img    = Image.open(sample["path"]).convert("RGB")

        annotations = []
        for ann in sample["anns"]:
            cat_name = self.coco_id_to_name.get(ann["category_id"])
            local_id = self.name_to_id.get(cat_name)
            if local_id is None:
                continue
            annotations.append({
                "bbox":        ann["bbox"],
                "category_id": local_id,
                "area":        ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd":     0,
            })

        target = {"image_id": sample["id"], "annotations": annotations}

        encoding = processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels       = encoding["labels"][0]

        return pixel_values, labels


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels       = [item[1] for item in batch]

    max_h = max(pv.shape[1] for pv in pixel_values)
    max_w = max(pv.shape[2] for pv in pixel_values)

    padded_images = []
    pixel_mask    = []
    for pv in pixel_values:
        c, h, w = pv.shape
        padded = torch.zeros((c, max_h, max_w), dtype=pv.dtype)
        padded[:, :h, :w] = pv
        padded_images.append(padded)

        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1
        pixel_mask.append(mask)

    return {
        "pixel_values": torch.stack(padded_images),
        "pixel_mask":   torch.stack(pixel_mask),
        "labels":       labels,
    }


def build_model(num_classes):
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t_start    = time.time()

    for i, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask   = batch["pixel_mask"].to(device)
        labels       = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if (i + 1) % PRINT_FREQ == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch [{epoch}] Batch [{i+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}  Time: {elapsed:.0f}s")

    return total_loss / max(n_batches, 1)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / max(union, 1e-6)


def evaluate(model, dataset, device, iou_threshold=0.5, score_threshold=0.5, max_eval=200):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    n_eval = min(len(dataset), max_eval)

    with torch.no_grad():
        for idx in range(n_eval):
            sample = dataset.samples[idx]
            img    = Image.open(sample["path"]).convert("RGB")
            encoding = processor(images=img, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)

            outputs = model(pixel_values=pixel_values)

            target_sizes = torch.tensor([[sample["height"], sample["width"]]]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_threshold
            )[0]

            pred_boxes  = results["boxes"].cpu().tolist()
            pred_labels = results["labels"].cpu().tolist()

            gt_boxes, gt_labels = [], []
            for ann in sample["anns"]:
                cat_name = dataset.coco_id_to_name.get(ann["category_id"])
                local_id = dataset.name_to_id.get(cat_name)
                if local_id is None:
                    continue
                x, y, w, h = ann["bbox"]
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(local_id)

            matched_gt = set()
            for pb, pl in zip(pred_boxes, pred_labels):
                best_iou, best_idx = 0.0, -1
                for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if j in matched_gt or pl != gl:
                        continue
                    iou = compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
                if best_iou >= iou_threshold:
                    total_tp += 1
                    matched_gt.add(best_idx)
                else:
                    total_fp += 1
            total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1}


def plot_results(loss_history, metrics_history, plots_dir):
    epochs = range(1, len(loss_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, loss_history, "b-o", linewidth=2)
    axes[0].set_title("DETR — Loss по эпохам")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].plot(epochs, metrics_history["precision"], "b-o", linewidth=2, label="Precision")
    axes[1].plot(epochs, metrics_history["recall"],    "r-o", linewidth=2, label="Recall")
    axes[1].plot(epochs, metrics_history["f1"],        "g-o", linewidth=2, label="F1")
    axes[1].set_title("DETR — Метрики по эпохам")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Значение")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(linestyle="--", alpha=0.4)

    fig.suptitle("DETR ResNet-50 — Результаты обучения", fontsize=14)
    fig.tight_layout()
    path = os.path.join(plots_dir, "results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График сохранён: {path}")


if __name__ == "__main__":
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("="*60)
    print("  DETR — Обучение")
    print("="*60)
    print(f"  Device:      {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Изображений: {MAX_IMAGES} из train2017")
    print(f"  Эпох:        {EPOCHS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LR}")
    print("="*60)

    print("\n▶ Загрузка датасетов...")
    train_dataset = COCODetectionDataset(
        ANN_TRAIN, IMAGES_DIR_TRAIN, SURVEILLANCE_CLASSES, augment=True
    )
    val_dataset = COCODetectionDataset(
        ANN_VAL, IMAGES_DIR_VAL, SURVEILLANCE_CLASSES, augment=False
    )

    if len(train_dataset) > MAX_IMAGES:
        indices = torch.randperm(len(train_dataset))[:MAX_IMAGES].tolist()
        train_dataset.samples = [train_dataset.samples[i] for i in indices]
        print(f"  Train обрезан до: {len(train_dataset)} изображений")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    print("\n▶ Инициализация модели...")
    model = build_model(NUM_CLASSES)
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметров: {total_params:,}")

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad],
            "lr": LR,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": LR_BACKBONE,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=LR, weight_decay=WEIGHT_DECAY)

    loss_history    = []
    metrics_history = {"precision": [], "recall": [], "f1": []}
    best_f1         = 0.0

    print("\n▶ Начинаем обучение...\n")

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()
        print(f"\n{'─'*50}")
        print(f"Эпоха {epoch}/{EPOCHS}")
        print(f"{'─'*50}")

        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)

        print(f"  Оценка на val...")
        metrics = evaluate(model, val_dataset, DEVICE)

        loss_history.append(avg_loss)
        for key in ["precision", "recall", "f1"]:
            metrics_history[key].append(metrics[key])

        elapsed = time.time() - t_epoch
        print(f"\n  Результаты эпохи {epoch}:")
        print(f"    Loss:      {avg_loss:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    Время:     {elapsed:.0f}s")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(os.path.join(WEIGHTS_DIR, "best"))
            print(f"    ✓ Лучшая модель сохранена (F1={best_f1:.4f})")

    print("\n▶ Построение графиков...")
    plot_results(loss_history, metrics_history, PLOTS_DIR)

    print("\n" + "="*60)
    print("  ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"  Лучший F1:  {best_f1:.4f}")
    print(f"  Веса:       {WEIGHTS_DIR}/best/")
    print(f"  Графики:    {PLOTS_DIR}/")
    print("="*60)