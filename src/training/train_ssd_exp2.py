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
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
import torchvision.transforms.functional as TF

EXPERIMENT = "exp2_lr_lower"
PROJECT_DIR  = "results/ssd"
WEIGHTS_DIR  = f"{PROJECT_DIR}/{EXPERIMENT}/weights"
PLOTS_DIR    = f"{PROJECT_DIR}/{EXPERIMENT}/plots"

ANN_TRAIN        = "data/processed/annotations_train.json"
ANN_VAL          = "data/processed/annotations_val.json"
IMAGES_DIR_TRAIN = "data/raw/coco/train2017"
IMAGES_DIR_VAL   = "data/raw/coco/val2017"

SURVEILLANCE_CLASSES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "backpack", "handbag", "suitcase",
]
NUM_CLASSES = len(SURVEILLANCE_CLASSES)

EPOCHS     = 10
BATCH_SIZE = 8
LR         = 0.001
MOMENTUM   = 0.9
WEIGHT_DECAY = 0.0005
MAX_IMAGES = 2000
PRINT_FREQ = 50
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class COCODetectionDataset(Dataset):
    def __init__(self, ann_file, images_dir, class_names, augment=False):
        self.images_dir = images_dir
        self.augment    = augment
        self.name_to_id = {
            name: idx for idx, name in enumerate(class_names)
            if name != "__background__"
        }

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
                })

        print(f"Датасет: {len(self.samples)} изображений ({images_dir})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img    = Image.open(sample["path"]).convert("RGB")
        img_tensor = TF.to_tensor(img)

        boxes, labels = [], []
        for ann in sample["anns"]:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue
            cat_name = self.coco_id_to_name.get(ann["category_id"])
            local_id = self.name_to_id.get(cat_name)
            if local_id is None:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(local_id)

        if boxes:
            boxes_tensor  = torch.tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}

        if self.augment and torch.rand(1).item() > 0.5:
            img_tensor = TF.hflip(img_tensor)
            if boxes_tensor.shape[0] > 0:
                w = sample["width"]
                b = target["boxes"].clone()
                b[:, [0, 2]] = w - b[:, [2, 0]]
                target["boxes"] = b

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]

    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes
    )

    for name, param in model.named_parameters():
        if "features" in name:
            param.requires_grad = False

    return model


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t_start    = time.time()

    for i, (images, targets) in enumerate(dataloader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses.item()
        n_batches  += 1

        if (i + 1) % PRINT_FREQ == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch [{epoch}] Batch [{i+1}/{len(dataloader)}] "
                  f"Loss: {losses.item():.4f}  Time: {elapsed:.0f}s")

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


def evaluate(model, dataloader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, targets in dataloader:
            images      = [img.to(device) for img in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                pred_boxes  = pred["boxes"][pred["scores"] > score_threshold].cpu()
                pred_labels = pred["labels"][pred["scores"] > score_threshold].cpu()
                gt_boxes    = target["boxes"]
                gt_labels   = target["labels"]

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
    axes[0].set_title("SSD — Loss по эпохам")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].plot(epochs, metrics_history["precision"], "b-o", linewidth=2, label="Precision")
    axes[1].plot(epochs, metrics_history["recall"],    "r-o", linewidth=2, label="Recall")
    axes[1].plot(epochs, metrics_history["f1"],        "g-o", linewidth=2, label="F1")
    axes[1].set_title("SSD — Метрики по эпохам")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Значение")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(linestyle="--", alpha=0.4)

    fig.suptitle("SSD300 VGG16 — Результаты обучения", fontsize=14)
    fig.tight_layout()
    path = os.path.join(plots_dir, "results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График сохранён: {path}")


if __name__ == "__main__":
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("="*60)
    print("  SSD300 — Обучение")
    print("="*60)
    print(f"  Device:     {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
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

    indices = torch.randperm(len(train_dataset))[:MAX_IMAGES].tolist()
    train_subset = torch.utils.data.Subset(train_dataset, indices)
    print(f"  Train subset: {len(train_subset)} изображений")

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print("\n▶ Инициализация модели...")
    model = build_model(NUM_CLASSES)
    model.to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Всего параметров:    {total_params:,}")
    print(f"  Обучаемых:           {trainable_params:,}")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )

    loss_history    = []
    metrics_history = {"precision": [], "recall": [], "f1": []}
    best_f1         = 0.0

    print("\n▶ Начинаем обучение...\n")

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()
        print(f"\n{'─'*50}")
        print(f"Эпоха {epoch}/{EPOCHS}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'─'*50}")

        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        lr_scheduler.step()

        print(f"  Оценка на val...")
        metrics = evaluate(model, val_loader, DEVICE)

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
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "f1":      best_f1,
                "metrics": metrics,
            }, os.path.join(WEIGHTS_DIR, "best.pt"))
            print(f"    ✓ Лучшая модель сохранена (F1={best_f1:.4f})")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
        }, os.path.join(WEIGHTS_DIR, "last.pt"))

    print("\n▶ Построение графиков...")
    plot_results(loss_history, metrics_history, PLOTS_DIR)

    print("\n" + "="*60)
    print("  ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"  Лучший F1:  {best_f1:.4f}")
    print(f"  Веса:       {WEIGHTS_DIR}/best.pt")
    print(f"  Графики:    {PLOTS_DIR}/")
    print("="*60)