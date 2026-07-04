import json
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from effdet import create_model

EXPERIMENT   = "baseline"
PROJECT_DIR  = "results/efficientdet"
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

IMG_SIZE     = 512
EPOCHS       = 10
BATCH_SIZE   = 4
LR           = 0.001
WEIGHT_DECAY = 0.0005
MAX_IMAGES   = 2000
PRINT_FREQ   = 50
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class COCODetectionDataset(Dataset):
    def __init__(self, ann_file, images_dir, class_names, img_size=512, augment=False):
        self.images_dir = images_dir
        self.augment    = augment
        self.img_size   = img_size
        self.name_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}

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
        orig_w, orig_h = img.size

        img_resized = img.resize((self.img_size, self.img_size))
        img_tensor  = TF.to_tensor(img_resized)
        img_tensor  = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        boxes, labels = [], []
        for ann in sample["anns"]:
            x, y, w, h = ann["bbox"]
            cat_name = self.coco_id_to_name.get(ann["category_id"])
            local_id = self.name_to_id.get(cat_name)
            if local_id is None:
                continue
            y1 = y * scale_y
            x1 = x * scale_x
            y2 = (y + h) * scale_y
            x2 = (x + w) * scale_x
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([y1, x1, y2, x2])
            labels.append(local_id)

        if self.augment and torch.rand(1).item() > 0.5:
            img_tensor = torch.flip(img_tensor, dims=[2])
            new_boxes = []
            for y1, x1, y2, x2 in boxes:
                nx1 = self.img_size - x2
                nx2 = self.img_size - x1
                new_boxes.append([y1, nx1, y2, nx2])
            boxes = new_boxes

        if boxes:
            boxes_tensor  = torch.tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        else:
            boxes_tensor  = torch.zeros((1, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((1,),   dtype=torch.float32)

        return img_tensor, boxes_tensor, labels_tensor


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    boxes  = [item[1] for item in batch]
    cls    = [item[2] for item in batch]

    batch_size = len(batch)
    img_size  = torch.tensor(
        [[IMG_SIZE, IMG_SIZE]] * batch_size, dtype=torch.float32
    )
    img_scale = torch.ones(batch_size, dtype=torch.float32)

    target = {
        "bbox":      boxes,
        "cls":       cls,
        "img_size":  img_size,
        "img_scale": img_scale,
    }
    return images, target


def build_model(num_classes, img_size, device):
    from effdet import create_model, get_efficientdet_config
    from effdet.anchors import Anchors, AnchorLabeler

    config = get_efficientdet_config("tf_efficientdet_d0")
    config.num_classes = num_classes
    config.image_size = (img_size, img_size)

    model = create_model(
        "tf_efficientdet_d0",
        bench_task="train",
        num_classes=num_classes,
        pretrained=True,
        image_size=(img_size, img_size),
    )

    anchors = Anchors.from_config(config).to(device)
    labeler = AnchorLabeler(anchors, num_classes, match_threshold=0.5)
    model.anchor_labeler = labeler

    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    return model


def build_predict_model(num_classes, img_size, state_dict):
    model = create_model(
        "tf_efficientdet_d0",
        bench_task="predict",
        num_classes=num_classes,
        pretrained=False,
        image_size=(img_size, img_size),
    )
    train_model = create_model(
        "tf_efficientdet_d0",
        bench_task="train",
        num_classes=num_classes,
        pretrained=False,
        image_size=(img_size, img_size),
    )
    train_model.load_state_dict(state_dict)
    model.model.load_state_dict(train_model.model.state_dict())
    return model


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t_start    = time.time()

    for i, (images, target) in enumerate(dataloader):
        images = images.to(device)
        target = {
            "bbox":      [b.to(device) for b in target["bbox"]],
            "cls":       [c.to(device) for c in target["cls"]],
            "img_size":  target["img_size"].to(device),
            "img_scale": target["img_scale"].to(device),
        }

        output = model(images, target)
        loss   = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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


def evaluate(model, dataset, device, img_size, iou_threshold=0.5, score_threshold=0.3, max_eval=400):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    n_eval = min(len(dataset), max_eval)

    with torch.no_grad():
        for idx in range(n_eval):
            img_tensor, gt_boxes_t, gt_labels_t = dataset[idx]
            images = img_tensor.unsqueeze(0).to(device)

            img_size_t = torch.tensor([[img_size, img_size]], dtype=torch.float32).to(device)
            img_scale  = torch.ones(1, dtype=torch.float32).to(device)

            target = {
                "bbox":      [gt_boxes_t.to(device)],
                "cls":       [gt_labels_t.to(device)],
                "img_size":  img_size_t,
                "img_scale": img_scale,
            }

            output = model(images, target)

            detections = output["detections"].detach().cpu().numpy()[0]

            pred_boxes, pred_labels = [], []
            for det in detections:
                x1, y1, x2, y2, score, cls_id = det[:6]
                if score < score_threshold:
                    continue
                pred_boxes.append([x1, y1, x2, y2])
                pred_labels.append(int(cls_id))

            gt_boxes, gt_labels = [], []
            for box, label in zip(gt_boxes_t.tolist(), gt_labels_t.tolist()):
                y1, x1, y2, x2 = box
                gt_boxes.append([x1, y1, x2, y2])
                gt_labels.append(int(label))

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
    axes[0].set_title("EfficientDet — Loss по эпохам")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].plot(epochs, metrics_history["precision"], "b-o", linewidth=2, label="Precision")
    axes[1].plot(epochs, metrics_history["recall"],    "r-o", linewidth=2, label="Recall")
    axes[1].plot(epochs, metrics_history["f1"],        "g-o", linewidth=2, label="F1")
    axes[1].set_title("EfficientDet — Метрики по эпохам")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Значение")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(linestyle="--", alpha=0.4)

    fig.suptitle("EfficientDet-D0 — Результаты обучения", fontsize=14)
    fig.tight_layout()
    path = os.path.join(plots_dir, "results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График сохранён: {path}")


if __name__ == "__main__":
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("="*60)
    print("  EfficientDet-D0 — Обучение")
    print("="*60)
    print(f"  Device:      {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Размер:      {IMG_SIZE}px")
    print(f"  Изображений: {MAX_IMAGES} из train2017")
    print(f"  Эпох:        {EPOCHS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LR}")
    print("="*60)

    print("\n▶ Загрузка датасетов...")
    train_dataset = COCODetectionDataset(
        ANN_TRAIN, IMAGES_DIR_TRAIN, SURVEILLANCE_CLASSES, IMG_SIZE, augment=True
    )
    val_dataset = COCODetectionDataset(
        ANN_VAL, IMAGES_DIR_VAL, SURVEILLANCE_CLASSES, IMG_SIZE, augment=False
    )

    if len(train_dataset) > MAX_IMAGES:
        indices = torch.randperm(len(train_dataset))[:MAX_IMAGES].tolist()
        train_dataset.samples = [train_dataset.samples[i] for i in indices]
        print(f"  Train обрезан до: {len(train_dataset)} изображений")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    print("\n▶ Инициализация модели")
    model = build_model(NUM_CLASSES, IMG_SIZE, DEVICE)
    model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Всего параметров: {total:,}")
    print(f"  Обучаемых:        {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    loss_history    = []
    metrics_history = {"precision": [], "recall": [], "f1": []}
    best_f1         = 0.0

    print("\n▶ Начинаем обучение\n")

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()
        print(f"\n{'─'*50}")
        print(f"Эпоха {epoch}/{EPOCHS}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'─'*50}")

        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        lr_scheduler.step()

        print(f"  Оценка на val...")
        metrics = evaluate(model, val_dataset, DEVICE, IMG_SIZE)

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