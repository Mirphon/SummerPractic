
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
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF

EXPERIMENT    = "baseline"
PROJECT_DIR   = "results/faster_rcnn"
WEIGHTS_DIR   = f"{PROJECT_DIR}/{EXPERIMENT}/weights"
PLOTS_DIR     = f"{PROJECT_DIR}/{EXPERIMENT}/plots"

# Данные
ANN_TRAIN      = "data/processed/annotations_train.json"
ANN_VAL        = "data/processed/annotations_val.json"
IMAGES_DIR_TRAIN = "data/raw/coco/train2017"
IMAGES_DIR_VAL   = "data/raw/coco/val2017"
SURVEILLANCE_CLASSES = [
    "__background__",  
    "person",          
    "bicycle",          
    "car",             
    "motorcycle",      
    "bus",              
    "truck",           
    "backpack",         
    "handbag",          
    "suitcase", 
]       
NUM_CLASSES = len(SURVEILLANCE_CLASSES)  


EPOCHS        = 5      
BATCH_SIZE    = 4
MAX_IMAGES    = 2000        
LR            = 0.005
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0005
PRINT_FREQ    = 50       
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

        self.coco_id_to_name = {
            cat["id"]: cat["name"] for cat in data["categories"]
        }

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
                    "path": img_path,
                    "anns": valid_anns,
                    "width":  img_info["width"],
                    "height": img_info["height"],
                })

        print(f"Датасет загружен: {len(self.samples)} изображений")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["path"]).convert("RGB")
        img_tensor = TF.to_tensor(img)

        boxes  = []
        labels = []

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

        target = {
            "boxes":  boxes_tensor,
            "labels": labels_tensor,
        }

        if self.augment and torch.rand(1).item() > 0.5:
            img_tensor, target = self._hflip(img_tensor, target, sample["width"])

        return img_tensor, target

    def _hflip(self, img, target, img_width):
        img = TF.hflip(img)
        boxes = target["boxes"].clone()
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] = img_width - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



def train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=50):
    model.train()

    total_loss       = 0.0
    loss_classifier  = 0.0
    loss_box_reg     = 0.0
    loss_objectness  = 0.0
    loss_rpn_box_reg = 0.0
    n_batches        = 0
    t_start          = time.time()

    for i, (images, targets) in enumerate(dataloader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss       += losses.item()
        loss_classifier  += loss_dict.get("loss_classifier", torch.tensor(0)).item()
        loss_box_reg     += loss_dict.get("loss_box_reg", torch.tensor(0)).item()
        loss_objectness  += loss_dict.get("loss_objectness", torch.tensor(0)).item()
        loss_rpn_box_reg += loss_dict.get("loss_rpn_box_reg", torch.tensor(0)).item()
        n_batches        += 1

        if (i + 1) % print_freq == 0:
            elapsed = time.time() - t_start
            print(
                f"  Epoch [{epoch}] Batch [{i+1}/{len(dataloader)}] "
                f"Loss: {losses.item():.4f} "
                f"(cls: {loss_dict.get('loss_classifier', torch.tensor(0)).item():.3f} "
                f"box: {loss_dict.get('loss_box_reg', torch.tensor(0)).item():.3f} "
                f"obj: {loss_dict.get('loss_objectness', torch.tensor(0)).item():.3f}) "
                f"Time: {elapsed:.0f}s"
            )

    avg_loss = total_loss / max(n_batches, 1)
    return {
        "total":       avg_loss,
        "classifier":  loss_classifier  / max(n_batches, 1),
        "box_reg":     loss_box_reg     / max(n_batches, 1),
        "objectness":  loss_objectness  / max(n_batches, 1),
        "rpn_box_reg": loss_rpn_box_reg / max(n_batches, 1),
    }



def evaluate(model, dataloader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]

            predictions = model(images)

            for pred, target in zip(predictions, targets):
                pred_boxes  = pred["boxes"][pred["scores"] > score_threshold].cpu()
                pred_labels = pred["labels"][pred["scores"] > score_threshold].cpu()
                gt_boxes    = target["boxes"]
                gt_labels   = target["labels"]

                matched_gt = set()

                for pb, pl in zip(pred_boxes, pred_labels):
                    best_iou   = 0.0
                    best_gt_idx = -1

                    for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if j in matched_gt:
                            continue
                        if pl != gl:
                            continue
                        iou = compute_iou(pb, gb)
                        if iou > best_iou:
                            best_iou    = iou
                            best_gt_idx = j

                    if best_iou >= iou_threshold:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1

                total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)

    return {"precision": precision, "recall": recall, "f1": f1}


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



def plot_losses(history, plots_dir):
    epochs = range(1, len(history["total"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["total"], "b-o", linewidth=2, label="Total Loss")
    axes[0].set_title("Total Loss по эпохам", fontsize=13)
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].plot(epochs, history["classifier"],  label="Classifier Loss",  linewidth=2)
    axes[1].plot(epochs, history["box_reg"],      label="Box Reg Loss",     linewidth=2)
    axes[1].plot(epochs, history["objectness"],   label="Objectness Loss",  linewidth=2)
    axes[1].plot(epochs, history["rpn_box_reg"],  label="RPN Box Reg Loss", linewidth=2)
    axes[1].set_title("Компоненты Loss", fontsize=13)
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(linestyle="--", alpha=0.4)

    fig.suptitle("Faster R-CNN — Кривые обучения", fontsize=14)
    fig.tight_layout()

    path = os.path.join(plots_dir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График сохранён: {path}")


def plot_metrics(history, plots_dir):
    epochs = range(1, len(history["precision"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["precision"], "b-o", linewidth=2, label="Precision")
    ax.plot(epochs, history["recall"],    "r-o", linewidth=2, label="Recall")
    ax.plot(epochs, history["f1"],        "g-o", linewidth=2, label="F1-score")
    ax.set_title("Faster R-CNN — Метрики качества по эпохам", fontsize=13)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(plots_dir, "metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График сохранён: {path}")


def visualize_predictions(model, dataset, device, plots_dir, n=4):
    model.eval()
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    class_names = SURVEILLANCE_CLASSES

    with torch.no_grad():
        for i in range(n):
            idx = torch.randint(len(dataset), (1,)).item()
            img_tensor, target = dataset[idx]

            pred = model([img_tensor.to(device)])[0]

            ax  = axes[i]
            img = img_tensor.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")

            # Предсказанные рамки
            for box, label, score in zip(
                pred["boxes"].cpu(),
                pred["labels"].cpu(),
                pred["scores"].cpu()
            ):
                if score < 0.5:
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="#FF6B6B", facecolor="none"
                )
                ax.add_patch(rect)
                name = class_names[label] if label < len(class_names) else str(label.item())
                ax.text(x1, y1 - 4, f"{name} {score:.2f}",
                        fontsize=7, color="white",
                        bbox=dict(facecolor="#FF6B6B", alpha=0.8, pad=1, edgecolor="none"))

            ax.set_title(f"Пример {i+1}", fontsize=9)

    fig.suptitle("Faster R-CNN — Предсказания модели", fontsize=13)
    fig.tight_layout()

    path = os.path.join(plots_dir, "predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Визуализация предсказаний: {path}")



if __name__ == "__main__":
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("="*60)
    print("  FASTER R-CNN — Обучение")
    print("="*60)
    print(f"  Device:     {DEVICE}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {vram:.1f} GB")
    print(f"  Классов:    {NUM_CLASSES} (включая фон)")
    print(f"  Эпох:       {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  LR:         {LR}")
    print("="*60)


    print("\n▶ Загрузка датасетов")
    train_dataset = COCODetectionDataset(
        ANN_TRAIN, IMAGES_DIR_TRAIN, SURVEILLANCE_CLASSES, augment=True
    )
    if MAX_IMAGES and len(train_dataset) > MAX_IMAGES:
        import torch
        indices = torch.randperm(len(train_dataset))[:MAX_IMAGES].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"  Используем {MAX_IMAGES} из {len(train_dataset.dataset)} изображений")
    val_dataset = COCODetectionDataset(
        ANN_VAL, IMAGES_DIR_VAL, SURVEILLANCE_CLASSES, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )


    print("\n▶ Инициализация модели")
    model = build_model(NUM_CLASSES)
    model.to(DEVICE)
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"  Заморожено слоёв: {frozen}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметров: {total_params:,}")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr           = LR,
        momentum     = MOMENTUM,
        weight_decay = WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    history = {
        "total": [], "classifier": [], "box_reg": [],
        "objectness": [], "rpn_box_reg": [],
        "precision": [], "recall": [], "f1": [],
    }

    best_f1 = 0.0

    print("\n▶ Начинаем обучение\n")

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()
        print(f"\n{'─'*50}")
        print(f"Эпоха {epoch}/{EPOCHS}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'─'*50}")

        losses = train_one_epoch(
            model, optimizer, train_loader, DEVICE, epoch, PRINT_FREQ
        )
        lr_scheduler.step()


        print(f"  Оценка на val")
        metrics = evaluate(model, val_loader, DEVICE)

        for key in ["total", "classifier", "box_reg", "objectness", "rpn_box_reg"]:
            history[key].append(losses[key])
        for key in ["precision", "recall", "f1"]:
            history[key].append(metrics[key])

        elapsed = time.time() - t_epoch
        print(f"\n  Результаты эпохи {epoch}:")
        print(f"    Loss:      {losses['total']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    Время:     {elapsed:.0f}s")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_path = os.path.join(WEIGHTS_DIR, "best.pt")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "f1":        best_f1,
                "metrics":   metrics,
            }, best_path)
            print(f"    ✓ Лучшая модель сохранена (F1={best_f1:.4f})")

        last_path = os.path.join(WEIGHTS_DIR, "last.pt")
        torch.save({
            "epoch":   epoch,
            "model":   model.state_dict(),
            "metrics": metrics,
        }, last_path)

    print("\n▶ Построение графиков")
    plot_losses(history,  PLOTS_DIR)
    plot_metrics(history, PLOTS_DIR)


    print("\n▶ Визуализация предсказаний")
    visualize_predictions(model, val_dataset, DEVICE, PLOTS_DIR)

    print("\n" + "="*60)
    print("  ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"  Лучший F1:   {best_f1:.4f}")
    print(f"  Веса:        {WEIGHTS_DIR}/best.pt")
    print(f"  Графики:     {PLOTS_DIR}/")
    print("="*60)