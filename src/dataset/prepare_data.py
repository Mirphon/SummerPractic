import json
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


ANNOTATIONS_TRAIN = "data/raw/coco/annotations/instances_train2017.json"
ANNOTATIONS_VAL   = "data/raw/coco/annotations/instances_val2017.json"
IMAGES_VAL_DIR    = "data/raw/coco/val2017"
OUTPUT_DIR        = "data/processed"


SURVEILLANCE_CLASSES = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "backpack", "handbag", "suitcase"
]

MIN_BBOX_AREA = 32 * 32 
MIN_BBOX_SIZE = 10      


VAL_SPLIT  = 0.5     
TEST_SPLIT = 0.5          

RANDOM_SEED = 42

def load_and_filter(ann_file, surveillance_classes):

    print(f"\n{'='*50}")
    print(f"Загружаем: {ann_file}")

    with open(ann_file, "r") as f:
        data = json.load(f)

    name_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
    id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    target_cat_ids = {
        name_to_id[name]
        for name in surveillance_classes
        if name in name_to_id
    }
    print(f"Целевые классы ({len(target_cat_ids)}): {surveillance_classes}")

    total_anns = len(data["annotations"])
    filtered_anns = []
    removed_stats = defaultdict(int)

    for ann in data["annotations"]:
        if ann["category_id"] not in target_cat_ids:
            removed_stats["wrong_class"] += 1
            continue

        if ann.get("iscrowd", 0) == 1:
            removed_stats["is_crowd"] += 1
            continue

        x, y, w, h = ann["bbox"]
        area = w * h
        if area < MIN_BBOX_AREA:
            removed_stats["too_small_area"] += 1
            continue
        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            removed_stats["too_small_side"] += 1
            continue

        filtered_anns.append(ann)

    print(f"\nАннотаций до фильтрации:  {total_anns:>7}")
    print(f"Убрано (чужой класс):     {removed_stats['wrong_class']:>7}")
    print(f"Убрано (толпа iscrowd):   {removed_stats['is_crowd']:>7}")
    print(f"Убрано (мал. площадь):    {removed_stats['too_small_area']:>7}")
    print(f"Убрано (мал. сторона):    {removed_stats['too_small_side']:>7}")
    print(f"Аннотаций после:          {len(filtered_anns):>7}")


    valid_image_ids = {ann["image_id"] for ann in filtered_anns}
    filtered_images = [
        img for img in data["images"]
        if img["id"] in valid_image_ids
    ]

    print(f"\nИзображений до фильтрации: {len(data['images']):>6}")
    print(f"Изображений после:         {len(filtered_images):>6}")

    filtered_cats = [
        cat for cat in data["categories"]
        if cat["id"] in target_cat_ids
    ]

    return {
        "info":        data.get("info", {}),
        "licenses":    data.get("licenses", []),
        "categories":  filtered_cats,
        "images":      filtered_images,
        "annotations": filtered_anns,
    }, id_to_name


def analyze_filtered(filtered_data, id_to_name, title=""):
    print(f"\n── Статистика {title} ──")
    class_counts = defaultdict(int)
    for ann in filtered_data["annotations"]:
        name = id_to_name.get(ann["category_id"], "unknown")
        class_counts[name] += 1

    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // max(1, max(class_counts.values()) // 20))
        print(f"  {name:<15} {count:>6}  {bar}")



def split_val_test(filtered_data, val_ratio=0.5, seed=42):
    random.seed(seed)

    image_ids = [img["id"] for img in filtered_data["images"]]
    random.shuffle(image_ids)

    split_idx = int(len(image_ids) * val_ratio)
    val_ids   = set(image_ids[:split_idx])
    test_ids  = set(image_ids[split_idx:])

    def make_subset(ids):
        images = [img for img in filtered_data["images"] if img["id"] in ids]
        anns   = [ann for ann in filtered_data["annotations"] if ann["image_id"] in ids]
        return {
            "info":        filtered_data["info"],
            "licenses":    filtered_data["licenses"],
            "categories":  filtered_data["categories"],
            "images":      images,
            "annotations": anns,
        }

    val_data  = make_subset(val_ids)
    test_data = make_subset(test_ids)

    print(f"\n── Разбивка val2017 ──")
    print(f"  Val:  {len(val_data['images']):>4} изображений, "
          f"{len(val_data['annotations']):>6} аннотаций")
    print(f"  Test: {len(test_data['images']):>4} изображений, "
          f"{len(test_data['annotations']):>6} аннотаций")

    return val_data, test_data


def save_annotations(data, path):
    """Сохраняет отфильтрованные аннотации в JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Сохранено: {path}")



def demo_augmentations(filtered_data, images_dir, output_dir):
    try:
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image, ImageEnhance
    except ImportError:
        print("torchvision не найден, пропускаем визуализацию аугментаций")
        return

    random.seed(RANDOM_SEED)
    sample_ann = random.choice(filtered_data["annotations"])
    sample_img_info = next(
        img for img in filtered_data["images"]
        if img["id"] == sample_ann["image_id"]
    )

    img_path = os.path.join(images_dir, sample_img_info["file_name"])
    if not os.path.exists(img_path):
        print(f"Файл не найден для аугментации: {img_path}")
        return

    pil_img = Image.open(img_path).convert("RGB")
    W, H = pil_img.size

    img_id = sample_img_info["id"]
    bboxes = [
        ann["bbox"] for ann in filtered_data["annotations"]
        if ann["image_id"] == img_id
    ]

    def draw_boxes(ax, img, bboxes, title):
        ax.imshow(np.array(img))
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        for x, y, w, h in bboxes:
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor="#FF6B6B", facecolor="none"
            )
            ax.add_patch(rect)

    orig_img    = pil_img.copy()
    orig_bboxes = bboxes.copy()

    flipped_img = TF.hflip(pil_img)
    flipped_bboxes = []
    for x, y, w, h in bboxes:
        new_x = W - x - w
        flipped_bboxes.append([new_x, y, w, h])

    brightness_factor = random.uniform(0.7, 1.3)
    contrast_factor   = random.uniform(0.7, 1.3)
    jitter_img = ImageEnhance.Brightness(pil_img).enhance(brightness_factor)
    jitter_img = ImageEnhance.Contrast(jitter_img).enhance(contrast_factor)

    crop_ratio = 0.8
    cw = int(W * crop_ratio)
    ch = int(H * crop_ratio)
    cx = random.randint(0, W - cw)
    cy = random.randint(0, H - ch)
    cropped_img = TF.crop(pil_img, cy, cx, ch, cw)
    cropped_img = TF.resize(cropped_img, (H, W))

    cropped_bboxes = []
    for x, y, w, h in bboxes:
        nx = max(0, x - cx)
        ny = max(0, y - cy)
        nw = min(x + w, cx + cw) - max(x, cx)
        nh = min(y + h, cy + ch) - max(y, cy)
        if nw > 0 and nh > 0:
            scale_x = W / cw
            scale_y = H / ch
            cropped_bboxes.append([
                nx * scale_x, ny * scale_y,
                nw * scale_x, nh * scale_y
            ])

    blurred_img = pil_img.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=2))


    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    draw_boxes(axes[0], orig_img,     orig_bboxes,    "Оригинал")
    draw_boxes(axes[1], flipped_img,  flipped_bboxes, "Горизонтальное отражение (flip)")
    draw_boxes(axes[2], jitter_img,   orig_bboxes,    f"Color Jitter\n(яркость ×{brightness_factor:.2f}, контраст ×{contrast_factor:.2f})")
    draw_boxes(axes[3], cropped_img,  cropped_bboxes, f"Случайный кроп ({int(crop_ratio*100)}%)")
    draw_boxes(axes[4], blurred_img,  orig_bboxes,    "Размытие (Gaussian Blur)")
    axes[5].set_visible(False)

    fig.suptitle("Демонстрация аугментаций (изображение + рамки)", fontsize=14, y=1.01)
    fig.tight_layout()

    path = os.path.join(output_dir, "augmentation_demo.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nДемо аугментаций сохранено: {path}")



def plot_split_summary(val_data, test_data, id_to_name, output_dir):
    def count_classes(data):
        counts = defaultdict(int)
        for ann in data["annotations"]:
            name = id_to_name.get(ann["category_id"], "unknown")
            counts[name] += 1
        return counts

    val_counts  = count_classes(val_data)
    test_counts = count_classes(test_data)
    classes = sorted(val_counts.keys())

    x      = np.arange(len(classes))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, [val_counts[c]  for c in classes], width,
           label="Val",  color="#3A86FF", edgecolor="white")
    ax.bar(x + width/2, [test_counts[c] for c in classes], width,
           label="Test", color="#FF6B6B", edgecolor="white")

    ax.set_title("Распределение классов по разбивкам val / test", fontsize=13)
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество объектов")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(output_dir, "split_distribution.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"График разбивки сохранён: {path}")




if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)


    print("\n▶ Шаг 1: Фильтрация val2017")
    val_filtered, id_to_name = load_and_filter(
        ANNOTATIONS_VAL, SURVEILLANCE_CLASSES
    )
    analyze_filtered(val_filtered, id_to_name, title="val2017 после фильтрации")

    print("\n▶ Шаг 2: Разбивка val -> val / test")
    val_data, test_data = split_val_test(
        val_filtered, val_ratio=VAL_SPLIT, seed=RANDOM_SEED
    )

    print("\n▶ Шаг 3: Сохранение аннотаций")
    save_annotations(val_data,  f"{OUTPUT_DIR}/annotations_val.json")
    save_annotations(test_data, f"{OUTPUT_DIR}/annotations_test.json")

    if os.path.exists(ANNOTATIONS_TRAIN):
        print("\n▶ Фильтрация train2017 (найден файл аннотаций)")
        train_filtered, _ = load_and_filter(
            ANNOTATIONS_TRAIN, SURVEILLANCE_CLASSES
        )
        analyze_filtered(train_filtered, id_to_name, title="train2017 после фильтрации")
        save_annotations(train_filtered, f"{OUTPUT_DIR}/annotations_train.json")
    else:
        print(f"\n  train2017 не найден ({ANNOTATIONS_TRAIN}), пропускаем.")
        print("  Скачайте train2017 позже для полноценного обучения.")

    print("\n▶ Шаг 4: Демонстрация аугментаций")
    demo_augmentations(val_filtered, IMAGES_VAL_DIR, plots_dir)

    print("\n▶ Шаг 5: График разбивки val/test")
    plot_split_summary(val_data, test_data, id_to_name, plots_dir)

    print("\n" + "="*50)
    print("Готово! Результаты:")
    print(f"  Аннотации → {OUTPUT_DIR}/")
    print(f"  Графики   → {plots_dir}/")
    print("="*50)